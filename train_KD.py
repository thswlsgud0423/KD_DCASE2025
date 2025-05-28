import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import transformers
import wandb
import os

from dataset.dcase25 import get_training_set, get_test_set
from helpers.init import worker_init_fn
from helpers.utils import mixstyle
from helpers import complexity
from models.net import get_model
from kd_handler import KnowledgeDistillationHandler # <-- Import the new handler

class PLModule(pl.LightningModule):
    """
    PyTorch Lightning Module for training DCASE'25 models.
    Uses KnowledgeDistillationHandler for KD when in 'student' mode.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.kd_handler = None

        # -------- Preprocessing Pipeline --------
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.Resample(
                orig_freq=config.orig_sample_rate,
                new_freq=config.sample_rate
            ),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.window_length,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                f_min=config.f_min,
                f_max=config.f_max
            )
        )
        self.mel_augment = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True),
            torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)
        )

        # -------- Model Initialization --------
        if config.mode == 'teacher':
            print("Initializing in Teacher mode.")
            self.model = get_model(
                n_classes=config.n_classes,
                in_channels=config.in_channels,
                base_channels=config.base_channels,
                channels_multiplier=config.channels_multiplier,
                expansion_rate=config.expansion_rate
            )
        elif config.mode == 'student':
            print("Initializing in Student mode.")
            # Initialize the student model architecture
            self.model = get_model(
                n_classes=config.n_classes,
                in_channels=config.in_channels,
                # Use specific student parameters from config
                base_channels=config.student_base_channels,
                channels_multiplier=config.student_channels_multiplier,
                expansion_rate=config.student_expansion_rate
            )
            
            effective_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.kd_handler = KnowledgeDistillationHandler(config, effective_device)
            print(f"Knowledge Distillation Handler initialized with teacher on {effective_device}.")

        else:
            raise ValueError(f"Invalid mode: {config.mode}. Choose 'teacher' or 'student'.")

        # -------- Device/Label Definitions --------
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = [
            'airport', 'bus', 'metro', 'metro_station', 'park',
            'public_square', 'shopping_mall', 'street_pedestrian',
            'street_traffic', 'tram'
        ]
        self.device_groups = {
            'a': "real", 'b': "real", 'c': "real",
            's1': "seen", 's2': "seen", 's3': "seen",
            's4': "unseen", 's5': "unseen", 's6': "unseen"
        }

        # Containers
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def setup(self, stage=None):
         if self.config.mode == 'student' and self.kd_handler:
             # Ensure teacher model is on the same device as the student
             self.kd_handler.teacher_model.to(self.device)
             print(f"Moved teacher model to PL device: {self.device} during setup.")


    def mel_forward(self, x):
        x = self.mel(x)
        if self.training:
            x = self.mel_augment(x)
        x = (x + 1e-5).log()
        return x

    def forward(self, x):
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def training_step(self, train_batch, batch_idx):
        x, _, labels, _, _ = train_batch
        mel_spec = self.mel_forward(x)

        if self.config.mixstyle_p > 0:
            mel_spec = mixstyle(mel_spec, self.config.mixstyle_p, self.config.mixstyle_alpha)

        # Forward pass
        logits = self.model(mel_spec)

        # calculate standard cross-entropy loss
        loss_ce = F.cross_entropy(logits, labels)

        # --- KD Loss Calculation (if in student mode) ---
        if self.config.mode == 'student' and self.kd_handler is not None:
            # calculate KD loss using the handler
            loss_kd = self.kd_handler.calculate_kd_loss(logits, mel_spec)

            # combine losses using the handler's method
            loss = self.kd_handler.combine_losses(loss_ce, loss_kd)

            # log individual losses
            self.log("train/loss_ce", loss_ce.detach().cpu(), on_step=True, on_epoch=False, prog_bar=False)
            self.log("train/loss_kd", loss_kd.detach().cpu(), on_step=True, on_epoch=False, prog_bar=False)
        else:
            loss = loss_ce

        # Log learning rate, epoch, and total loss
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=False)
        self.log("epoch", self.current_epoch, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True) # Log combined loss

        return loss

    # --- Validation Step ---
    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, _ = val_batch
        y_hat = self.forward(x) # Uses self.model (student or teacher)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct,
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }
        # Initialize and accumulate per-device stats
        for d in self.device_ids:
            results[f"devloss.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devcnt.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devn_correct.{d}"] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(devices):
            results[f"devloss.{d}"] += samples_loss[i]
            results[f"devcnt.{d}"] += 1
            results[f"devn_correct.{d}"] += n_correct_per_sample[i]
        # Initialize and accumulate per-label stats
        for lbl in self.label_ids:
             results[f"lblloss.{lbl}"] = torch.as_tensor(0., device=self.device)
             results[f"lblcnt.{lbl}"] = torch.as_tensor(0., device=self.device)
             results[f"lbln_correct.{lbl}"] = torch.as_tensor(0., device=self.device)
        for i, lbl_index in enumerate(labels):
             lbl_name = self.label_ids[lbl_index]
             results[f"lblloss.{lbl_name}"] += samples_loss[i]
             results[f"lbln_correct.{lbl_name}"] += n_correct_per_sample[i]
             results[f"lblcnt.{lbl_name}"] += 1

        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)


    # --- Validation Epoch End ---
    def on_validation_epoch_end(self):
         if not self.validation_step_outputs:
             return
         outputs = {}
         keys = self.validation_step_outputs[0].keys()
         for k in keys:
             outputs[k] = torch.stack([step_output[k] for step_output in self.validation_step_outputs if k in step_output])

         avg_loss = outputs["loss"].mean()
         acc = outputs["n_correct"].sum() / outputs["n_pred"].sum()
         logs = {"acc": acc, "loss": avg_loss}
         # Per-device stats aggregation
         for d in self.device_ids:
             dev_loss = outputs[f"devloss.{d}"].sum()
             dev_cnt = outputs[f"devcnt.{d}"].sum()
             dev_correct = outputs[f"devn_correct.{d}"].sum()
             logs[f"loss.{d}"] = dev_loss / dev_cnt if dev_cnt > 0 else torch.tensor(0.0)
             logs[f"acc.{d}"] = dev_correct / dev_cnt if dev_cnt > 0 else torch.tensor(0.0)
             logs[f"cnt.{d}"] = dev_cnt
             # Group stats
             group_name = self.device_groups[d]
             logs[f"acc.{group_name}"] = logs.get(f"acc.{group_name}", 0.) + dev_correct
             logs[f"count.{group_name}"] = logs.get(f"count.{group_name}", 0.) + dev_cnt
             logs[f"lloss.{group_name}"] = logs.get(f"lloss.{group_name}", 0.) + dev_loss
         # Reduce group stats
         for grp in set(self.device_groups.values()):
             count_grp = logs[f"count.{grp}"]
             logs[f"acc.{grp}"] = logs[f"acc.{grp}"] / count_grp if count_grp > 0 else torch.tensor(0.0)
             logs[f"lloss.{grp}"] = logs[f"lloss.{grp}"] / count_grp if count_grp > 0 else torch.tensor(0.0)
         # Per-label stats aggregatio
         for lbl in self.label_ids:
             lbl_loss = outputs[f"lblloss.{lbl}"].sum()
             lbl_cnt = outputs[f"lblcnt.{lbl}"].sum()
             lbl_correct = outputs[f"lbln_correct.{lbl}"].sum()
             logs[f"loss.{lbl}"] = lbl_loss / lbl_cnt if lbl_cnt > 0 else torch.tensor(0.0)
             logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt if lbl_cnt > 0 else torch.tensor(0.0)
             logs[f"cnt.{lbl}"] = lbl_cnt.float()
         # Compute macro-average accuracy
         label_accs = [logs[f"acc.{l}"] for l in self.label_ids if logs[f"cnt.{l}"] > 0]
         if label_accs:
             logs["macro_avg_acc"] = torch.mean(torch.stack(label_accs))
         else:
             logs["macro_avg_acc"] = torch.tensor(0.0)


         self.log_dict({f"val/{k}": v for k, v in logs.items()})
         self.validation_step_outputs.clear()


    # --- Test Step ---
    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, _ = test_batch
        y_hat = self.forward(x) # Uses self.model
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct,
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }
        # Per-device stats
        for d in self.device_ids:
             results[f"devloss.{d}"] = torch.as_tensor(0., device=self.device)
             results[f"devcnt.{d}"] = torch.as_tensor(0., device=self.device)
             results[f"devn_correct.{d}"] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(devices):
             results[f"devloss.{d}"] += samples_loss[i]
             results[f"devn_correct.{d}"] += n_correct_per_sample[i]
             results[f"devcnt.{d}"] += 1
        # Per-label stats
        for lbl in self.label_ids:
            results[f"lblloss.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lblcnt.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lbln_correct.{lbl}"] = torch.as_tensor(0., device=self.device)
        for i, lbl_index in enumerate(labels):
            lbl_name = self.label_ids[lbl_index]
            results[f"lblloss.{lbl_name}"] += samples_loss[i]
            results[f"lbln_correct.{lbl_name}"] += n_correct_per_sample[i]
            results[f"lblcnt.{lbl_name}"] += 1

        self.test_step_outputs.append({k: v.cpu() for k, v in results.items()})


    # --- Test Epoch End ---
    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        outputs = {}
        keys = self.test_step_outputs[0].keys()
        for k in keys:
             outputs[k] = torch.stack([step_output[k] for step_output in self.test_step_outputs if k in step_output])

        total_preds = outputs["n_pred"].sum()
        if total_preds == 0:
             print("Warning: Zero predictions in test epoch.")
             avg_loss = torch.tensor(0.0)
             acc = torch.tensor(0.0)
        else:
             avg_loss = outputs["loss"].mean()
             acc = outputs["n_correct"].sum() / total_preds

        logs = {"acc": acc, "loss": avg_loss}
        # Device-level stats aggregation
        for d in self.device_ids:
             dev_loss = outputs[f"devloss.{d}"].sum()
             dev_cnt = outputs[f"devcnt.{d}"].sum()
             dev_correct = outputs[f"devn_correct.{d}"].sum()
             logs[f"loss.{d}"] = dev_loss / dev_cnt if dev_cnt > 0 else torch.tensor(0.0)
             logs[f"acc.{d}"] = dev_correct / dev_cnt if dev_cnt > 0 else torch.tensor(0.0)
             logs[f"cnt.{d}"] = dev_cnt
             # Device groups
             grp = self.device_groups[d]
             logs[f"acc.{grp}"] = logs.get(f"acc.{grp}", 0.) + dev_correct
             logs[f"count.{grp}"] = logs.get(f"count.{grp}", 0.) + dev_cnt
             logs[f"lloss.{grp}"] = logs.get(f"lloss.{grp}", 0.) + dev_loss
        # Group-level stats aggregation
        for grp in set(self.device_groups.values()):
             count_grp = logs[f"count.{grp}"]
             logs[f"acc.{grp}"] = logs[f"acc.{grp}"] / count_grp if count_grp > 0 else torch.tensor(0.0)
             logs[f"lloss.{grp}"] = logs[f"lloss.{grp}"] / count_grp if count_grp > 0 else torch.tensor(0.0)
        # Label-level stats aggregation
        for lbl in self.label_ids:
             lbl_loss = outputs[f"lblloss.{lbl}"].sum()
             lbl_cnt = outputs[f"lblcnt.{lbl}"].sum()
             lbl_correct = outputs[f"lbln_correct.{lbl}"].sum()
             logs[f"loss.{lbl}"] = lbl_loss / lbl_cnt if lbl_cnt > 0 else torch.tensor(0.0)
             logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt if lbl_cnt > 0 else torch.tensor(0.0)
             logs[f"cnt.{lbl}"] = lbl_cnt
        # Macro-average accuracy
        label_accs = [logs[f"acc.{l}"] for l in self.label_ids if logs[f"cnt.{l}"] > 0]
        if label_accs:
             logs["macro_avg_acc"] = torch.mean(torch.stack(label_accs))
        else:
             logs["macro_avg_acc"] = torch.tensor(0.0)

        self.log_dict({f"test/{k}": v for k, v in logs.items()})
        self.test_step_outputs.clear()


def train(config):
    """
    Main training loop using PyTorch Lightning. Handles Teacher and Student modes.
    """
    # Add mode to experiment name for clarity
    experiment_name = f"{config.experiment_name}_{config.mode}"
    if config.mode == 'student':
        # Include KD params in name only for student mode
        experiment_name += f"_a{config.kd_alpha}_t{config.kd_temperature}"

    # Configure W&B logger
    wandb_logger = WandbLogger(
        project=config.project_name,
        tags=["DCASE25", config.mode.upper()],
        config=vars(config),
        name=experiment_name
    )

    # Dataloader for training
    assert config.subset == 25, "In DCASE`25 subset must be fixed to 25%."
    roll_samples = config.orig_sample_rate * config.roll_sec
    train_ds = get_training_set(config.subset, device=None, roll=roll_samples)
    train_dl = DataLoader(
        dataset=train_ds,
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True # Usually good for GPU training
    )

    # Dataloader for testing/validation (using test set for both here as in original)
    test_ds = get_test_set(device=None)
    test_dl = DataLoader(
        dataset=test_ds,
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True
    )

    # Create PyTorch Lightning module
    pl_module = PLModule(config)

    # --- Compute model complexity and log ---
    try:
        pl_module.cpu() # Move to CPU temporarily for calculation
        sample_input = torch.randn(1, config.orig_sample_rate * 5) # Example 5 sec audio
        mel_output_shape = pl_module.mel_forward(sample_input).size()
        print(f"Calculating complexity for input shape: {mel_output_shape}")

        # Calculate complexity for the *actual model being trained* (student or teacher)
        macs, params_bytes = complexity.get_torch_macs_memory(pl_module.model, input_size=mel_output_shape)
        print(f"Calculated MACs: {macs}, Params (Bytes): {params_bytes}")

        # Log to W&B config BEFORE trainer initialization if possible
        wandb_logger.experiment.config.update({
            "MACs": macs,
            "Parameters_Bytes": params_bytes,
            "Input_Shape_for_Complexity": list(mel_output_shape)
        })
        # wandb.config.update({"MACs": macs, "Parameters_Bytes": params_bytes}, allow_val_change=True)

    except Exception as e:
        print(f"Warning: Could not compute model complexity. Error: {e}")
        wandb_logger.experiment.config.update({"MACs": "Error", "Parameters_Bytes": "Error"})
        # wandb.config.update({"MACs": "Error", "Parameters_Bytes": "Error"}, allow_val_change=True)

    # Move back to GPU potential if needed before trainer starts (PL trainer will handle it anyway)
    if torch.cuda.is_available():
         pl_module.cuda()


    # Create the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=config.precision,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
        # Gradient clipping can sometimes help stabilize training, especially with KD
        # gradient_clip_val=1.0
    )

    # Fit (train + validation)
    # Pass both train and val dataloaders
    trainer.fit(pl_module, train_dl, test_dl) # Using test_dl for validation as per original code

    # Final test (using the best checkpoint or last if none is better)
    # Testing with 'last' checkpoint as per original code
    trainer.test(ckpt_path="last", dataloaders=test_dl)

    # Finish W&B run
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE 25 Task 1 Training with optional KD')

    # General arguments
    parser.add_argument("--project_name", type=str, default="DCASE25_Task1_KD_Refactor")
    parser.add_argument("--experiment_name", type=str, default="Baseline", help="Base name for the experiment (mode/KD params will be appended)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="32", choices=["16-mixed", "bf16-mixed", "32", "64"])
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5)
    parser.add_argument("--orig_sample_rate", type=int, default=44100)
    parser.add_argument("--subset", type=int, default=25, choices=[25], help="DCASE'25 requires 25%")

    # Mode Selection
    parser.add_argument("--mode", type=str, required=True, choices=['teacher', 'student'], help="Training mode: 'teacher' or 'student'")

    # --- Model Architecture Parameters ---
    # Teacher Model (used directly if mode='teacher', used for KD handler if mode='student')
    parser.add_argument("--base_channels", type=int, default=48, help="Base channels for TEACHER architecture")
    parser.add_argument("--channels_multiplier", type=float, default=2.0, help="Channel multiplier for TEACHER architecture")
    parser.add_argument("--expansion_rate", type=float, default=2.1, help="Expansion rate for TEACHER architecture")

    # Student Model (only used if mode='student')
    parser.add_argument("--student_base_channels", type=int, default=16, help="Base channels for STUDENT architecture")
    parser.add_argument("--student_channels_multiplier", type=float, default=1.5, help="Channel multiplier for STUDENT architecture")
    parser.add_argument("--student_expansion_rate", type=float, default=1.8, help="Expansion rate for STUDENT architecture")

    # Knowledge Distillation (only used if mode=='student')
    parser.add_argument("--teacher_checkpoint_path", type=str, default=None, help="Path to the TEACHER model checkpoint (REQUIRED if mode='student')")
    parser.add_argument("--kd_temperature", type=float, default=2.0, help="Temperature for KD softening")
    parser.add_argument("--kd_alpha", type=float, default=0.25, help="Weight balance between CE loss and KD loss (alpha * KD + (1-alpha) * CE)")

    # Shared model parameters
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--in_channels", type=int, default=1) # Input is single channel (mono audio -> mel)

    # Training hyperparameters
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mixstyle_p", type=float, default=0.4)
    parser.add_argument("--mixstyle_alpha", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--roll_sec", type=float, default=0.1)

    # Learning rate schedule
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--warmup_steps", type=int, default=2000)

    # Spectrogram parameters
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--window_length", type=int, default=3072)
    parser.add_argument("--hop_length", type=int, default=500)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--n_mels", type=int, default=256)
    parser.add_argument("--freqm", type=int, default=48)
    parser.add_argument("--timem", type=int, default=0)
    parser.add_argument("--f_min", type=int, default=0)
    parser.add_argument("--f_max", type=int, default=None) # Defaults to sample_rate / 2

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.mode == 'student' and not args.teacher_checkpoint_path:
        parser.error("--teacher_checkpoint_path is REQUIRED when --mode is 'student'")
    if args.mode == 'student' and not os.path.exists(args.teacher_checkpoint_path):
         parser.error(f"Teacher checkpoint path '{args.teacher_checkpoint_path}' does not exist.")


    # --- Start Training ---
    train(args)