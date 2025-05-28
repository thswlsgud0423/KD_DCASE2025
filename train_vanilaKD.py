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

class PLModule(pl.LightningModule):
    """
    PyTorch Lightning Module for training DCASE'25 models, supporting Teacher and Student (KD) modes.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # preprocessing pipeline
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.Resample(
                orig_freq=config.orig_sample_rate,
                new_freq=config.sample_rate
            ),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate, # number of samples per second
                n_fft=config.n_fft, # size of the FFT window
                win_length=config.window_length, 
                hop_length=config.hop_length, # overlapping length
                n_mels=config.n_mels,
                f_min=config.f_min,
                f_max=config.f_max
            )
        )

        self.mel_augment = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True), # whether to apply different masks to each example/channel in the batch
            torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)
        )
        # KD 
        if config.mode == 'teacher':
            print("Initializing in Teacher mode.")
            self.model = get_model(
                n_classes=config.n_classes,
                in_channels=config.in_channels,
                base_channels=config.base_channels,
                channels_multiplier=config.channels_multiplier,
                expansion_rate=config.expansion_rate
            )
            self.teacher_model = None # No teacher needed when training the teacher

        elif config.mode == 'student':
            print("Initializing in Student mode.")
            self.model = get_model(
                n_classes=config.n_classes,
                in_channels=config.in_channels,
                base_channels=config.student_base_channels,
                channels_multiplier=config.student_channels_multiplier,
                expansion_rate=config.student_expansion_rate

            )

            print(f"Loading teacher model from: {config.teacher_checkpoint_path}")
            if not config.teacher_checkpoint_path or not os.path.exists(config.teacher_checkpoint_path):
                raise ValueError(f"Teacher checkpoint path not found or not provided: {config.teacher_checkpoint_path}")
            
            # Instantiate teacher architecture (using original parameters)
            self.teacher_model = get_model(
                 n_classes=config.n_classes,
                 in_channels=config.in_channels,
                 base_channels=config.base_channels,
                 channels_multiplier=config.channels_multiplier,
                 expansion_rate=config.expansion_rate
            )
            checkpoint = torch.load(config.teacher_checkpoint_path, map_location='cpu')

            # remove "model." and keep the rest part of the tensors (expect mel)
            # why? cuz student will have its own mel
            teacher_state_dict = {}
            pl_model_prefix = "model."
            for k, v in checkpoint['state_dict'].items():
                if k.startswith(pl_model_prefix):
                    teacher_state_dict[k[len(pl_model_prefix):]] = v
                elif not any(prefix in k for prefix in ["mel", "mel_augment."]):
                    teacher_state_dict[k] = v

            self.teacher_model.load_state_dict(teacher_state_dict)
            print("Teacher model weights loaded successfully :>")

            # freeze teacher model parameters and set to eval mode
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            print("Teacher model frozen.")

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

        # Containers (using lists is fine for PL >= 1.x)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    
    def mel_forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x)
        if self.training:
            x = self.mel_augment(x)
        x = (x + 1e-5).log()
        return x

    def forward(self, x):
        x = self.mel_forward(x)
        x = self.model(x) # Always use self.model (which is student if mode='student')
        return x
    
    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: optimizer and learning rate scheduler
        """

        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        # Learning rate scheduler (cosine with warmup)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: loss to update model parameters
        """

        x, _, labels, _, _ = train_batch
        mel_spec = self.mel_forward(x) # Mel conversion + augmentation if training


        # still dont know what mixstyle exactly does....
        if self.config.mixstyle_p > 0:
            mel_spec = mixstyle(mel_spec, self.config.mixstyle_p, self.config.mixstyle_alpha)
        
        # Forward pass through the primary model
        student_logits = self.model(mel_spec)

        # calculate hard loss
        loss_ce = F.cross_entropy(student_logits, labels)

        # KD Loss
        if self.config.mode == 'student' and self.teacher_model is not None:
            # Ensure teacher is on the same device as input
            self.teacher_model.to(mel_spec.device)


            # To do: Check if teacher is on eval mode.
            assert not self.teacher_model.training, "Teacher model is not in evaluation mode!"
            
            # To do: Check if teacher works better with augmented mel spec or with non-augmented mel spec
            with torch.no_grad():
                teacher_logits = self.teacher_model(mel_spec)
            """
            calculate KD loss (KL Divergence)
            soften probabilities with temperature
            -> so that student can learn from teacher's "dark knowledge" 
            Dividing logits by T > 1 before softmax makes the distribution smoother (less peaked), 
            preserving more of this relative information compared to a standard (T=1) softmax or a one-hot label. Provides a richer, "softer" target for the student
            """

            soft_targets = F.log_softmax(teacher_logits / self.config.kd_temperature, dim=-1) 
            soft_prob = F.log_softmax(student_logits / self.config.kd_temperature, dim=-1)

            # calculate KL Divergence loss between the student's and teacher's softened distributions.
            # student learns to mimic the teacher's output distribution, not just the hard labels
            loss_kd = F.kl_div(soft_prob, soft_targets.detach(), reduction='batchmean', log_target=True) * (self.config.kd_temperature ** 2)

            loss = self.config.kd_alpha * loss_kd + (1.0 - self.config.kd_alpha) * loss_ce

            self.log("train/loss_ce", loss_ce.detach().cpu(), on_step=True, on_epoch=False, prog_bar=False)
            self.log("train/loss_kd", loss_kd.detach().cpu(), on_step=True, on_epoch=False, prog_bar=False)
        else:
            loss = loss_ce

        # Log learning rate, epoch, and total loss
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=False)
        self.log("epoch", self.current_epoch, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, _ = val_batch
        y_hat = self.forward(x)

        # Compute loss per sample
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct,
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }

        # I am a bit confused with these steps. We are computing loss and accuracy per ? but why do we need this?

        # Initialize per-device stats
        for d in self.device_ids:
            results[f"devloss.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devcnt.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devn_correct.{d}"] = torch.as_tensor(0., device=self.device)

        # Accumulate device-wise stats
        for i, d in enumerate(devices):
            results[f"devloss.{d}"] += samples_loss[i]
            results[f"devcnt.{d}"] += 1
            results[f"devn_correct.{d}"] += n_correct_per_sample[i]

        # Initialize per-label stats
        for lbl in self.label_ids:
            results[f"lblloss.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lblcnt.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lbln_correct.{lbl}"] = torch.as_tensor(0., device=self.device)

        # Accumulate label-wise stats
        for i, lbl_index in enumerate(labels):
            lbl_name = self.label_ids[lbl_index]
            results[f"lblloss.{lbl_name}"] += samples_loss[i]
            results[f"lbln_correct.{lbl_name}"] += n_correct_per_sample[i]
            results[f"lblcnt.{lbl_name}"] += 1

        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch. Aggregates per-device/per-label stats and logs them.
        [{'loss': tensor(0.2), 'n_correct': tensor(8), 'n_pred': tensor(10)},  ...]
        """
        # Flatten the outputs into a dict of lists
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k, v in step_output.items():
                outputs[k].append(v)

        # Stack each list of tensors
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        # Compute overall metrics
        avg_loss = outputs["loss"].mean()
        acc = outputs["n_correct"].sum() / outputs["n_pred"].sum()

        logs = {"acc": acc, "loss": avg_loss}

        # Per-device stats
        for d in self.device_ids:
            dev_loss = outputs[f"devloss.{d}"].sum()
            dev_cnt = outputs[f"devcnt.{d}"].sum()
            dev_correct = outputs[f"devn_correct.{d}"].sum()
            logs[f"loss.{d}"] = dev_loss / dev_cnt
            logs[f"acc.{d}"] = dev_correct / dev_cnt
            logs[f"cnt.{d}"] = dev_cnt

            # Group stats
            group_name = self.device_groups[d]
            logs[f"acc.{group_name}"] = logs.get(f"acc.{group_name}", 0.) + dev_correct
            logs[f"count.{group_name}"] = logs.get(f"count.{group_name}", 0.) + dev_cnt
            logs[f"lloss.{group_name}"] = logs.get(f"lloss.{group_name}", 0.) + dev_loss

        # Reduce group stats
        for grp in set(self.device_groups.values()):
            logs[f"acc.{grp}"] = logs[f"acc.{grp}"] / logs[f"count.{grp}"]
            logs[f"lloss.{grp}"] = logs[f"lloss.{grp}"] / logs[f"count.{grp}"]

        # Per-label stats
        for lbl in self.label_ids:
            lbl_loss = outputs[f"lblloss.{lbl}"].sum()
            lbl_cnt = outputs[f"lblcnt.{lbl}"].sum()
            lbl_correct = outputs[f"lbln_correct.{lbl}"].sum()

            logs[f"loss.{lbl}"] = lbl_loss / lbl_cnt
            logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt
            logs[f"cnt.{lbl}"] = lbl_cnt.float()

        # Compute macro-average accuracy over all labels
        logs["macro_avg_acc"] = torch.mean(torch.stack([logs[f"acc.{l}"] for l in self.label_ids]))

        # Log everything with 'val/' prefix
        self.log_dict({f"val/{k}": v for k, v in logs.items()})
        self.validation_step_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, _ = test_batch

        # For memory constraints, switch model to half-precision

        # what does "half" do?
        self.model.half()
        x = self.mel_forward(x)
        x = x.half()

        y_hat = self.model(x)
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

    def on_test_epoch_end(self):
        # Flatten the outputs
        outputs = {k: [] for k in self.test_step_outputs[0]}
        for step_output in self.test_step_outputs:
            for k, v in step_output.items():
                outputs[k].append(v)

        # Stack each list of tensors
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs["loss"].mean()
        acc = outputs["n_correct"].sum() / outputs["n_pred"].sum()
        logs = {"acc": acc, "loss": avg_loss}

        # Device-level stats
        for d in self.device_ids:
            dev_loss = outputs[f"devloss.{d}"].sum()
            dev_cnt = outputs[f"devcnt.{d}"].sum()
            dev_correct = outputs[f"devn_correct.{d}"].sum()
            logs[f"loss.{d}"] = dev_loss / dev_cnt
            logs[f"acc.{d}"] = dev_correct / dev_cnt
            logs[f"cnt.{d}"] = dev_cnt

            # Device groups
            grp = self.device_groups[d]
            logs[f"acc.{grp}"] = logs.get(f"acc.{grp}", 0.) + dev_correct
            logs[f"count.{grp}"] = logs.get(f"count.{grp}", 0.) + dev_cnt
            logs[f"lloss.{grp}"] = logs.get(f"lloss.{grp}", 0.) + dev_loss

        # Group-level stats
        for grp in set(self.device_groups.values()):
            logs[f"acc.{grp}"] = logs[f"acc.{grp}"] / logs[f"count.{grp}"]
            logs[f"lloss.{grp}"] = logs[f"lloss.{grp}"] / logs[f"count.{grp}"]

        # Label-level stats
        for lbl in self.label_ids:
            lbl_loss = outputs[f"lblloss.{lbl}"].sum()
            lbl_cnt = outputs[f"lblcnt.{lbl}"].sum()
            lbl_correct = outputs[f"lbln_correct.{lbl}"].sum()
            logs[f"loss.{lbl}"] = lbl_loss / lbl_cnt
            logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt
            logs[f"cnt.{lbl}"] = lbl_cnt

        # Macro-average accuracy over all labels
        logs["macro_avg_acc"] = torch.mean(torch.stack([logs[f"acc.{l}"] for l in self.label_ids]))

        self.log_dict({f"test/{k}": v for k, v in logs.items()})
        self.test_step_outputs.clear()


def train(config):
    """
        Main training loop using PyTorch Lightning.

        Args:
            config: Namespace or dictionary containing all experiment hyperparameters
                    (subset size, number of epochs, LR, etc.).
        """
    
    experiment_name = f"{config.experiment_name}_{config.mode}"
    if config.mode == 'student':
         experiment_name += f"_a{config.kd_alpha}_t{config.kd_temperature}"

    # Configure W&B logger
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="Baseline System for DCASE'25 Task 1.",
        tags=["DCASE25"],
        config=config,  # logs all hyperparameters
        name=config.experiment_name
    )

    # Dataloader for training
    assert config.subset == 25, (
        "In DCASE`25 subset must be fixed to 25%."
    )

    roll_samples = config.orig_sample_rate * config.roll_sec
    train_ds = get_training_set(config.subset, device=None, roll=roll_samples)
    train_dl = DataLoader(
        dataset=train_ds,
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers, # higher num_workers faster data loading, but more RAM/CPU usage
        batch_size=config.batch_size,
        shuffle=True
    )

    # Dataloader for testing
    test_ds = get_test_set(device=None)
    test_dl = DataLoader(
        dataset=test_ds,
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size
    )

    # Create PyTorch Lightning module
    pl_module = PLModule(config)

    # Compute model complexity (MACs, parameters) and log to W&B
    sample = next(iter(test_dl))[0][0].unsqueeze(0)  # Single sample
    shape = pl_module.mel_forward(sample).size()
    macs, params_bytes = complexity.get_torch_macs_memory(pl_module.model, input_size=shape)
    wandb_logger.experiment.config["MACs"] = macs
    wandb_logger.experiment.config["Parameters_Bytes"] = params_bytes

    # create the pytorch lightening trainer
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1,
                         precision=config.precision,
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)])

    # Fit (train + validation)
    trainer.fit(pl_module, train_dl, test_dl)

    # Final test (using the same test_dl here)
    trainer.test(ckpt_path="last", dataloaders=test_dl)

    # Finish W&B run
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE 25 Task 1 Training with KD')

    # General arguments
    parser.add_argument("--project_name", type=str, default="DCASE25_Task1_KD") # Teacherwith64,2.5,2.1
    parser.add_argument("--experiment_name", type=str, default="Baseline_KD", help="Base name for the experiment (mode will be appended)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="32", choices=["16-mixed", "bf16-mixed", "32", "64"]) # Updated choices
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5) # Check more often maybe
    parser.add_argument("--orig_sample_rate", type=int, default=44100)
    parser.add_argument("--subset", type=int, default=25, choices=[25], help="DCASE'25 requires 25%")

    # Mode Selection
    parser.add_argument("--mode", type=str, required=True, choices=['teacher', 'student'], help="Training mode: 'teacher' or 'student'")

    # To do: make a plot that visualize teacher models' performances
    # x-axis: parameters
    # y-axis: accuracy 
    # Try with 150 epochs but 3-4 teacher hyperparameters

    # Teacher Model hyperparameters
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--channels_multiplier", type=float, default=2.5) # controls how rapidly the number of channels increases as the data progresses through deeper layers or stages of the network
    parser.add_argument("--expansion_rate", type=float, default=2.8) # determines the expansion rate in inverted bottleneck blocks

    # NO CHANGE HERE
    # Student Model hyperparameters (only used if mode=='student') 
    parser.add_argument("--student_base_channels", type=int, default=32)
    parser.add_argument("--student_channels_multiplier", type=float, default=1.8)
    parser.add_argument("--student_expansion_rate", type=float, default=2.1)

    # Knowledge Distillation hyperparameters (only used if mode=='student') ---
    parser.add_argument("--teacher_checkpoint_path", type=str, default=None)
    parser.add_argument("--kd_temperature", type=float, default=1.0) # start with 1.0 and try other numbers if it doesnt perform well
    parser.add_argument("--kd_alpha", type=float, default=0.5) # start with 0.5 and try higher values later

    # n classes remains the same for both teacher and student
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--in_channels", type=int, default=1)

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
    parser.add_argument("--f_max", type=int, default=None)
    args = parser.parse_args()

    # --- Start Training ---
    train(args)