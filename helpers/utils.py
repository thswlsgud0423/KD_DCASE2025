import numpy as np
import torch
from torch.distributions.beta import Beta

import random
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
from PIL import Image
from torchvision.transforms import functional as TF

def mixstyle(x, p=0.4, alpha=0.3, eps=1e-6):
    if np.random.rand() > p:
        return x
    batch_size = x.size(0)

    # frequency-wise statistics
    f_mu = x.mean(dim=[1, 3], keepdim=True)
    f_var = x.var(dim=[1, 3], keepdim=True)

    f_sig = (f_var + eps).sqrt()  # compute instance standard deviation
    f_mu, f_sig = f_mu.detach(), f_sig.detach()  # block gradients
    x_normed = (x - f_mu) / f_sig  # normalize input
    lmda = Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device)  # sample instance-wise convex weights
    perm = torch.randperm(batch_size).to(x.device)  # generate shuffling indices
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]  # shuffling
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
    sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)  # generate mixed standard deviation
    x = x_normed * sig_mix + mu_mix  # denormalize input using the mixed frequency statistics
    return x


class Mixup:

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        
    def __call__(self, batch):

        data, targets = batch
        batch_size = data.size(0)
        
        # Generate lambda from beta distribution -> decide how much we want to mix
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        # Create random permutation of indices
        indices = torch.randperm(batch_size, device=data.device)
        
        # Mix the data
        mixed_data = lam * data + (1 - lam) * data[indices]
        mixed_targets = (targets, targets[indices], lam)
        
        return mixed_data, mixed_targets


class MixupLoss:
    def __init__(self, reduction='mean'):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        
    def __call__(self, preds, targets):

        targets1, targets2, lam = targets
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(preds, targets2)


# reference: https://github.com/pyyush/SpecAugment/blob/master/augment.py
class SpecAugment:
    """
    Implementation of SpecAugment for audio spectrograms.
    
    Policy parameters:
    -----------------------------------------
    Policy | W  | F  | m_F |  T  |  p  | m_T
    -----------------------------------------
    None   |  0 |  0 |  -  |  0  |  -  |  -
    -----------------------------------------
    LB     | 80 | 27 |  1  | 100 | 1.0 | 1
    -----------------------------------------
    LD     | 80 | 27 |  2  | 100 | 1.0 | 2
    -----------------------------------------
    SM     | 40 | 15 |  2  |  70 | 0.2 | 2
    -----------------------------------------
    SS     | 40 | 27 |  2  |  70 | 0.2 | 2
    -----------------------------------------
    
    LB: LibriSpeech basic
    LD: LibriSpeech double
    SM: Switchboard mild
    SS: Switchboard strong
    
    W:   Time Warp parameter
    F:   Frequency Mask parameter
    m_F: Number of Frequency masks
    T:   Time Mask parameter
    p:   Parameter for calculating upper bound for time mask
    m_T: Number of time masks
    """
    
    def __init__(self, policy='LB', device=None):
        self.policy = policy
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set policy parameters
        if self.policy == 'LB':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 1, 100, 1.0, 1
        elif self.policy == 'LD':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 2, 100, 1.0, 2
        elif self.policy == 'SM':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 15, 2, 70, 0.2, 2
        elif self.policy == 'SS':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 27, 2, 70, 0.2, 2
        else:
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 0, 0, 0, 0, 0, 0
            
    def time_warp(self, spec):
        """
        Apply time warping to the spectrogram.
        """
        if self.W <= 0:
            return spec

        # Correctly unpack the 4 dimensions (assuming this fix was applied from the previous error)
        batch_size, channels, n_mels, time_steps = spec.shape

        # Create a tensor to store the result
        warped_spec = torch.empty_like(spec)

        for i in range(batch_size):
            # Extract one sample, shape [channels, n_mels, time_steps]
            sample_np = spec[i].cpu().numpy()

            # Loop over each channel for this sample
            for c in range(channels):
                # Get one channel, shape [n_mels, time_steps]
                channel_np = sample_np[c]

                # --- Convert 2D NumPy array channel to PIL Image ---
                # Normalize channel data to [0, 1]
                channel_min = channel_np.min()
                channel_max = channel_np.max()
                if (channel_max - channel_min) == 0: # Avoid division by zero if channel is constant
                    channel_np_norm = np.zeros_like(channel_np)
                else:
                    channel_np_norm = (channel_np - channel_min) / (channel_max - channel_min)

                # Convert to uint8 (0-255) for PIL
                channel_np_uint8 = (channel_np_norm * 255).astype(np.uint8)

                # Create PIL image from the 2D (height, width) array - this is a grayscale image
                img_pil = Image.fromarray(channel_np_uint8) # This should work

                # --- Apply Warp using PIL ---
                # PIL works with (width, height) -> so (time_steps, n_mels) for a spectogram channel
                width, height = img_pil.size # width is time_steps, height is n_mels

                # Determine warping point and distance (warping is typically on the time axis, which is 'width' in PIL terms here)
                center_frame = width // 2 # Center time step
                distance = random.randint(-self.W, self.W) # Random shift

                # Define source and destination points for perspective transform (applies to the 2D channel image)
                src_points = [(center_frame, 0), (center_frame, height), (0, height//2)] # (time, freq) points
                dst_points = [(center_frame + distance, 0), (center_frame + distance, height), (0, height//2)]

                # Get and apply the transform coefficients
                coeffs = TF._get_perspective_coeffs(src_points, dst_points)
                warped_img_pil = img_pil.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

                warped_channel_np = np.array(warped_img_pil).astype(np.float32) / 255.0
                
                warped_spec[i, c, :, :] = torch.from_numpy(warped_channel_np).to(spec.device)

        return warped_spec
            
    def freq_mask(self, spec):
        """
        Apply frequency masking to the spectrogram.
        """
        if self.F <= 0 or self.m_F <= 0:
            return spec
            
        masked_spec = spec.clone()
        _, _, n_mels, _ = spec.shape # Unpack 4 dimensions
      
        for _ in range(self.m_F):
            f = int(np.random.uniform(0, self.F))  # Mask width
            f0 = random.randint(0, n_mels - f)  # Mask start
            masked_spec[:, f0:f0+f, :] = 0
            
        return masked_spec
        
    def time_mask(self, spec):
        """
        Apply time masking to the spectrogram.
        """
        masked_spec = spec.clone()
        _, _, _, time_steps = spec.shape # Unpack 4 dimensions
 
        masked_spec = spec.clone()
        _, _, _, time_steps = spec.shape

        for _ in range(self.m_T):
            t = int(np.random.uniform(0, self.T * self.p))
            t = min(t, int(time_steps * self.p))
            t0 = random.randint(0, time_steps - t)
            masked_spec[:, :, :, t0:t0+t] = 0
        return masked_spec
        
    def __call__(self, spec):
        """
        Apply SpecAugment to a batch of spectrograms.
        """
        # Make sure spec is on the correct device
        spec = spec.to(self.device)
        
        # Apply augmentations in sequence
        augmented_spec = self.time_warp(spec)
        augmented_spec = self.freq_mask(augmented_spec)
        augmented_spec = self.time_mask(augmented_spec)
        
        return augmented_spec

# reference: https://github.com/noisemix/noisemix
class NoiseMix:
    """
    Mix clean audio with background noise from other classes.
    """
    def __init__(self, noise_dataset, snr_range=(5, 20), p=0.5):
        self.noise_dataset = noise_dataset
        self.snr_min, self.snr_max = snr_range
        self.p = p
        
    def __call__(self, audio):
        """
        Apply noise mixing to audio.
        """
        if random.random() > self.p:
            return audio
            
        batch_size = audio.shape[0]
        mixed_audio = audio.clone()
        
        for i in range(batch_size):
            # Randomly select a noise sample
            noise_idx = random.randint(0, len(self.noise_dataset) - 1)
            noise, _ = self.noise_dataset[noise_idx]
            
            # Ensure noise has the same length as audio
            if noise.shape[-1] > audio.shape[-1]:
                start = random.randint(0, noise.shape[-1] - audio.shape[-1])
                noise = noise[..., start:start + audio.shape[-1]]
            else:
                # Pad noise if it's shorter
                padding = audio.shape[-1] - noise.shape[-1]
                noise = F.pad(noise, (0, padding))
                
            # Calculate signal and noise power
            signal_power = torch.mean(audio[i] ** 2)
            noise_power = torch.mean(noise ** 2)
            
            # Calculate desired noise power for target SNR
            target_snr = random.uniform(self.snr_min, self.snr_max)
            target_noise_power = signal_power / (10 ** (target_snr / 10))
            
            # Scale noise to achieve target SNR
            scaling_factor = torch.sqrt(target_noise_power / (noise_power + 1e-6))
            scaled_noise = noise * scaling_factor
            
            # Mix signal with noise
            mixed_audio[i] = audio[i] + scaled_noise
            
            # Normalize to prevent clipping
            max_val = torch.max(torch.abs(mixed_audio[i]))
            if max_val > 1.0:
                mixed_audio[i] = mixed_audio[i] / max_val
                
        return mixed_audio

# reference: https://github.com/clovaai/CutMix-PyTorch
class FeatureCutMix:
    """
    Implementation of CutMix for audio spectrograms.
    
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch):
        """
        Apply CutMix to a batch of spectrograms and their labels.
        """
        data, targets = batch
        batch_size, channels, height, width = data.shape
        
        # Generate mixed sample
        indices = torch.randperm(batch_size, device=data.device)
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get random box dimensions
        cut_height = int(height * np.sqrt(1.0 - lam))
        cut_width = int(width * np.sqrt(1.0 - lam))
        
        # Get random box position
        cy = np.random.randint(0, height)
        cx = np.random.randint(0, width)
        
        # Calculate box boundaries
        y1 = np.clip(cy - cut_height // 2, 0, height)
        y2 = np.clip(cy + cut_height // 2, 0, height)
        x1 = np.clip(cx - cut_width // 2, 0, width)
        x2 = np.clip(cx + cut_width // 2, 0, width)
        
        # Create mixed data
        mixed_data = data.clone()
        mixed_data[:, :, y1:y2, x1:x2] = shuffled_data[:, :, y1:y2, x1:x2]
        
        # Adjust lambda to reflect the actual area ratio
        actual_box_area = (y2 - y1) * (x2 - x1)
        actual_lam = 1.0 - (actual_box_area / (height * width))
        
        mixed_targets = (targets, shuffled_targets, actual_lam)
        
        return mixed_data, mixed_targets

# basically same as mix up
class CutMixCriterion:
    """
    Loss function for CutMix training.
    """
    def __init__(self, reduction='mean'):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        
    def __call__(self, preds, targets):
        """
        Compute the mixed loss.
        """
        targets1, targets2, lam = targets
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(preds, targets2)