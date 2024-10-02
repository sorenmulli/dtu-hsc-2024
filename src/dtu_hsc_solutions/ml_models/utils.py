import torch
import os
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from torch import stft


def create_data_path(data_path: Path, task: int, level: int, ir: bool = False):
    # Return the path to the data directory for the given task and level
    if ir:
        return os.path.join(data_path, "output", "linear-filter", f"Task_{task}_Level_{level}")
    return os.path.join(data_path,f"Task_{task}_Level_{level}")


def create_signal_path(data_path: Path, task: int, level: int):
    # Return the path to the recorded signal file
    return os.path.join(create_data_path(data_path, task, level),"Recorded", f"task_{task}_level_{level}_recorded_001.wav")

def load_dccrnet_model():
    from asteroid.models import BaseModel
    model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
    return model

def load_voicefixer_model():
    from voicefixer import VoiceFixer
    model = VoiceFixer()
    return model

def si_sdr(estimated_signal, target_signal, eps=1e-8):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
    - estimated_signal: Tensor of shape (batch_size, signal_length)
    - target_signal: Tensor of shape (batch_size, signal_length)
    - eps: Small value to avoid division by zero
    
    Returns:
    - si_sdr_loss: Tensor of shape (batch_size,) containing SI-SDR values
    """
    
    # Reshape if necessary (flatten if not already batch x length)
    estimated_signal = estimated_signal.view(estimated_signal.size(0), -1)
    target_signal = target_signal.view(target_signal.size(0), -1)
    
    # Compute scaling factor alpha
    dot_product = torch.sum(estimated_signal * target_signal, dim=1, keepdim=True)
    target_energy = torch.sum(target_signal ** 2, dim=1, keepdim=True)
    alpha = dot_product / (target_energy + eps)
    
    # Compute the projection of the estimated signal onto the target signal
    projection = alpha * target_signal
    
    # Compute the noise (difference between estimated signal and projection)
    noise = estimated_signal - projection
    
    # Compute the SI-SDR
    si_sdr_value = 10 * torch.log10((torch.sum(projection ** 2, dim=1) + eps) /
                                    (torch.sum(noise ** 2, dim=1) + eps))
    
    # Since this is a loss, return the negative SI-SDR (minimizing loss is maximizing SI-SDR)
    return -si_sdr_value.mean()

def stft_magnitude(signal, n_fft=512, hop_length=None, win_length=None):
    # Reshape the signal to (batch_size * channels, samples) if necessary
    if len(signal.shape) == 3:  # [batch_size, channels, samples]
        batch_size, channels, samples = signal.shape
        signal = signal.view(batch_size * channels, samples)  # Flatten batch and channels

    # Compute the STFT (Short-Time Fourier Transform)
    stft_result = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)

    # Return the magnitude spectrogram (absolute value)
    return torch.abs(stft_result).view(batch_size, channels, -1, stft_result.size(-1))  # Reshape back to [batch_size, channels, freq_bins, time_frames]

def spectral_convergence_loss(clean_signal, predicted_signal, n_fft=512, hop_length=None, win_length=None):
    # Compute the magnitude spectrograms
    S_clean = stft_magnitude(clean_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S_pred = stft_magnitude(predicted_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # Compute the Frobenius norms
    sc_loss = torch.norm(S_pred - S_clean, p='fro') / torch.norm(S_clean, p='fro')
    
    return sc_loss

def combined_loss(clean_signal, predicted_signal, n_fft=512, hop_length=None, win_length=None):
    # Compute spectral convergence loss
    sc_loss = spectral_convergence_loss(clean_signal, predicted_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # Compute SI-SNR loss (using the existing si_snr_loss function)
    si_snr_loss_value = si_sdr(clean_signal, predicted_signal)
    
    # Combine the losses (with a weighting factor, e.g., 0.5)
    combined_loss_value = 0.5 * sc_loss + 0.5 * si_snr_loss_value
    
    return combined_loss_value