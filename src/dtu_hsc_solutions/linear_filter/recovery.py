from pathlib import Path
import numpy as np

from dtu_hsc_data.audio import SAMPLE_RATE
from .utils import MODEL_NAME, spectral_subtraction_full_band

def design_inverse_filter(ir, N, regularization=0.01):
    H = np.fft.rfft(ir, n=N)

    # H_inv = np.conj(H) / (np.abs(H)**2 + regularization)
    epsilon = 1e-10
    mask = np.abs(H) < epsilon
    H[mask] = epsilon
    H_inv = 1 / H
    return H_inv

def apply_inverse_filter(audio, H_inv):
    X = np.fft.rfft(audio)

    epsilon = 1e-10
    mask = np.abs(X) < epsilon
    X[mask] = epsilon
    H_inv[mask] = epsilon

    Y = X * H_inv
    return np.fft.irfft(Y, n=len(audio))

def high_frequency_recovery(recorded_sweep, ir):
    N = len(recorded_sweep)
    H_inv = design_inverse_filter(ir, N)

    recovered_audio = apply_inverse_filter(recorded_sweep, H_inv)
    recovered_audio = recovered_audio / np.max(np.abs(recovered_audio))
    return recovered_audio

def run_linear_filter_recovery(audio: np.ndarray, data_path: Path, level: str):
    level = level.lower()
    path = data_path / MODEL_NAME / level
    if not path.exists():
        raise ValueError(f"Filter {path} does not exist, you need to compute filters using " "python -m dtu_hsc_solutions.linear_filter.compute_filter <data_path>")
    noise_power = None
    if (noise_power_path := path / "noise_power.npy").exists():
        noise_power = np.load(noise_power_path)
        audio = spectral_subtraction_full_band(audio, noise_power, SAMPLE_RATE)
    ir = np.load(path / "ir.npy")
    audio = high_frequency_recovery(audio, ir)
    if noise_power is not None:
        audio = spectral_subtraction_full_band(audio, noise_power, SAMPLE_RATE)
    audio = audio / np.max(np.abs(audio)) / 2
    return audio
