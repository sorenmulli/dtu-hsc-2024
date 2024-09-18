from pathlib import Path
import numpy as np

from dtu_hsc_data.audio import SAMPLE_RATE
from .utils import MODEL_NAME, spectral_subtraction_full_band

def design_inverse_filter(ir, N, regularization=1e-10):
    H = np.fft.rfft(ir, n=N)
    # H_inv = np.conj(H) / (np.abs(H)**2 + regularization)
    H_inv = 1 / (H + regularization)
    return H_inv


def apply_inverse_filter(audio, H_inv, regularization=1e-10):
    X = np.fft.rfft(audio)
    Y = X * H_inv
    return np.fft.irfft(Y, n=len(audio))

def high_frequency_recovery(recorded_audio, ir):
    N = len(recorded_audio)
    H_inv = design_inverse_filter(ir, N)

    recovered_audio = apply_inverse_filter(recorded_audio, H_inv)
    return recovered_audio

def run_linear_filter_recovery(audio: np.ndarray, data_path: Path, level: str):
    level = level.lower()
    path = data_path / MODEL_NAME / level
    if not path.exists():
        raise ValueError(f"Filter {path} does not exist, you need to compute filters using " "python -m dtu_hsc_solutions.linear_filter.compute_filter <data_path>")
    ir = np.load(path / "ir.npy")
    audio = high_frequency_recovery(audio, ir)
    audio = spectral_subtraction_full_band(audio, SAMPLE_RATE)
    audio = audio / np.max(np.abs(audio)) / 2
    return audio
