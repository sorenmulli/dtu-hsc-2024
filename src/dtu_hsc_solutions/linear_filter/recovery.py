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

def fft_dereverberation(reverb_audio, ir, regularization=1e-10):
    """
    Perform dereverberation using FFT-based inverse filtering.

    Parameters:
    reverb_audio: The reverberated audio signal (recorded speech).
    ir: The impulse response of the room (IR).
    regularization: Small value to avoid division by zero in frequency domain.

    Returns:
    dereverb_audio: The dereverberated audio signal.
    """
    reverb_fft = np.fft.fft(reverb_audio, n=len(reverb_audio))
    ir_fft = np.fft.fft(ir, n=len(reverb_audio))
    # Apply inverse filtering in the frequency domain
    dereverb_fft = reverb_fft / (ir_fft + regularization)
    # Perform IFFT to return to time domain
    dereverb_audio = np.fft.ifft(dereverb_fft).real
    # dereverb_audio = apply_time_window(dereverb_audio)
    return dereverb_audio

from ..solution import Solution


class LinearFilter(Solution):

    def __init__(self, data_path: Path, level: str, **kwargs):
        super().__init__(data_path, level)
        self.ir = np.load(self.data_path / MODEL_NAME / self.level / "ir.npy")


    def predict(self, audio: np.ndarray) -> np.ndarray:
        if self.level.startswith("task_1"):
            audio = high_frequency_recovery(audio, self.ir)
        elif self.level.startswith("task_2"):
            audio = fft_dereverberation(audio, self.ir)
        else:
            raise NotImplementedError(f"Filter for level {self.level} not implemented")
        audio = spectral_subtraction_full_band(audio, SAMPLE_RATE)
        return audio / np.max(np.abs(audio)) / 2
