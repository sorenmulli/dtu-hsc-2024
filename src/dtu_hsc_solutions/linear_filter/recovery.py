from pathlib import Path
import numpy as np
import cvxpy as cp
import warnings

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

def reg_high_frequency_recovery(recorded_audio, ir):
    warnings.filterwarnings("ignore") # sometimes cvxpy throws a warning
    H = np.fft.rfft(ir, n=len(recorded_audio))
    X = np.fft.rfft(recorded_audio)
    Y = cp.Variable(len(X),complex=True)
    obj = cp.Minimize(cp.norm(cp.multiply(H,Y)-X,2)**2)#+1e-2*cp.norm(Y,2)**2)
    constraint = [cp.max(cp.abs(Y))<=np.max(abs(X))]
    prob = cp.Problem(obj,constraint)
    prob.solve(solver=cp.CLARABEL,ignore_dpp = True,max_iter=20)
    if (prob.status == "optimal") or (prob.status == "user_limit"):
        Y.value[2500:len(Y.value)] = 10*Y.value[2500:len(Y.value)]/np.max(Y.value[2500:len(Y.value)])
        return np.fft.irfft(Y.value).real
    else:
        return np.fft.irfft(X / (H+1e-10),n=len(recorded_audio))


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

def reg_fft_dereverberation(reverb_audio, ir): #lam=1e-10):
    """
    Perform dereverberation using FFT-based inverse filtering with prior based regularization
    using the convex solver package cvxpy

    Parameters:
    reverb_audio: The reverberated audio signal (recorded speech).
    ir: The impulse response of the room (IR).
    regularization: Small value to avoid division by zero in frequency domain.

    Returns:
    dereverb_audio: The dereverberated audio signal.
    """

    warnings.filterwarnings("ignore") # sometimes cvxpy throws a warning
    reverb_fft = np.fft.rfft(reverb_audio, n=len(reverb_audio))
    ir_fft = np.fft.rfft(ir, n=len(reverb_audio))
    Y = cp.Variable(len(reverb_fft),complex=True)
    obj = cp.Minimize(cp.norm(cp.multiply(ir_fft+1e-10,Y)-reverb_fft,2)**2) #+lam*cp.norm(Y,1))
    constraint = [cp.max(cp.abs(Y))<=np.max(abs(reverb_fft))]
    prob = cp.Problem(obj,constraint)
    prob.solve(solver=cp.CLARABEL,ignore_dpp = True,max_iter=10)
    if (prob.status == "optimal") or (prob.status == "user_limit"):
        return np.fft.irfft(Y.value).real
    else:
        return np.fft.irfft(reverb_fft / (ir_fft + 1e-10)).real

def cutoff_fft_dereverberation(reverb_audio, ir, lam=1e-10,max_comp=800):
    max_comp = np.max(np.fft.rfft(reverb_audio))
    out = np.clip(out, None, max_comp)
    return out

from ..solution import Solution

class SpectralSubtraction(Solution):

    def predict(self, audio: np.ndarray) -> np.ndarray:
        audio = spectral_subtraction_full_band(audio, SAMPLE_RATE)
        return audio / np.max(np.abs(audio)) / 2


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

class RegLinearFilter(Solution):
        def __init__(self, data_path: Path, level: str, **kwargs):
            super().__init__(data_path, level)
            self.ir = np.load(self.data_path / MODEL_NAME / self.level / "ir.npy")

        def predict(self, audio: np.ndarray) -> np.ndarray:
            if self.level.startswith("task_2"):
                audio = spectral_subtraction_full_band(audio, SAMPLE_RATE)
                audio = reg_fft_dereverberation(audio, self.ir)
            elif self.level.startswith("task_1"):
                audio = reg_high_frequency_recovery(audio, self.ir)
            else:
                raise NotImplementedError(f"RegLinearFilter for level {self.level} not implemented")
            audio = spectral_subtraction_full_band(audio, SAMPLE_RATE)
            return audio / np.max(np.abs(audio)) / 2

class SpectralSubtraction(Solution):
        def __init__(self, data_path: Path, level: str, **kwargs):
            super().__init__(data_path, level)

        def predict(self, audio: np.ndarray) -> np.ndarray:
            audio = spectral_subtraction_full_band(audio, SAMPLE_RATE)
            return audio / np.max(np.abs(audio)) / 2

class CombinedLinearFilter(Solution):
        def __init__(self, data_path: Path, level: str, **kwargs):
            super().__init__(data_path, level)
            self.ir1 = np.load(self.data_path / MODEL_NAME / self.level / "ir_task1.npy")
            self.ir2 = np.load(self.data_path / MODEL_NAME / self.level / "ir_task2.npy")

        def predict(self, audio: np.ndarray) -> np.ndarray:
            if self.level.startswith("task_3"):
                audio = high_frequency_recovery(audio, self.ir1)
                audio = spectral_subtraction_full_band(audio, SAMPLE_RATE)
                audio = reg_fft_dereverberation(audio, self.ir2)
                audio = spectral_subtraction_full_band(audio, SAMPLE_RATE)
                return audio / np.max(np.abs(audio)) / 2
            else:
                raise NotImplementedError(f"CombinedLinearFilter for level {self.level} not implemented")
