from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000


def load_audio(audio_path: Union[str, Path], normalize=True) -> np.ndarray:
    audio, sr = sf.read(audio_path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate of {SAMPLE_RATE}, but got {sr}")
    if normalize:
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    return audio


def save_audio(audio_path: Union[str, Path], audio: np.ndarray):
    sf.write(audio_path, audio, SAMPLE_RATE)
