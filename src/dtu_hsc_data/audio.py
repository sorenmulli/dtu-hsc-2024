from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000


def load_audio(audio_path: Union[str, Path]) -> np.ndarray:
    audio, sr = sf.read(audio_path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate of {SAMPLE_RATE}, but got {sr}")
    return audio


def save_audio(audio_path: Union[str, Path], audio: np.ndarray):
    sf.write(audio_path, audio, SAMPLE_RATE)
