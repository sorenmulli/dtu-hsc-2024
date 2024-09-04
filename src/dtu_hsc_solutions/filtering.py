from pathlib import Path
from scipy.signal import wiener
import numpy as np


def run_wiener(audio: np.ndarray, data_path: Path, level: str) -> np.ndarray:
    """
    https://en.wikipedia.org/wiki/Wiener_filter
    """
    return wiener(audio)
