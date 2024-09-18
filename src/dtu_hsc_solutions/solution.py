from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Solution(ABC):
    def __init__(self, data_path: Path, level: str):
        self.data_path = data_path
        self.level = level.lower()


    @abstractmethod
    def predict(self, audio: np.ndarray) -> np.ndarray:
        # Put your solution here!
        ...
