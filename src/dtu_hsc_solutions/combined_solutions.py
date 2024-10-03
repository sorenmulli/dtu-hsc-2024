from dtu_hsc_solutions.ml_models.huggingface_model import DccrNet
from dtu_hsc_solutions.linear_filter.recovery import LinearFilter
from dtu_hsc_solutions.ml_models.voicefixer_model import VoiceFixerUntuned
from dtu_hsc_solutions.solution import Solution
from pathlib import Path
import numpy as np

class LinearToDccrUntuned(Solution):
    def __init__(self, data_path: Path, level: str, **kwargs):
        super().__init__(data_path, level)
        self.linear_filter = LinearFilter(data_path, level)
        self.Dccrnet = DccrNet(data_path, level)


    def predict(self, audio: np.ndarray) -> np.ndarray:
        linear_filtered_audio = self.linear_filter.predict(audio)
        final_audio = self.Dccrnet.predict(linear_filtered_audio)
        return final_audio

class LinearToVoiceFixerUntuned(Solution):
    def __init__(self, data_path: Path, level: str, **kwargs):
        super().__init__(data_path, level)
        self.linear_filter = LinearFilter(data_path, level)
        self.voicefixer = VoiceFixerUntuned(data_path, level)


    def predict(self, audio: np.ndarray) -> np.ndarray:
        linear_filtered_audio = self.linear_filter.predict(audio)
        final_audio = self.voicefixer.predict(linear_filtered_audio)
        return final_audio
