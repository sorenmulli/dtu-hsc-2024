from pathlib import Path
import numpy as np
import librosa
import torch
from dtu_hsc_data.audio import SAMPLE_RATE
from ..solution import Solution
#from voicefixer import VoiceFixer
from dtu_hsc_solutions.linear_filter.recovery import LinearFilter


class VoiceFixerUntuned(Solution):

    def __init__(self, data_path: Path, level: str, **kwargs):
        super().__init__(data_path, level)
        self.model = VoiceFixer() # Tuned for a sample rate of 44100

    def predict(self, audio: np.ndarray) -> np.ndarray:
        # Will use CUDA is available
        cuda_available = torch.cuda.is_available()
        upsampled_audio = librosa.resample(audio,SAMPLE_RATE,44100)
        filtered_audio = self.model.restore_inmem(upsampled_audio,cuda=cuda_available,mode=0,your_vocoder_func=None)
        downsampled_filtered_audio = librosa.resample(filtered_audio,44100,SAMPLE_RATE)
        return downsampled_filtered_audio

class LinearToVoiceFixerUntuned(Solution):
    def __init__(self, data_path: Path, level: str, **kwargs):
        super().__init__(data_path, level)
        self.linear_filter = LinearFilter(data_path, level)
        self.voicefixer = VoiceFixerUntuned(data_path, level)


    def predict(self, audio: np.ndarray) -> np.ndarray:
        linear_filtered_audio = self.linear_filter.predict(audio)
        final_audio = self.voicefixer.predict(linear_filtered_audio)
        return final_audio
