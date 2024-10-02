from pathlib import Path
import numpy as np

from dtu_hsc_data.audio import SAMPLE_RATE

from ..solution import Solution

from voicefixer import VoiceFixer
import librosa
import torch

class VoiceFixerUntuned(Solution):

    def __init__(self, data_path: Path, level: str):
        super().__init__(data_path, level)
        self.model = VoiceFixer() # Tuned for a sample rate of 44100

    def predict(self, audio: np.ndarray) -> np.ndarray:
        # Will use CUDA is available
        cuda_available = torch.cuda.is_available()
        upsampled_audio = librosa.resample(audio,SAMPLE_RATE,44100)
        filtered_audio = self.model.restore_inmem(upsampled_audio,cuda=cuda_available,mode=0,your_vocoder_func=None)
        downsampled_filtered_audio = librosa.resample(filtered_audio,44100,SAMPLE_RATE)
        return downsampled_filtered_audio
