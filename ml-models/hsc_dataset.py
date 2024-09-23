import os
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import torch

class AudioDataset(Dataset):
    def __init__(self, data_dir:Path):
        # Define the paths for clean and recorded subfolders
        self.clean_dir = Path(data_dir) / "Clean"
        self.recorded_dir = Path(data_dir) / "Recorded"

        # Get a list of all clean audio files
        self.clean_files = sorted(list(self.clean_dir.glob("*.wav")))
        self.recorded_files = sorted(list(self.recorded_dir.glob("*.wav")))

    def __len__(self):
        # Number of samples in the dataset
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Get the clean and recorded file paths
        clean_path = self.clean_files[idx]
        recorded_path = self.recorded_files[idx]

        # Load both the clean and recorded signals
        clean_sig, sr_clean = torchaudio.load(clean_path)
        recorded_sig, sr_recorded = torchaudio.load(recorded_path)

        # Ensure both signals have the same sampling rate (optional: resample if needed)
        assert sr_clean == sr_recorded, "Sampling rates do not match!"

        # Return the recorded signal (input) and the clean signal (target)
        return recorded_sig, clean_sig

def collate_fn(batch):
    recorded_signals = [item[0] for item in batch]
    clean_signals = [item[1] for item in batch]

    # Back padding for recorded signals (default behavior)
    recorded_signals_padded = pad_sequence(recorded_signals, batch_first=True, padding_value=0.0)
    
    # Get the maximum length of the clean signals
    max_len = max([x.size(0) for x in clean_signals])
    
    # Front padding for clean signals
    def pad_front(signal, max_len):
        padding_size = max_len - signal.size(0)
        return torch.cat([torch.zeros(padding_size), signal])

    clean_signals_padded = torch.stack([pad_front(signal, max_len) for signal in clean_signals])

    return recorded_signals_padded, clean_signals_padded