import os
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import torch

class AudioDataset(Dataset):
    def __init__(self, data_dir:Path):
        self.data_dir = data_dir

        # Define the paths for clean and recorded subfolders
        self.clean_dir = Path(data_dir) / "Clean"
        self.recorded_dir = Path(data_dir) / "Recorded"

        # Get a list of all clean audio files
        self.clean_files = sorted(list(self.clean_dir.glob("*.wav")))
        self.recorded_files = sorted(list(self.recorded_dir.glob("*.wav")))

        # Get name of audio files
        sample_file = list(Path(data_dir).glob("*.txt"))[0]
        with open(sample_file,"r") as file:
            self.names = sorted([name.strip().split("\t")[0] for name in file])

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
        return recorded_sig, clean_sig, self.names[idx]

# Custom collate_fn to pad sequences
def collate_fn(batch):
    # Extract recorded and clean signals from the batch
    recorded_sigs, clean_sigs, names = zip(*batch)
    
    # Find the max length of the signals in the batch
    max_len = max(sig.size(1) for sig in recorded_sigs)
    
    # Back pad recorded signals (pad zeros at the end)
    padded_recorded = [torch.nn.functional.pad(sig, (0, max_len - sig.size(1))) for sig in recorded_sigs]
    
    # Front pad clean signals (pad zeros at the beginning)
    padded_clean = [torch.nn.functional.pad(sig, (max_len - sig.size(1), 0)) for sig in clean_sigs]
    
    # Stack the padded signals into tensors
    padded_recorded = torch.stack(padded_recorded)
    padded_clean = torch.stack(padded_clean)

    return padded_recorded, padded_clean, names