import os
import torchaudio
from torch.utils.data import Dataset, default_collate
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import torch
import torch.nn.functional as F


class AudioDataset(Dataset):
    def __init__(self, data_dir:Path, task, level):
        """
        Args:
            data_dir (Path): Direct directory to a given task-level
        """

        self.clean_dir = Path(data_dir) / "Clean"
        self.recorded_dir = Path(data_dir) / "Recorded"

        self.clean_files = sorted(list(self.clean_dir.glob("*.wav")))
        self.recorded_files = sorted(list(self.recorded_dir.glob("*.wav")))
        
        self.original_texts = Path(data_dir) / f'Task_{task}_Level_{level}_text_samples.txt'

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        recorded_path = self.recorded_files[idx]

        # Load both the clean and recorded signals
        clean_sig, sr_clean = torchaudio.load(clean_path)
        recorded_sig, sr_recorded = torchaudio.load(recorded_path)

        # Ensure both signals have the same sampling rate (optional: resample if needed)
        assert sr_clean == sr_recorded, "Sampling rates do not match!"

        # Return the recorded signal (input) and the clean signal (target), as well as path to recorded signal for evaluation
        return recorded_sig, clean_sig, recorded_path

def pad_collate_fn(batch):
    # recorded_signals, clean_signals, file_names = default_collate(batch)
    recorded_signals = [item[0] for item in batch]
    clean_signals = [item[1] for item in batch]
    file_names = [item[2] for item in batch]

    pad_len = find_nearest_valid_pad(max([x.size(1) for x in recorded_signals + clean_signals]))
    
    recorded_signals = dynamic_pad(recorded_signals, 0.0, pad_len, front=False)
    clean_signals = dynamic_pad(clean_signals, 0.0, pad_len, front=True)

    return recorded_signals, clean_signals, file_names


def find_nearest_valid_pad(n, z=12):
    """
    Since the model uses 12 layers of downsampling, which cuts input size in half,
    the input signal length must be divisible by 2^12=4096.

    Args:
        n (_type_): _description_
        z (int, optional): _description_. Defaults to 12.
    """

    # Calculate the multiple of 2^z
    multiple = 2 ** z

    # Round n down and up to the nearest multiple of 2^z
    # lower_p = (n // multiple) * multiple
    upper_p = ((n + multiple - 1) // multiple) * multiple

    # Return the closest one
    # if abs(n - lower_p) <= abs(n - upper_p):
    #     return lower_p
    
    return upper_p


def dynamic_pad(sequences, pad_value, max_len, front=True):
    """
    Pad a batch of 1D sequences based on the maximum sequence length in the batch.
    
    Args:
        sequences (list of torch.Tensor)
        pad_value_func (function): Function that determines the pad value for each sequence.
        max_len (int): The length to pad each sequence to.
        front: (bool): Whether to pad in the front, if false, pads to back
        
    Returns:
        torch.Tensor: Padded batch of sequences as a tensor.
    """
    # Find the maximum sequence length
    # max_len = max([seq.size(1) for seq in sequences])
    
    # Pad each sequence dynamically based on its length and the max length
    padded_sequences = []
    
    for seq in sequences:
        # Calculate how much padding is needed for each sequence
        pad_len = max_len - seq.size(1)
                
        # Apply padding (only to the back in this case, modify if front padding is needed)
        padded_seq = F.pad(seq, (pad_len * front, pad_len * (1-front)), value=pad_value)
        padded_sequences.append(padded_seq)
    
    # Stack all padded sequences into a single tensor
    return torch.stack(padded_sequences)