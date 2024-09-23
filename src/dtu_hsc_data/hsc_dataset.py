import os
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import torch
import glob
import re
import librosa
import numpy as np

def sort_by_last_three_digits(files):
    # Define a sorting key function
    def extract_digits(filename):
        # Use regex to find the last three digits before .wav
        match = re.search(r'(\d{3})\.wav$', filename)
        if match:
            return int(match.group(1))  # Return the digits as an integer for sorting
        else:
            return float('inf')  # Return infinity if no match is found, so they go to the end
    
    # Sort the list using the custom key function
    return sorted(files, key=extract_digits)

def load_and_process_data(tasks, levels, get_every=5, min_freq=0, max_freq=1000):
    """_summary_

    Args:
        data_path (_type_): Path to .wav files of data to get
        get_every (int): The granularity of frequencies to get, higher values will get lower number of frequencies 
        min_max_freq (list): Frequencies above or below these values will be cut off

    Returns:
        np.ndarray: Fourier transform of data all collected in one
    """

    data_processed = []

    data_root = 'dtu_hsc_data/data'

    for task in tasks:
        for level in levels:
            data_path = os.path.join(data_root, f'Task_{task}_Level_{level}')

            for tp_dat in ['Clean', 'Recorded']:
                glob_path = os.path.join(data_path, '*.wav')
                files_in_dir = glob.glob(glob_path)
                # Sort by last three digits before file extension, so we are sure that the correct clean and recordeds match up with one another
                files_in_dir = sort_by_last_three_digits(files_in_dir)

                for file in files_in_dir:
                    audio, sr = librosa.load(file, sr=16000, mono=True)

                    # Remembr, start:stop:step  
                    torch.tensor(audio)
            
    return np.array(data_processed) 

class AudioDataset(Dataset):
    def __init__(self, data_dir:Path, tasks: list, levels: list, dataset_name='placeholder', return_recorded_path=True):
        # Ensure tasks and levels are strings to placate the great glob monstere
        tasks, levels = list(map(str, tasks)), list(map(str, levels))

        self.clean_glob = f'src/dtu_hsc_data/data/Task_[{",".join(tasks)}]_Level_[{",".join(levels)}]/Clean/*.wav'
        self.recorded_glob = f'src/dtu_hsc_data/data/Task_[{",".join(tasks)}]_Level_[{",".join(levels)}]/Recorded/*.wav'

        self.clean_files = sorted(list(glob.glob(self.clean_glob)))
        self.recorded_files = sorted(list(glob.glob(self.recorded_glob)))

        self.orignal_texts_glob = f'src/dtu_hsc_data/data/Task_[{",".join(tasks)}]_Level_[{",".join(levels)}]/Task_[{",".join(tasks)}]_Level_[{",".join(levels)}]_text_samples.txt'

        self.original_texts = sorted(list(glob.glob(self.orignal_texts_glob)))

        self.dataset_name = f"torchDataset_tasks_{tasks}_levels_{levels}_" + dataset_name + '.pt'
        
        self.return_recorded_path = return_recorded_path

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        recorded_path = self.recorded_files[idx]

        clean_sig, sr_clean = torchaudio.load(clean_path)
        recorded_sig, sr_recorded = torchaudio.load(recorded_path)

        # Ensure both signals have the same sampling rate (optional: resample if needed)
        assert sr_clean == sr_recorded, "Sampling rates do not match!"

        # Return the recorded signal (input) and the clean signal (target)
        if self.return_recorded_path:
            recorded_sig, clean_sig, recorded_path
        
        return recorded_sig, clean_sig

    def save(self, save_location):
        torch.save(self, os.path.join(save_location, self.dataset_name))


# Custom collate_fn to pad sequences
def collate_fn(batch):
    # Extract recorded and clean signals from the batch
    recorded_sigs, clean_sigs = zip(*batch)
    
    # Find the max length of the signals in the batch
    max_len = max(sig.size(1) for sig in recorded_sigs)
    
    # Pad recorded and clean signals to the max length
    padded_recorded = [torch.nn.functional.pad(sig, (0, max_len - sig.size(1))) for sig in recorded_sigs]
    padded_clean = [torch.nn.functional.pad(sig, (0, max_len - sig.size(1))) for sig in clean_sigs]
    
    # Stack the padded signals into tensors
    padded_recorded = torch.stack(padded_recorded)
    padded_clean = torch.stack(padded_clean)

    return padded_recorded, padded_clean


# dat_set = AudioDataset('dtu_hsc_data_data', tasks=[1,2,3], levels=[1,2,3])
