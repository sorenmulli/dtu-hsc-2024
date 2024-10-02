import os
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np
from scipy import signal
from ..linear_filter.utils import calculate_signal_delay
import matplotlib.pyplot as plt
from .utils import load_dccrnet_model, create_data_path


class AudioDataset(Dataset):
    def __init__(self, data_dir:Path, aligned:bool=False, ir:bool=False):
        # Define the paths for clean and recorded subfolders
        self.clean_dir = Path(data_dir) / "Clean"
        if ir:
            print("Using IR data")
            self.recorded_dir = Path(data_dir) / "IR"
        elif aligned:
            print("Using aligned data")
            self.recorded_dir = Path(data_dir) / "Aligned"
        else:
            print("Using raw recorded data")
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
        clean_sig, sr_clean = torchaudio.load(str(clean_path))
        recorded_sig, sr_recorded = torchaudio.load(str(recorded_path))

        # Ensure both signals have the same sampling rate (optional: resample if needed)
        assert sr_clean == sr_recorded, "Sampling rates do not match!"

        # Return the recorded signal (input) and the clean signal (target)
        return recorded_sig, clean_sig, recorded_path, clean_path

# Custom collate_fn to pad sequences
def collate_fn_naive(batch):
    # Extract recorded and clean signals from the batch
    recorded_sigs, clean_sigs,_,_ = zip(*batch)
    
    # Find the max length of the signals in the batch
    max_len = max(sig.size(1) for sig in recorded_sigs)
    
    # Back pad recorded signals (pad zeros at the end)
    padded_recorded = [torch.nn.functional.pad(sig, (0, max_len - sig.size(1))) for sig in recorded_sigs]
    
    # Front pad clean signals (pad zeros at the beginning)
    padded_clean = [torch.nn.functional.pad(sig, (max_len - sig.size(1), 0)) for sig in clean_sigs]
    
    # Stack the padded signals into tensors
    padded_recorded = torch.stack(padded_recorded)
    padded_clean = torch.stack(padded_clean)

    return padded_recorded, padded_clean, None, None



import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def save_spectrogram(signal, fs, save_path='spectrogram.png'):
    f, t, Sxx = signal.spectrogram(signal, fs=fs)
    
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    
    # Save as an image
    plt.savefig(save_path)
    plt.close()

def calculate_signal_delay_torch(clean_signal, recorded_signal, fs, max_shift=16000):
    # Convert numpy arrays to torch tensors
    clean_signal_torch = torch.from_numpy(clean_signal).float()
    recorded_signal_torch = torch.from_numpy(recorded_signal).float()
    
    # Ensure input tensors are 3D: [batch_size, in_channels, signal_length]
    clean_signal_torch = clean_signal_torch.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, signal_length]
    recorded_signal_torch = recorded_signal_torch.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, signal_length]
    
    # Ensure the input tensors are 3D, if they still have an extra dimension, use .squeeze() to remove it
    if clean_signal_torch.dim() == 4:
        clean_signal_torch = clean_signal_torch.squeeze(1)  # Remove any extra channel dimension
    if recorded_signal_torch.dim() == 4:
        recorded_signal_torch = recorded_signal_torch.squeeze(1)

    # Ensure clean_signal is not longer than recorded_signal
    if clean_signal_torch.shape[-1] > recorded_signal_torch.shape[-1]:
        clean_signal_torch = clean_signal_torch[..., :recorded_signal_torch.shape[-1]]

    # Perform cross-correlation using conv1d
    correlation = torch.nn.functional.conv1d(recorded_signal_torch, clean_signal_torch.flip(2), padding=len(clean_signal) - 1)
    
    delay_samples = correlation.argmax() - len(clean_signal) + 1
    
    if abs(delay_samples) > max_shift:
        delay_samples = max_shift if delay_samples > 0 else -max_shift
    
    if delay_samples > 0:
        processed_speech_aligned = recorded_signal_torch[delay_samples:]
        processed_speech_aligned = torch.nn.functional.pad(processed_speech_aligned, (0, delay_samples))
    elif delay_samples < 0:
        processed_speech_aligned = torch.nn.functional.pad(recorded_signal_torch, (abs(delay_samples), 0))
        processed_speech_aligned = processed_speech_aligned[:len(recorded_signal)]
    else:
        processed_speech_aligned = recorded_signal_torch
    
    return processed_speech_aligned

def collate_fn(batch, fs=16000, spectrogram_save_path='spectrogram.png'):
    clean_signals, recorded_signals = zip(*batch)
    
    # Convert to numpy arrays
    clean_sigs_np = [sig.numpy() for sig in clean_signals]
    recorded_sigs_np = [sig.numpy() for sig in recorded_signals]

    # Align each recorded signal with the corresponding clean signal
    aligned_signals = []
    for i, (clean_sig, recorded_sig) in enumerate(zip(clean_sigs_np, recorded_sigs_np)):
        aligned_sig = calculate_signal_delay_torch(clean_sig, recorded_sig, fs)
        aligned_signals.append(aligned_sig)
        
        # Calculate and save spectrogram for the first signal in the batch
        #if i == 0:
        #    save_spectrogram(aligned_sig.numpy(), fs, save_path=spectrogram_save_path)

    # Stack aligned signals into a tensor
    aligned_signals_tensor = torch.stack(aligned_signals)
    clean_signals_tensor = torch.stack([torch.from_numpy(sig) for sig in clean_sigs_np])
    print(aligned_signals_tensor.shape)
    print(clean_signals_tensor.shape)

    return aligned_signals_tensor, clean_signals_tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchaudio.transforms as T
    import torchaudio
    import torch
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--task", default="1")
    parser.add_argument("--level", default="1")
    parser.add_argument("--data-path", default="data", help="Directory containing downloaded data from the challenge.")
    parser.add_argument("--plot", default="False", help="If true plots are produced else the data is aligned and saved.")

    args = parser.parse_args()
    data_path = create_data_path(args.data_path, args.task, args.level)

    # Load the dataset
    dataset = AudioDataset(data_path,aligned=True, ir=False)

    # Define the loss function (SI-SNR)
    #loss_fn = ScaleInvariantSignalNoiseRatio().to(device)
    if not args.plot == "True":
        for i in range(len(dataset)):
            recorded, clean, recorded_path, clean_path = dataset[i]
            clean = clean.squeeze(0)
            recorded = recorded.squeeze(0)
            recorded = torch.tensor(calculate_signal_delay(clean, recorded, fs=16000))

            clean = clean.unsqueeze(0)
            recorded = recorded.unsqueeze(0)

            # Save the aligned recorded signal where "Recorded" is replaced with "Aligned" in the folder structure
            aligned_path = recorded_path.parent.parent / "Aligned" / recorded_path.name
            os.makedirs(aligned_path.parent, exist_ok=True)
            torchaudio.save(str(aligned_path), recorded, 16000)

    else:
        i = 0
        recorded, clean,_,_ = dataset[i]

        print(clean.shape)
        print(recorded.shape)

        # Compute the spectrogram of the clean and recorded signals
        spec_clean = T.Spectrogram()(clean)
        spec_recorded = T.Spectrogram()(recorded)

        # Convert the spectrograms to decibels
        spec_clean_db = T.AmplitudeToDB()(spec_clean)
        spec_recorded_db = T.AmplitudeToDB()(spec_recorded)

        # Plot the spectrograms
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"Clean Signal, len {clean.shape[1]}")
        plt.imshow(spec_clean_db[0].numpy(), aspect="auto", origin="lower")
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.title(f"Recorded Signal, len {recorded.shape[1]}")
        plt.imshow(spec_recorded_db[0].numpy(), aspect="auto", origin="lower")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'spectrogram{i}_raw.png')

        clean, recorded = collate_fn_naive([(clean, recorded)])
        print("After collate fn")
        print(clean.shape)
        print(recorded.shape)

        clean = clean.squeeze(0)
        recorded = recorded.squeeze(0)

        # Compute the spectrogram of the clean and recorded signals
        spec_clean = T.Spectrogram()(clean)
        spec_recorded = T.Spectrogram()(recorded)

        # Convert the spectrograms to decibels
        spec_clean_db = T.AmplitudeToDB()(spec_clean)
        spec_recorded_db = T.AmplitudeToDB()(spec_recorded)

        # Plot the spectrograms
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"Clean Signal, len {clean.shape[1]}")
        plt.imshow(spec_clean_db[0].numpy(), aspect="auto", origin="lower")
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.title(f"Recorded Signal, len {recorded.shape[1]}")
        plt.imshow(spec_recorded_db[0].numpy(), aspect="auto", origin="lower")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'spectrogram{i}_naive.png')

        clean, recorded,_,_ = dataset[i]
        clean = clean.squeeze(0)
        recorded = recorded.squeeze(0)
        recorded = torch.tensor(calculate_signal_delay(clean, recorded, fs=16000))

        clean = clean.unsqueeze(0)
        recorded = recorded.unsqueeze(0)

        print(clean.shape)
        print(recorded.shape)

        # Compute the spectrogram of the clean and recorded signals
        spec_clean = T.Spectrogram()(clean)
        spec_recorded = T.Spectrogram()(recorded)

        # Convert the spectrograms to decibels
        spec_clean_db = T.AmplitudeToDB()(spec_clean)
        spec_recorded_db = T.AmplitudeToDB()(spec_recorded)

        # Plot the spectrograms
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"Clean Signal, len {clean.shape[1]}")
        plt.imshow(spec_clean_db[0].numpy(), aspect="auto", origin="lower")
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.title(f"Recorded Signal, len {recorded.shape[1]}")
        plt.imshow(spec_recorded_db[0].numpy(), aspect="auto", origin="lower")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'spectrogram{i}_crosscor.png')





    
