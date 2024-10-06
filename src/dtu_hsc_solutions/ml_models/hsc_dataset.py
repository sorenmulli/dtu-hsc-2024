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
    def __init__(self, data_dir:Path, aligned:bool=False, ir:bool=False, synth:bool=False):
        # Define the paths for clean and recorded subfolders
        self.clean_dir = Path(data_dir) / "Clean"
        if synth:
            print("Using synthetic data")
            self.clean_dir = Path(data_dir) / "SynthClean"
            self.recorded_dir = Path(data_dir) / "SynthAligned"
        elif ir:
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

        # Normalize both signals by dividing by their absolute maximum
        clean_sig = clean_sig / torch.max(torch.abs(clean_sig))
        recorded_sig = recorded_sig / torch.max(torch.abs(recorded_sig))

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
    padded_clean = [torch.nn.functional.pad(sig, (0, max_len - sig.size(1))) for sig in clean_sigs]

    # Stack the padded signals into tensors
    padded_recorded = torch.stack(padded_recorded)
    padded_clean = torch.stack(padded_clean)

    return padded_recorded, padded_clean, None, None


def save_spectrogram(signal, fs, save_path='spectrogram.png'):
    f, t, Sxx = signal.spectrogram(signal, fs=fs)
    
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    
    # Save as an image
    plt.savefig(save_path)
    plt.close()


def create_aligned_data(dataset):
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

def test_vocoder(audio, sample_rate:int=16000):
    import torch
    import torchaudio
    import matplotlib.pyplot as plt
    from vocos import Vocos

    # Define the transformation
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=80)

    # Apply the transformation
    mel_spectrogram = transform(audio)

    # Convert to log scale (dB)
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

    # Display the mel spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(log_mel_spectrogram[0].numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig("mel_spectrogram.png")

    vocos = Vocos.from_pretrained("BSC-LT/wavenext-mel")

    y = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=22050)
    y_hat = vocos(y)
    torchaudio.save(str("vocoder_audio_hat.wav"), y_hat, 22050)

    audio_out = vocos.decode(mel_spectrogram)
    torchaudio.save(str("vocoder_audio.wav"), audio_out, 16000)

    # is there a difference between the two audio files?
    print(np.mean(audio_out-audio))

# Function to calculate the frequency distribution (density) of clean signals in the dataset
def plot_frequency_density(dataset, n_fft=512, hop_length=128, win_length=512, sample_rate=16000):
    """
    Plots the density of frequency content across all clean signals in the dataset using STFT.
    
    Parameters:
    - dataset: A dataset containing clean and recorded audio signals
    - n_fft: Number of FFT points
    - hop_length: Number of audio samples between adjacent STFT columns
    - win_length: Window size for each FFT
    - sample_rate: Sample rate of the audio files
    
    Returns:
    - None: This function directly plots the frequency density.
    """
    # Initialize an array to accumulate the frequency content (only positive frequencies)
    total_frequencies = np.zeros(n_fft // 2)
    
    for idx in range(len(dataset)):
        # Load clean signal (ignoring the recorded signal)
        clean_signal, _, _, _ = dataset[idx]
        
        # Check if the clean signal is mono or multi-channel (use only first channel)
        if clean_signal.shape[0] > 1:
            clean_signal = clean_signal[0, :]  # Take the first channel
        
        # Compute STFT to get frequency content
        stft = torch.stft(clean_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
        
        # Convert STFT to magnitude (absolute value)
        magnitude = torch.abs(stft)
        
        # Sum magnitude across time (sum along the columns to get total energy for each frequency bin)
        frequency_sum = torch.sum(magnitude, dim=1).numpy()

        # Squeeze the frequency sum to remove any extra dimensions and ensure it's a 1D array
        frequency_sum = np.squeeze(frequency_sum)

        # Ensure the frequency_sum has the expected length of n_fft // 2
        if frequency_sum.shape[0] != n_fft // 2:
            frequency_sum = frequency_sum[:n_fft // 2]  # Adjust shape if needed

        # Only keep the positive frequencies (first half)
        total_frequencies += frequency_sum

    # Normalize to get a density-like distribution
    density = total_frequencies / np.sum(total_frequencies)

    # Generate the frequency axis (convert bins to Hz)
    freqs = np.fft.fftfreq(n_fft, 1.0 / sample_rate)
    
    # Only plot the positive frequencies (first half of the spectrum)
    positive_freqs = freqs[:n_fft // 2]

    # Plot the frequency density
    plt.figure(figsize=(10, 5))
    plt.plot(positive_freqs, density)
    plt.title("Frequency Density Across Clean Signals")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig("freq.png")
    plt.show()

def plot_frequency_density_clean_vs_recorded(dataset, n_fft=512, hop_length=128, win_length=512, sample_rate=16000):
    """
    Plots the density of frequency content across all clean and recorded signals in the dataset using STFT.
    
    Parameters:
    - dataset: A dataset containing clean and recorded audio signals
    - n_fft: Number of FFT points
    - hop_length: Number of audio samples between adjacent STFT columns
    - win_length: Window size for each FFT
    - sample_rate: Sample rate of the audio files
    
    Returns:
    - None: This function directly plots the frequency density.
    """
    # Initialize arrays to accumulate the frequency content for clean and recorded signals
    total_frequencies_clean = np.zeros(n_fft // 2)
    total_frequencies_recorded = np.zeros(n_fft // 2)
    
    for idx in range(len(dataset)):
        # Load clean and recorded signals
        recorded_signal, clean_signal, _, _ = dataset[idx]
        
        # Ensure both clean and recorded signals are mono (use only first channel)
        if clean_signal.shape[0] > 1:
            clean_signal = clean_signal[0, :]
        if recorded_signal.shape[0] > 1:
            recorded_signal = recorded_signal[0, :]
        
        # Compute STFT for both clean and recorded signals
        stft_clean = torch.stft(clean_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
        stft_recorded = torch.stft(recorded_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
        
        # Convert STFT to magnitude (absolute value)
        magnitude_clean = torch.abs(stft_clean)
        magnitude_recorded = torch.abs(stft_recorded)
        
        # Sum magnitude across time (sum along the columns to get total energy for each frequency bin)
        frequency_sum_clean = torch.sum(magnitude_clean, dim=1).numpy()
        frequency_sum_recorded = torch.sum(magnitude_recorded, dim=1).numpy()

        # Squeeze to ensure they're 1D arrays
        frequency_sum_clean = np.squeeze(frequency_sum_clean)
        frequency_sum_recorded = np.squeeze(frequency_sum_recorded)

        # Ensure the sums have the expected length of n_fft // 2
        if frequency_sum_clean.shape[0] != n_fft // 2:
            frequency_sum_clean = frequency_sum_clean[:n_fft // 2]
        if frequency_sum_recorded.shape[0] != n_fft // 2:
            frequency_sum_recorded = frequency_sum_recorded[:n_fft // 2]
        
        # Accumulate the frequency content
        total_frequencies_clean += frequency_sum_clean
        total_frequencies_recorded += frequency_sum_recorded

    # Normalize to get a density-like distribution
    density_clean = total_frequencies_clean / np.sum(total_frequencies_clean)
    density_recorded = total_frequencies_recorded / np.sum(total_frequencies_recorded)

    # Generate the frequency axis (convert bins to Hz)
    freqs = np.fft.fftfreq(n_fft, 1.0 / sample_rate)
    
    # Only plot the positive frequencies (first half of the spectrum)
    positive_freqs = freqs[:n_fft // 2]

    # Plot the frequency density for clean and recorded signals
    plt.figure(figsize=(10, 5))
    plt.plot(positive_freqs, density_clean, label="Clean Signal", color="blue")
    plt.plot(positive_freqs, density_recorded, label="Recorded Signal", color="orange")
    plt.title("Frequency Density: Clean vs. Recorded Signals")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.savefig("freq_clean_vs_recorded.png")
    plt.show()

def plot_frequency_density_recorded_only(dataset, n_fft=512, hop_length=128, win_length=512, sample_rate=16000):
    """
    Plots the density of frequency content across all recorded signals in the dataset using STFT.
    
    Parameters:
    - dataset: A dataset containing recorded audio signals
    - n_fft: Number of FFT points
    - hop_length: Number of audio samples between adjacent STFT columns
    - win_length: Window size for each FFT
    - sample_rate: Sample rate of the audio files
    
    Returns:
    - None: This function directly plots the frequency density for recorded signals.
    """
    # Initialize an array to accumulate the frequency content for recorded signals
    total_frequencies_recorded = np.zeros(n_fft // 2)
    
    for idx in range(len(dataset)):
        # Load only the recorded signal (ignore the clean signal)
        recorded_signal, _, _, _ = dataset[idx]
        print(f"Recorded signal {idx} min: {recorded_signal.min()}, max: {recorded_signal.max()}")

        
        # Ensure the recorded signal is mono (use only first channel)
        if recorded_signal.shape[0] > 1:
            recorded_signal = recorded_signal[0, :]
        
        # Compute STFT to get frequency content
        stft_recorded = torch.stft(recorded_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
        
        # Convert STFT to magnitude (absolute value)
        magnitude_recorded = torch.abs(stft_recorded)
        
        # Sum magnitude across time (sum along the columns to get total energy for each frequency bin)
        frequency_sum_recorded = torch.sum(magnitude_recorded, dim=1).numpy()

        # Squeeze to ensure it's a 1D array
        frequency_sum_recorded = np.squeeze(frequency_sum_recorded)

        # Ensure the sums have the expected length of n_fft // 2
        if frequency_sum_recorded.shape[0] != n_fft // 2:
            frequency_sum_recorded = frequency_sum_recorded[:n_fft // 2]
        
        # Accumulate the frequency content
        total_frequencies_recorded += frequency_sum_recorded

    # Normalize to get a density-like distribution
    density_recorded = total_frequencies_recorded / np.sum(total_frequencies_recorded)

    # Generate the frequency axis (convert bins to Hz)
    freqs = np.fft.fftfreq(n_fft, 1.0 / sample_rate)
    
    # Only plot the positive frequencies (first half of the spectrum)
    positive_freqs = freqs[:n_fft // 2]

    # Plot the frequency density for recorded signals
    plt.figure(figsize=(10, 5))
    plt.plot(positive_freqs, density_recorded, label="Recorded Signal", color="orange")
    plt.title("Frequency Density of Recorded Signals")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.savefig("freq_recorded.png")
    plt.show()

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
    parser.add_argument("--freq", default="False", help="If true plot frequencies.")

    args = parser.parse_args()
    data_path = create_data_path(args.data_path, args.task, args.level)

    # Load the dataset
    dataset = AudioDataset(data_path,aligned=False, ir=False)

    if args.freq == "True":
        #plot_frequency_density(dataset)
        #plot_frequency_density_clean_vs_recorded(dataset)
        plot_frequency_density_recorded_only(dataset)

    if not args.plot == "True":
        create_aligned_data(dataset)
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

        recorded, clean,_,_ = dataset[i]
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
        plt.title(f"Cross-correlated Aligned Signal, len {recorded.shape[1]}")
        plt.imshow(spec_recorded_db[0].numpy(), aspect="auto", origin="lower")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'spectrogram{i}_crosscor.png')





    
