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
    parser.add_argument("--vocoder", default="False", help="If true the vocoder is tested.")

    args = parser.parse_args()
    data_path = create_data_path(args.data_path, args.task, args.level)

    # Load the dataset
    dataset = AudioDataset(data_path,aligned=False, ir=False)

    if args.vocoder == "True":
        i = 0
        recorded, clean,_,_ = dataset[i]
        test_vocoder(recorded)

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





    
