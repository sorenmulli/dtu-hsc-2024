import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import librosa
from scipy import signal
import argparse
import os
import argparse

SAMPLE_RATE = 16_000  # sample rates for audiofiles


def load_audio(audio_path, normalize=True) -> np.ndarray:
    audio, sr = sf.read(audio_path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate of {SAMPLE_RATE}, but got {sr}")
    if normalize:
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    return audio


# Load audio files for this example
def main(recorded_dir, clean_dir, suffix_recorded="_recorded_", suffix_clean="_clean_"):
    """
    Loop through matching audio file pairs in recorded and clean directories based on numeric suffix.

    Parameters:
        recorded_dir (str): Path to folder with distorted audio files.
        clean_dir (str): Path to folder with clean audio files.
        suffix_recorded (str): Middle part of recorded filenames (e.g., '_recorded_').
        suffix_clean (str): Middle part of clean filenames (e.g., '_clean_').

    Returns:
        List of tuples: [(recorded_audio, clean_audio), ...]
    """
    total = 0
    const_break = 0
    for filename in os.listdir(recorded_dir):
        if filename.endswith(".wav") and suffix_recorded in filename:
            # Extract numeric suffix
            parts = filename.split(suffix_recorded)
            if len(parts) == 2:
                base, suffix = parts
                clean_filename = f"{base}{suffix_clean}{suffix}"
                recorded_path = os.path.join(recorded_dir, filename)
                clean_path = os.path.join(clean_dir, clean_filename)

                if os.path.exists(clean_path):
                    recorded_audio = load_audio(recorded_path)
                    clean_audio = load_audio(clean_path)
                else:
                    print(f"Missing clean file for: {filename}")
                record_fft = np.fft.rfft(recorded_audio, n=len(recorded_audio))
                clean_fft = np.fft.rfft(clean_audio, n=len(clean_audio))
                if np.max(abs(clean_fft)) > np.max(abs(record_fft)):
                    const_break += 1
                total += 1

    return const_break, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check if max(fft(clean)) <= max(fft(recorded)) for audio files"
    )
    parser.add_argument(
        "--path_record", type=str, required=True, help="Path to recorded audio files"
    )
    parser.add_argument("--path_clean", type=str, help="Path to clean audio files")
    args = parser.parse_args()
    const_break, total = main(args.path_record, args.path_clean)
    print(
        f"Number of files {total}, Number of max-norm constraint violations: {const_break}"
    )
# C:\Users\chris\OneDrive\Desktop\dtu\hsc24\old_hpc_git\hsc24\dtu-hsc-2024\data\Task_2_Level_2\Recorded
# C:\Users\chris\OneDrive\Desktop\dtu\hsc24\old_hpc_git\hsc24\dtu-hsc-2024\data\Task_2_Level_2\Clean
