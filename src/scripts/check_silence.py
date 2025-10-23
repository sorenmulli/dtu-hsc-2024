import os
import torch
import argparse
import glob
import torch
from dtu_hsc_data.audio import load_audio
from silero_vad import get_speech_timestamps, load_silero_vad

SAMPLE_RATE = 16000  # Sample rate for the audio files

def is_silent(audio_path, model, start_ms, end_ms):
    """
    Check if the audio file has silence in the specified range using a simple magnitude comparison.

    Args:
        audio_path (str): Path to the audio file.
        model: Silero VAD model (not used in this check).
        start_ms (int): Start time in milliseconds.
        end_ms (int): End time in milliseconds.

    Returns:
        bool: True if the specified range is silent, False otherwise.
    """
    try:
        audio = load_audio(audio_path, normalize=False)
    except Exception as e:
        raise ValueError(f"Error loading audio file {audio_path}: {e}")

    # Convert start and end times to sample indices
    start_sample = int(start_ms * SAMPLE_RATE / 1000)
    end_sample = int(end_ms * SAMPLE_RATE / 1000)

    # Ensure the range is within the audio length
    if end_sample > len(audio):
        raise ValueError(f"Specified range {start_ms}-{end_ms} ms exceeds audio length.")

    # Extract the relevant segment
    if audio.ndim > 1:
        audio = audio[:, 0]  # take first channel
    segment = audio[start_sample:end_sample]
    segment = torch.from_numpy(segment).float()

    # Calculate average magnitude of the entire audio and the segment
    avg_magnitude = abs(audio).mean()
    segment_magnitude = abs(segment).mean()

    # Use Silero VAD to detect speech timestamps
    timestamps = get_speech_timestamps(segment, model, sampling_rate=SAMPLE_RATE)
    return len(timestamps) == 0  # True if no speech detected

def main(audio_dir, start_ms, end_ms):
    """
    Main function to check silence in audio clips using Silero VAD.

    Args:
        audio_dir (str): Directory containing audio files.
        start_ms (int): Start time in milliseconds.
        end_ms (int): End time in milliseconds.
    """
    model = load_silero_vad()

    # Recursively find all .wav files in Task_1_*/Aligned and Task_2_*/Aligned
    search_pattern = os.path.join(audio_dir, "Task_[12]_*/Aligned/*.wav")
    audio_files = glob.glob(search_pattern, recursive=True)
    total_files = len(audio_files)
    non_silent_count = 0

    for audio_path in audio_files:
        try:
            if not is_silent(audio_path, model, start_ms, end_ms):
                non_silent_count += 1
        except ValueError as e:
            print(f"Skipping {audio_path}: {e}")
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    print(f"Total audio files: {total_files}")
    print(f"Audio files without silence in the first {start_ms}-{end_ms} ms: {non_silent_count}")
    print(f"Audio files with silence in the first {start_ms}-{end_ms} ms: {total_files - non_silent_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check silence in audio clips.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Base directory containing Task_1 and Task_2 folders.")
    parser.add_argument("--start_ms", type=int, default=50, help="Start time in milliseconds.")
    parser.add_argument("--end_ms", type=int, default=800, help="End time in milliseconds.")

    args = parser.parse_args()
    main(args.audio_dir, args.start_ms, args.end_ms)
