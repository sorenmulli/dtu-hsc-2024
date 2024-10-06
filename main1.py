"""
 This script is the main submission of the DTU solutions, to run it, you should
 1. Install the package dtu_hsc_solutions from this repository with `pip install -e .` from top-level.
    This installs many dependencies as well.
 2. Have a local directory called `models` containing downloaded model weights. The script assumes
    that this is placed in in the repo root path but you can also customize it with the --models-path argument
 3. Run e.g. `python3 main.py path/to/input/files path/to/output/files T1L3`
 """
import time
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from dtu_hsc_data.audio import SAMPLE_RATE, load_audio, save_audio
from dtu_hsc_solutions.solution import Solution

DCCRNET_TUNED_T2L1 = "dccrnet/load_dccrnet_model_fold_1_model.pth"

def get_solution_configuration(task: str, models_path: Path) -> Solution:
    level_full_name = f"task_{task[1]}_level_{task[3]}"
    if task in {"T1L1", "T1L2", "T1L3"}:
        # Only import when necessary to make work with dependencies easier
        from dtu_hsc_solutions.linear_filter.recovery import LinearFilter
        return LinearFilter(models_path, level_full_name)
    if task in {"T1L4", "T1L5", "T1L6", "T1L7"}:
        from dtu_hsc_solutions.ml_models.voicefixer_model import LinearToVoiceFixerUntuned
        return LinearToVoiceFixerUntuned(models_path, level_full_name)
    if task in {"T2L1",}:
        ## Version 1
        # from dtu_hsc_solutions.ml_models.huggingface_model import DccrNetTuned
        # return DccrNetTuned(models_path, level_full_name, weights_dir=DCCRNET_TUNED_T2L1)
        ## Version 2
        from dtu_hsc_solutions.linear_filter.recovery import SpectralSubtraction
        return SpectralSubtraction(models_path, level_full_name)
    if task in {"T2L2","T2L3"}:
        from dtu_hsc_solutions.linear_filter.recovery import RegLinearFilter
        return RegLinearFilter(models_path, level_full_name)
    if task in {"T3L1","T3L2"}:
        from dtu_hsc_solutions.linear_filter.recovery import CombinedLinearFilter
        return CombinedLinearFilter(models_path, level_full_name)
    raise ValueError(f"Unknown task: {task}")

def main(input_folder: str, output_folder: str, task: str, models_path: str = "models"):
    print(f"Setting up models for task {task} ...")
    solution = get_solution_configuration(task, Path(models_path))

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    audio_files = sorted(list(input_path.glob("*.wav")))
    print(f"Running on {len(audio_files)} audio files in {input_path} ...")

    rtfs = []
    for audio_file in tqdm(audio_files, desc="Running solution"):
        audio = load_audio(audio_file)
        # Timing around prediction
        start_time = time.time()
        output_audio = solution.predict(audio)
        rtfs.append((time.time() - start_time) / len(audio) * SAMPLE_RATE)
        save_audio(output_path / audio_file.name, output_audio)
    print(f"Done with mean Real-Time Factor: {sum(rtfs) / len(rtfs)}.")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Path to the input folder")
    parser.add_argument("output_folder", type=str, help="Path to the output folder")
    parser.add_argument("task", type=str, help="Task to run")
    parser.add_argument("--models-path", type=str, default="models", help="Path to the directory containing model weights")
    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.task, args.models_path)
