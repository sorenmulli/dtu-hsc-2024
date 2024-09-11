# Combination of Yue's get_noise_power.py and get_ir.py
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from dtu_hsc_data import get_task_data

from .ir_processing import process_ir
from .utils import MODEL_NAME, estimate_noise_from_vad, load_audio

IMPULSE_RESPONSE_NAME = "Impulse_Responses"
TASK_1_IMPULSE_RESPONSE = Path("white_noise_short.wav")
TASK_2_IMPULSE_RESPONSE = Path("swept_sine_wave.wav")

def main(data_path: Path):
    model_path = data_path / MODEL_NAME
    model_path.mkdir(exist_ok=True)
    ir_path = data_path / IMPULSE_RESPONSE_NAME
    print(f"Computing filters for all tasks and saving to model path {model_path}")
    task_data = get_task_data(data_path)

    task_1_clean_signal, _ = load_audio(ir_path / "Clean" / TASK_1_IMPULSE_RESPONSE)
    task_2_clean_signal, _ = load_audio(ir_path / "Clean" / TASK_2_IMPULSE_RESPONSE)

    for level in task_data.keys():
        level = level.lower()
        print(f"Processing level {level}")
        if level.startswith("task_1"):
            clean_signal = task_1_clean_signal
            ir_file = TASK_1_IMPULSE_RESPONSE
        elif level.startswith("task_2"):
            clean_signal = task_2_clean_signal
            ir_file = TASK_2_IMPULSE_RESPONSE
        else:
            print("Skipping Task 3 for now")
            continue

        recorded_ir, fs = load_audio(ir_path / "Recorded" / f"{ir_file.stem}_{level}.wav")
        out_path = model_path / level
        out_path.mkdir(exist_ok=True)

        if level.startswith("task_1"):
            # ir_file = ir_path / TASK_1_IMPULSE_RESPONSE
            print("\t Performing noise power estimation")
            noise_power = estimate_noise_from_vad(recorded_ir, fs)
            np.save(noise_out := out_path / "noise_power.npy", noise_power)
            print("\t\tSaved to", noise_out)
        print("\t Performing IR processing")
        ir = process_ir(clean_signal, recorded_ir, fs, level.startswith("task_2"), plot=False)
        print(f"\t\tIR length: {len(ir)} samples ({len(ir)/fs:.3f} seconds)")
        np.save(ir_out := out_path / "ir.npy", ir)
        print("\t\tSaved to", ir_out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "data-path",
        help="Directory containing downloaded data from the challenge."
        " The output will also be placed here in a sub-folder",
    )
    args = parser.parse_args()
    main(Path(vars(args)["data-path"]))
