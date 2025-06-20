# Combination of Yue's get_noise_power.py and get_ir.py
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from dtu_hsc_data import get_task_data

from .ir_processing import process_ir
from .utils import MODEL_NAME, load_audio

IMPULSE_RESPONSE_NAME = "Impulse_Responses"
TASK_1_IMPULSE_RESPONSE = Path("white_noise_short.wav")
TASK_2_IMPULSE_RESPONSE = Path("swept_sine_wave.wav")

TASK_3_SPECIALS = {
    "task_1_level_2": "task_3_level_1/ir_task1.npy",
    "task_2_level_2": "task_3_level_1/ir_task2.npy",
    "task_1_level_4": "task_3_level_2/ir_task1.npy",
    "task_2_level_3": "task_3_level_2/ir_task2.npy",
}

CUT_POINTS = {
    "task_1_level_1": 740,
    "task_1_level_2": 736,
    "task_1_level_3": 825,
    "task_1_level_4": 633,
    "task_1_level_5": 530,
    "task_1_level_6": 593,
    "task_1_level_7": 662,
    "task_2_level_1": 52404,
    "task_2_level_2": 58388,
    "task_2_level_3": 54672,
}

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
            continue

        cut_point = CUT_POINTS[level]
        recorded_ir, fs = load_audio(ir_path / "Recorded" / f"{ir_file.stem}_{level}.wav")
        out_path = model_path / level
        out_path.mkdir(exist_ok=True)

        print("\t Performing IR processing")
        ir = process_ir(clean_signal, recorded_ir, fs, cut_point, plot=False)
        print(f"\t\tIR length: {len(ir)} samples ({len(ir)/fs:.3f} seconds)")
        np.save(ir_out := out_path / "ir.npy", ir)
        print("\t\tSaved to", ir_out)

        if level in TASK_3_SPECIALS:
            task3_ir_out = model_path / TASK_3_SPECIALS[level]
            task3_ir_out.parent.mkdir(exist_ok=True)
            np.save(task3_ir_out, ir)
            print("\t\tAlso saved to", ir_out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "data-path",
        help="Directory containing downloaded data from the challenge."
        " The output will also be placed here in a sub-folder",
    )
    args = parser.parse_args()
    main(Path(vars(args)["data-path"]))
