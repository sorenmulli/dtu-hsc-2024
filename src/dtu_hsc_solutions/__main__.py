import time
from pathlib import Path
from argparse import ArgumentParser
from typing import Union

import numpy as np
from tqdm import tqdm

from dtu_hsc_data import get_task_data, SAMPLE_RATE, save_audio
from .solution import Solution
from .linear_filter.recovery import LinearFilter
from .ml_models.huggingface_model import DccrNet, DccrNetTuned
from .combined_solutions import LinearToDccrUntuned, LinearToVoiceFixerUntuned
from .ml_models.voicefixer_model import VoiceFixerUntuned

OUTPUT_DIR = "output"

KNOWN_SOLUTIONS: dict[str, type[Solution]] = {
    "linear-filter": LinearFilter,
    "dccrnet": DccrNet,
    "dccrnet-tuned": DccrNetTuned,
    "voicefixer": VoiceFixerUntuned,
    "linear-to-dccrnet": LinearToDccrUntuned,
    "linear-to-voicefixer": LinearToVoiceFixerUntuned
}

def run_solution(
    data_path: Union[str, Path],
    solution: str,
    level: str,
    overwrite_output: bool,
    dir_postfix: str,
):
    solution_class = KNOWN_SOLUTIONS.get(solution.lower())

    if solution_class is None:
        raise ValueError(f"Unknown solution: {solution}")

    solution_object = solution_class(Path(data_path), level)

    data_examples = get_task_data(data_path)[level]
    level_path = level + dir_postfix
    output_path = Path(data_path) / OUTPUT_DIR / solution / level_path
    try:
        output_path.mkdir(parents=True, exist_ok=overwrite_output)
    except FileExistsError:
        raise ValueError(
            f"Output path already exists: {output_path}. Use --overwrite-output to overwrite"
        )

    print(
        f"Running solution {solution} on {len(data_examples)} examples from {level}, "
        f"writing output to:\n{output_path.resolve()}"
    )
    rtfs: list[float] = []
    for example in tqdm(data_examples, desc="Running solution"):
        audio = example.get_recorded()
        # Have timing around to compute real-time factor
        start_time = time.time()
        output_audio = solution_object.predict(audio)
        end_time = time.time()
        rtfs.append((end_time - start_time) / len(audio) * SAMPLE_RATE)

        save_audio(output_path / example.recorded_audio_path.name, output_audio)
    print(f"Mean Real-Time Factor: {np.mean(rtfs)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "data-path",
        help="Directory containing downloaded data from the challenge."
        " The output will also be placed here in a sub-folder",
    )
    parser.add_argument("solution", choices=KNOWN_SOLUTIONS.keys())
    parser.add_argument("--level", default="Task_1_Level_1")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--dir-postfix", default="")
    args = parser.parse_args()
    run_solution(**{name.replace("-", "_"): val for name, val in vars(args).items()})
