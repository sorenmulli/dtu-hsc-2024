from pathlib import Path
from argparse import ArgumentParser
from typing import Callable, Union

import numpy as np

import torchaudio
import os
import time
import inspect


def create_path(data_path: Path, task: int, level: int):
    return os.path.join(data_path,f"Task_{task}_Level_{level}",f"Task_{task}_Level_{level}","Recorded", f"task_{task}_level_{level}_recorded_001.wav")


def sepformer(sig_path: Path):
    from speechbrain.inference.separation import SepformerSeparation as separator
    model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='ml-models/pretrained_models/sepformer-wham16k-enhancement')

    sig_path = 'data/Task_2_Level_1/Task_2_Level_1/Recorded/task_2_level_1_recorded_001.wav'
    sig, sr = torchaudio.load(sig_path)
    est_sources = model.separate_file(path=sig_path)

    save_path =sig_path.split("/")[-1].replace(".wav", "_sepForm.wav")
    torchaudio.save(save_path, est_sources[:, :, 0].detach().cpu(), 16000)

def dccrnet(sig_path: Path):
    from asteroid.models import BaseModel
    model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")

    inspect.getmembers(model, predicate=inspect.ismethod)
    sig, sr = torchaudio.load(sig_path)

    est_sources = model.separate_file(path=sig_path)

    save_path =sig_path.split("/")[-1].replace(".wav", "_.wav")
    torchaudio.save(save_path, est_sources[:, :, 0].detach().cpu(), sr)


KNOWN_SOLUTIONS: dict[str, Callable[[Path], np.ndarray]] = {
    "sepformer": sepformer,
    "dccrnet": dccrnet,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", choices=KNOWN_SOLUTIONS.keys())
    parser.add_argument("--task", default="1")
    parser.add_argument("--level", default="1")
    parser.add_argument("--data-path", default="data", help="Directory containing downloaded data from the challenge.")
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()
    sig_path = create_path(args.data_path, args.task, args.level)

    start = time.time()
    KNOWN_SOLUTIONS[args.model.lower()](sig_path)

    end = time.time()
    print(f"Time taken: {end-start} seconds")


