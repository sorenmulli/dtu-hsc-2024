from pathlib import Path
from argparse import ArgumentParser
from typing import Callable, Union
from utils import create_signal_path

import numpy as np

import torchaudio
import torch
import os
import time


def sepformer(sig_path: Path):
    from speechbrain.inference.separation import SepformerSeparation as separator
    model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='ml-models/pretrained_models/sepformer-wham16k-enhancement')

    sig, sr = torchaudio.load(sig_path)
    est_sources = model.separate_file(path=sig_path)

    save_path = os.path.split(sig_path)[1].replace(".wav", "_sepForm.wav")
    torchaudio.save(save_path, est_sources[:, :, 0].detach().cpu(), 16000)

def dccrnet(sig_path: Path):
    from asteroid.models import BaseModel
    model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")

    sig, sr = torchaudio.load(sig_path)

    output = torch.squeeze(model(sig))

    save_path = os.path.split(sig_path)[1].replace(".wav", "_dccrnet.wav")
    # Debugging: Print save path and check est_sources dimensions
    print(f"Saving enhanced file to: {save_path}")
    print(f"Estimated sources shape: {output.shape}")

    torchaudio.save(save_path, output.detach().cpu().unsqueeze(0), sr)

def dccrnet_tuned(sig_path: Path):
    from asteroid.models import BaseModel
    model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
    # load the fine-tuned model
    model.load_state_dict(torch.load('ml-models/pretrained_models/dccrnet_model.pth', map_location=torch.device('cpu')))
    #load('ml-models/pretrained_models/dccrnet_model.pth', map_location=torch.device('cpu'))

    sig, sr = torchaudio.load(sig_path)

    output = torch.squeeze(model(sig))

    save_path = os.path.split(sig_path)[1].replace(".wav", "_dccrnet_tuned.wav")
    # Debugging: Print save path and check est_sources dimensions
    print(f"Saving enhanced file to: {save_path}")
    print(f"Estimated sources shape: {output.shape}")

    torchaudio.save(save_path, output.detach().cpu().unsqueeze(0), sr)


KNOWN_SOLUTIONS: dict[str, Callable[[Path], np.ndarray]] = {
    "sepformer": sepformer,
    "dccrnet": dccrnet,
    "dccrnet_tuned": dccrnet_tuned
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", choices=KNOWN_SOLUTIONS.keys())
    parser.add_argument("--task", default="1")
    parser.add_argument("--level", default="1")
    parser.add_argument("--data-path", default="data", help="Directory containing downloaded data from the challenge.")
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()
    sig_path = create_signal_path(args.data_path, args.task, args.level)

    start = time.time()
    KNOWN_SOLUTIONS[args.model.lower()](sig_path)

    end = time.time()
    print(f"Time taken: {end-start} seconds")


