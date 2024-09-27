import torch
import os
from pathlib import Path


def create_data_path(data_path: Path, task: int, level: int, ir: bool = False):
    # Return the path to the data directory for the given task and level
    if ir:
        return os.path.join(data_path, "linear-filter", f"task_{task}_level_{level}")
    return os.path.join(data_path,f"Task_{task}_Level_{level}")


def create_signal_path(data_path: Path, task: int, level: int):
    # Return the path to the recorded signal file
    return os.path.join(create_data_path(data_path, task, level),"Recorded", f"task_{task}_level_{level}_recorded_001.wav")


def load_dccrnet_model():
    from asteroid.models import BaseModel
    model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
    return model
