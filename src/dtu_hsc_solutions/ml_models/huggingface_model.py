from pathlib import Path
import numpy as np
import torch
import os
from ..solution import Solution

class DccrNet(Solution):
    def __init__(self, data_path: Path, level: str):
        from asteroid.models import BaseModel
        self.model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
        self.model.eval()

    def predict(self, audio: np.ndarray):
        with torch.no_grad():
            torch_audio = torch.from_numpy(audio).float()
            audio = torch.squeeze(self.model(torch_audio))
            return audio.numpy()

class DccrNetTuned(Solution):
    def __init__(self, data_path: Path, level: str):
        from asteroid.models import BaseModel
        model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
        # load the fine-tuned model
        model_path = os.path.join(data_path, "pretrained_models", "load_dccrnet_model_10epochs_fold_3_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        self.model = model
        self.model.eval()

    def predict(self, audio: np.ndarray):
        with torch.no_grad():
            torch_audio = torch.from_numpy(audio).float()
            audio = torch.squeeze(self.model(torch_audio))
            return audio.numpy()
