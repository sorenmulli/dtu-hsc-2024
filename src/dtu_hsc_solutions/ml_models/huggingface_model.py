from pathlib import Path
import numpy as np
import torch
import os

class DccrNet:
    def __init__(self, data_path: Path, level: str):
        from asteroid.models import BaseModel
        self.model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")

    def forward(self, audio: np.ndarray):
        audio = torch.squeeze(self.model(audio))
        return audio.unsqueeze(0)

class DccrNetTuned:
    def __init__(self, data_path: Path, level: str):
        from asteroid.models import BaseModel
        model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
        # load the fine-tuned model
        model_path = os.path.join(data_path, "pretrained_models", "dccrnet_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model = model

    def forward(self, audio: np.ndarray):
        audio = torch.squeeze(self.model(audio))
        return audio.unsqueeze(0)
