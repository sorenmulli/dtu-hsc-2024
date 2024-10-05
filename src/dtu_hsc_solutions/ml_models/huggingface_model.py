from pathlib import Path
import numpy as np
import torch
from ..solution import Solution
from dtu_hsc_solutions.linear_filter.recovery import LinearFilter

class DccrNet(Solution):
    def __init__(self, data_path: Path, level: str, **kwargs):
        from asteroid.models import BaseModel
        self.model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
        self.model.eval()

    def predict(self, audio: np.ndarray):
        with torch.no_grad():
            torch_audio = torch.from_numpy(audio).float()
            audio = torch.squeeze(self.model(torch_audio))
            return audio.numpy()

class DccrNetTuned(Solution):

    def __init__(
        self,
        data_path: Path,
        level: str,
        weights_dir: str="pretrained_models/load_dccrnet_model_10epochs_fold_3_model.pth",
        **kwargs,
    ):
        from asteroid.models import BaseModel
        model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
        # load the fine-tuned model
        model_path = str(data_path / Path(weights_dir))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        self.model = model
        self.model.eval()

    def predict(self, audio: np.ndarray):
        with torch.no_grad():
            torch_audio = torch.from_numpy(audio).float()
            audio = torch.squeeze(self.model(torch_audio))
            return audio.numpy()

class LinearToDccrNetTuned(Solution):

    def __init__(
        self,
        data_path: Path,
        level: str,
        weights_dir: str="pretrained_models/load_dccrnet_model_10epochs_fold_3_model.pth",
        **kwargs,
    ):
        from asteroid.models import BaseModel
        model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
        # load the fine-tuned model
        model_path = str(data_path / Path(weights_dir))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        self.model = model
        self.model.eval()
        self.linear_filter = LinearFilter(data_path, level)


    def predict(self, audio: np.ndarray):
        linear_filtered_audio = self.linear_filter.predict(audio)
        with torch.no_grad():
            torch_audio = torch.from_numpy(linear_filtered_audio).float()
            audio = torch.squeeze(self.model(torch_audio))
            return audio.numpy()
        
class LinearToDccrUntuned(Solution):
    def __init__(self, data_path: Path, level: str, **kwargs):
        super().__init__(data_path, level)
        self.linear_filter = LinearFilter(data_path, level)
        self.Dccrnet = DccrNet(data_path, level)


    def predict(self, audio: np.ndarray) -> np.ndarray:
        linear_filtered_audio = self.linear_filter.predict(audio)
        final_audio = self.Dccrnet.predict(linear_filtered_audio)
        return final_audio