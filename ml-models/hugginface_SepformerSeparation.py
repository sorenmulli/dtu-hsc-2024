from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
import os

cwd = os.getcwd()

#model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir='pretrained_models/sepformer-whamr-enhancement')
model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='ml-models/pretrained_models/sepformer-wham16k-enhancement')

sig_path = 'data/Task_3_Level_2/Task_3_Level_2/Recorded/task_3_level_2_recorded_001.wav'
est_sources = model.separate_file(path=sig_path)

save_path =sig_path.split("/")[-1].replace(".wav", "_sepForm.wav")
torchaudio.save(save_path, est_sources[:, :, 0].detach().cpu(), 8000)
