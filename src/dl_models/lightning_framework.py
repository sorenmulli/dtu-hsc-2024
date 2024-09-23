import torch
import torch.nn as nn
import lightning as L
from torchmetrics.functional import accuracy
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
import torchaudio
import os
import argparse
from hsc_given_code.evaluate import evaluate

parser = argparse.ArgumentParser()
args = parser.parse_args()

class lightningFramework(torch.nn.Module):
    def __init__(self, in_model, save_location) -> None:
        """Reuse from above to debug Lightningcli

        Args:
            in_features: Size of input layer very important I swear
            out_features: Should not be called features I know, number of classes in dataset 
            lr: Learning rate of adam optimizer used
        
        Returns:
            torch.tensor
        """

        
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=64*6*6, out_features=600),
            nn.Dropout1d(0.25),
            nn.Linear(in_features=600, out_features=120),
            nn.Linear(in_features=120, out_features=out_features)
        )
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input.unsqueeze(1))


def create_model(input_size, output_size):
    return FashionMnistModel(in_features=input_size, out_features=output_size)

class lightningFramework(L.LightningModule):
    def __init__(self, models: list, save_location=None, model_name='lightning_model_',  **kwargs):
        super(lightningFramework, self).__init__()

        self.models = models

        # Ensure all models implement a forward function
        for model in models:
            forward = getattr(model, "forward", None)
            assert callable(forward)
                
        # UNDERSCORE AFTER NAME PLEASE
        self.model_name = model_name

        # Do not change this!!
        self.test_dataloader_batch_size = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SHOULD NOT BE CHANGED"""
        for model in self.models:
            x = model.forward(x)
        return x
        
    def training_step(self, batch, batch_idx, eval_on_deepspeech=False):
        """CAN BE CHANGED DEPENDING ON MODEL"""
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y.to(torch.long))
        self.log("train_loss", loss)

        # if eval_on_deepspeech:
        #     # Cast torch tensor to numpy int 16 array
        #     # Use DeepSpeech.stt(array)
            # mean_cer = self.eval_on_deepspeech()
            # self.log("mean_cer", mean_cer)
        #     return loss

    def eval_on_deepspeech(self, test_data, wav_save_path='filtered', csv_path=None, verbose=False):
        """
        Evaluates given test data on deep speech model according to HSC documentation

        Args:
            test_data (torch.tensor): test data that will then be filtered by pipeline
            wav_save_path (str, optional): Location to save resulting .wav files, must be speicified to evalutae on deepspeeech
            csv_path (str, optional): Location to save .csv file from HSC evaluate script, if None will save same place as wav_save_path.
        """
        
        if csv_path is None:
            csv_path = wav_save_path

        all_file_names = []

        for batch, labels, file_names in test_data:
            filtered = self(batch)

            for file_name in file_names:
                all_file_names.append(file_name)

            save_path = [os.path.join(wav_save_path , self.model_name) + i for i in file_names]

            for filtered_audio, path in zip(save_path, filtered):
                # TODO: CHECK DIMENSIONS ON FILTERED_AUDIO
                torchaudio.save(path, filtered_audio[:, :, 0].detach().cpu(), 16000)
                

        args.model_path = "HSC/deepspeech-0.9.3-models.pbmm" 
        args.scorer_path = "HSC/deepspeech-0.9.3-models.scorer" 
        args.verbose = verbose
        # TODO: ENSURE WAV_SAVE_PATH ALSO INCLUDES PATH ALL THE WAY FROM ROOT!
            # Not necessary - Will just save to root directory, is fine!
        args.audio_dir = wav_save_path


        for unique_combination in unique_combinations:
                
            # TODO: Ensure original texts are all loaded one at a time or all at once, or change evaluate
            args.text_file = test_data.dataset.original_texts

            mean_CER = evaluate(args)

    def configure_optimizers(self):
        # TODO IMPLEMENT SOME OTHER STUFF HERE
# TODO: FUCK IT, JUST DO IT FOR ONE FUCKING TASK AND LEVEL AT A FUCKING TIME!!!!
    
import re

# Example list of files
file_list = [
    "task_2_level_2_recorded_001.wav",
    "task_1_level_1_recorded_001.wav",
    "task_3_level_3_recorded_001.wav",
    "task_2_level_1_recorded_001.wav",
    "task_1_level_2_recorded_001.wav",
    "task_3_level_1_recorded_001.wav"
]

# Regex pattern to extract task and level numbers
pattern = re.compile(r"task_(\d+)_level_(\d+)_")

# Function to sort files
def sort_files(files):
    return sorted(files, key=lambda f: tuple(map(int, pattern.search(f).groups())))

# Function to get unique task_level combinations
def get_unique_combinations(files):
    unique_combinations = set()
    for file in files:
        match = pattern.search(file)
        if match:
            task_level = f"task_{match.group(1)}_level_{match.group(2)}"
            unique_combinations.add(task_level)
    return sorted(unique_combinations, key=lambda x: tuple(map(int, re.findall(r'\d+', x))))

# Sorting the file list
sorted_files = sort_files(file_list)
print("Sorted Files:")
for file in sorted_files:
    print(file)

# Getting unique task_level combinations
unique_combinations = get_unique_combinations(file_list)
print("\nUnique Task_Level Combinations:")
for combination in unique_combinations:
    print(combination)














    # def setup_deepspeech_eval(self):
    #     # TODO: SET ARGS PATH HERE!!!
    #     self.DeepSpeech_model = Model(args.model_path)
    #     self.DeepSpeech_model.enableExternalScorer(args.scorer_path) 
    #     self.transformation = jiwer.Compose(
    #     [
    #         jiwer.ToLowerCase(),
    #         normalize_us_spelling,
    #         jiwer.ExpandCommonEnglishContractions(),
    #         replace_dashes,
    #         replace_z,
    #         jiwer.RemoveMultipleSpaces(),
    #         jiwer.RemovePunctuation(),
    #         jiwer.RemoveWhiteSpace(replace_by_space=True),
    #         jiwer.Strip(),

    #     ]
    # )

    # def evaluate(self, batch, stage=None):
    #     x, y = batch
    #     logits = self(x)
    #     loss = self.criterion(logits, y.to(torch.long))
    #     preds = torch.argmax(logits, dim=1)
    #     acc = accuracy(preds, y, task="multiclass", num_classes=10)

    #     if stage:
    #         self.log(f"{stage}_loss", loss, prog_bar=True)
    #         self.log(f"{stage}_acc", acc, prog_bar=True)

    # def eval_on_deepspeech(processed_results):
    #     """

    #     Args:
    #         processed_results (list (str)): list of file paths leading to already processed .wav files
    #     """

    #     # Directory and file processing
    #     audio_files = [f for f in os.listdir(args.audio_dir) if f.endswith(".wav")]

    #     # List to store the result
    #     full_result = []

    #     with open(args.text_file, "r") as file:
    #         for audio_file in sorted(audio_files):
    #             full_path = os.path.join(args.audio_dir, audio_file)
    #             if args.verbose > 0:
    #                 print(f"Processing and transcribing {audio_file}...")

    #             transcribed_text = process_and_transcribe(model, full_path)

    #             line = file.readline()
    #             parts = line.strip().split("\t")
    #             while parts[0] != audio_file:
    #                 line = file.readline()
    #                 parts = line.strip().split("\t")

    #             if parts[0] == audio_file:
    #                 original_text = parts[1]
    #                 if args.verbose > 0:
    #                     print(f"Transcription: {transcribed_text}")
    #                     print(f"True: {original_text}")
    #                 metrics = calculate_metrics(original_text, transcribed_text, transformation)
    #                 result = {
    #                     "Filename": audio_file,
    #                     "Original Text": original_text,
    #                     "Transcribed Text": transcribed_text,
    #                 }
    #                 if metrics:
    #                     result.update(metrics)
    #                     if args.verbose > 0:
    #                         for metric, value in metrics.items():
    #                             print(f"{metric}: {value:.2f}")

    #                 full_result.append(result)

    #     # Save results to CSV
    #     df = pd.DataFrame(full_result)
    #     if args.output_csv:
    #         df.to_csv(args.output_csv, index=False)
    #         print(f"Results saved to {args.output_csv}")
    #     else:
    #         print("No output CSV file specified; results will not be saved.")

    #     print(f"Mean CER: {df['CER'].mean()}")
    #     print("CER Quantiles:")
    #     print(df["CER"].quantile([0.25, 0.5, 0.75]))
    #     return df["CER"].mean()



    # def validation_step(self, batch, batch_idx):
    #     self.evaluate(batch, "val")

    # def test_step(self, batch, batch_idx):
    #     self.evaluate(batch, "test") 

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(
    #         self.model.parameters(),
    #         lr=self.lr,
    #     )
    #     return {"optimizer": optimizer}

    # def train_dataloader(self):
    #     train_data = torch.load('data/processed/train.pt')
    #     return DataLoader(train_data, batch_size=16)

    # def test_dataloader(self):
    #     test_data =  torch.load('data/processed/test.pt')
    #     return DataLoader(test_data, batch_size=16)

    # def predict_dataloader(self):
    #     predict_data =  torch.load('data/processed/test.pt')
    #     return DataLoader(predict_data, batch_size=16)
