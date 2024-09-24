import torch
import torch.nn as nn
import torch.optim as optim
import lightning
from simple_unet.unet_basic import UnetModel
from pytorch_lightning.loggers import WandbLogger
from hsc_dataset import AudioDataset, pad_collate_fn

class UnetLightning(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UnetModel() # Default parameters
        
        self.loss_function = nn.SmoothL1Loss()
        self.loss_total = 0.0


    def forward(self, x):
        return model.forward(x)

    def training_step(self, batch, batch_idx):
        mixture, clean, _ = batch
        
        # TODO: Assume lightning takes care of this
        # mixture = batch.to(self.device)
        # clean = clean.to(self.device)

        # Lightning does this
        # self.optimizer.zero_grad()
        enhanced = self.model(mixture)
        loss = self.loss_function(clean, enhanced)

        # Lightning does this
        # loss.backward()
        # self.optimizer.step()

        self.loss_total += loss.item()

        self.logger.log_metrics({"loss": loss.item()}, step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        # Define the validation step logic here
        pass

    def test_step(self, batch, batch_idx):
        # Define the test step logic here
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return optimizer

    def train_dataloader(self):
        train_data_path = 'src/dtu_hsc_data/data/Task_1_Level_1'
        dataset = AudioDataset(train_data_path, task=1, level=1) 
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=pad_collate_fn)
        return train_loader

    def val_dataloader(self):
        train_data_path = 'src/dtu_hsc_data/data/Task_1_Level_1'
        dataset = AudioDataset(train_data_path, task=1, level=1) 
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=pad_collate_fn)
        return train_loader

    def test_dataloader(self):
        # Define your test data loader here
        pass

if __name__ == "__main__":
    logger = WandbLogger(project='hsc_unet_test')
    # Instantiate the model and trainer, and start training
    model = UnetLightning()
    trainer = lightning.Trainer(logger=logger)
    trainer.fit(model)