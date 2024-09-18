from argparse import ArgumentParser
from typing import Callable
from torch.utils.data import DataLoader
from hsc_dataset import AudioDataset, collate_fn
from utils import load_dccrnet_model, create_data_path
import numpy as np
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import torch
import torch.optim as optim
import time


# Define the training loop
def train_model(model, dataloader, optimizer, loss_fn, epochs=10, device="cpu"):
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for recorded_sig, clean_sig in tqdm(dataloader):
            # Move data to the appropriate device if using a GPU
            recorded_sig = recorded_sig.to(device)
            clean_sig = clean_sig.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: predict clean signal from recorded signal
            predicted_clean = model(recorded_sig)

            # Compute loss
            loss = loss_fn(predicted_clean, clean_sig)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader)}")
    return model

KNOWN_SOLUTIONS: dict[str, Callable[[], np.ndarray]] = {
    "dccrnet": load_dccrnet_model,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", choices=KNOWN_SOLUTIONS.keys())
    parser.add_argument("--task", default="1")
    parser.add_argument("--level", default="1")
    parser.add_argument("--data-path", default="data", help="Directory containing downloaded data from the challenge.")
    args = parser.parse_args()
    data_path = create_data_path(args.data_path, args.task, args.level)

    start = time.time()

    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your dataset
    dataset = AudioDataset(data_path)

    # Create a DataLoader to iterate over the dataset
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)

    model = KNOWN_SOLUTIONS[args.model.lower()]()
    model = model.to(device)

    # Set the model to training mode
    model.train()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Define the loss function scale invariant signal-to-noise ratio
    loss_fn = ScaleInvariantSignalNoiseRatio().to(device)

    # Train the model
    model = train_model(model, dataloader, optimizer, loss_fn, epochs=10, device=device)

    # Save the trained model
    torch.save(model.state_dict(), f"{args.model}_model.pth")

    end = time.time()
    print(f"Time taken: {end-start} seconds")