from argparse import ArgumentParser
from typing import Callable
from torch.utils.data import DataLoader, Subset
from hsc_dataset import AudioDataset, collate_fn
from utils import load_dccrnet_model, create_data_path
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt



# Define the training loop
def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=10, device="cpu"):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0
        
        for recorded_sig, clean_sig in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
        
        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {epoch_loss / len(train_loader)}")

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for recorded_sig, clean_sig in tqdm(val_loader, desc="Validation"):
                recorded_sig = recorded_sig.to(device)
                clean_sig = clean_sig.to(device)

                predicted_clean = model(recorded_sig)
                loss = loss_fn(predicted_clean, clean_sig)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss / len(val_loader)}")

    return model, train_losses, val_losses

# Function to plot training and validation losses for each fold
def plot_losses_per_fold(all_train_losses, all_val_losses, k_folds):
    plt.figure()

    # Generate a color map to have different colors for each fold
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for fold in range(k_folds):
        color = colors[fold % len(colors)]  # Cycle through colors

        # Plot training loss (solid line)
        plt.plot(all_train_losses[fold], label=f"Fold {fold+1} Train", color=color, linestyle='-')
        
        # Plot validation loss (dashed line)
        plt.plot(all_val_losses[fold], label=f"Fold {fold+1} Val", color=color, linestyle='--')

    plt.title("Training and Validation Losses for Each Fold")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("losses_per_fold.png")  # Save the plot
    plt.show()

# K-Fold Cross-Validation
def cross_validate(model_class, dataset, optimizer_class, loss_fn, k_folds=5, epochs=10, device="cpu"):
    kfold = KFold(n_splits=k_folds, shuffle=True)

    all_train_losses = []
    all_val_losses = []

    fold = 1
    for train_idx, val_idx in kfold.split(dataset):
        print(f"Fold {fold}/{k_folds}")
        
        # Create data loaders for training and validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
        
        # Load model and move to the device
        model = model_class()
        model = model.to(device)
        model.train()

        # Define optimizer
        optimizer = optimizer_class(model.parameters(), lr=1e-4)

        # Train the model on the current fold
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=epochs, device=device)

        # Save the model for this fold
        torch.save(model.state_dict(), f"{model_class.__name__}_fold_{fold}_model.pth")

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        fold += 1

    all_train_losses = np.array(all_train_losses)
    all_val_losses = np.array(all_val_losses)
    
    np.save("all_train_losses.npy", all_train_losses)
    np.save("all_val_losses.npy", all_val_losses)

    plot_losses_per_fold(all_train_losses, all_val_losses, k_folds)
    # save losses


KNOWN_SOLUTIONS: dict[str, Callable[[], torch.nn.Module]] = {
    "dccrnet": load_dccrnet_model,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", choices=KNOWN_SOLUTIONS.keys())
    parser.add_argument("--task", default="1")
    parser.add_argument("--level", default="1")
    parser.add_argument("--data-path", default="data", help="Directory containing downloaded data from the challenge.")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    args = parser.parse_args()
    data_path = create_data_path(args.data_path, args.task, args.level)

    start = time.time()

    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    dataset = AudioDataset(data_path)

    # Define the loss function (SI-SNR)
    loss_fn = ScaleInvariantSignalNoiseRatio().to(device)

    # Cross-validation with K-Folds
    cross_validate(
        model_class=KNOWN_SOLUTIONS[args.model.lower()],
        dataset=dataset,
        optimizer_class=optim.Adam,
        loss_fn=loss_fn,
        k_folds=args.k_folds,
        epochs=args.epochs,
        device=device
    )

    end = time.time()
    print(f"Time taken: {end-start} seconds")
