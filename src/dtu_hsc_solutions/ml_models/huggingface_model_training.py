from argparse import ArgumentParser
from typing import Callable
from torch.utils.data import DataLoader, Subset
from .hsc_dataset import AudioDataset, collate_fn_naive
from .utils import load_dccrnet_model, create_data_path, si_sdr, spectral_convergence_loss, combined_loss
#from ...hsc_given_code.evaluate import evaluate
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
#from asteroid.losses import SingleSrcPITLossWrapper, pairwise_neg_sisdr
import torch
import torch.optim as optim
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os
import torchaudio
from pathlib import Path
from ...hsc_given_code.evaluate import evaluate
from dtu_hsc_data.audio import SAMPLE_RATE
import shutil

# Define the training loop
def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=10, device="cpu"):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0
        
        for recorded_sig, clean_sig, _,_ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
            for recorded_sig, clean_sig,_,_ in tqdm(val_loader, desc="Validation"):
                recorded_sig = recorded_sig.to(device)
                clean_sig = clean_sig.to(device)

                predicted_clean = model(recorded_sig)
                loss = loss_fn(predicted_clean, clean_sig)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss / len(val_loader)}")

    return model, train_losses, val_losses

class ArgObject(object):
    pass

def evaluate_model(model, val_loader, device="cpu"):
    audio_dir = Path("data/output/hugginface_model_mini_evaluation/")
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Creates output path
        if audio_dir.exists() and audio_dir.is_dir():
            shutil.rmtree(audio_dir)
        Path(audio_dir).mkdir(parents=True, exist_ok=True)

        # Evaluates each audio in the validation set
        for recorded_sig, _, names, _ in tqdm(val_loader, desc="Evaluation"):
            recorded_sig = recorded_sig.to(device)

            predicted_clean = model(recorded_sig)
            for j in range(len(names)):
                name = names[j]
                torchaudio.save(audio_dir / name, predicted_clean[j].detach().cpu(), SAMPLE_RATE)
    
    parts = name.strip().split("_")
    task_name = parts[0].capitalize() + "_" + parts[1] + "_" + parts[2].capitalize() + "_" + parts[3]
    args = ArgObject()
    args.model_path = "data/deepspeech-0.9.3-models.pbmm"
    args.scorer_path = "data/deepspeech-0.9.3-models.scorer"
    args.verbose = 1
    args.audio_dir = audio_dir
    args.text_file = "data/" + task_name + "/" + task_name + "_text_samples.txt"
    args.output_csv = None
    return evaluate(args)

# Function to plot training and validation losses for each fold
def plot_losses_per_fold(all_train_losses, all_val_losses, k_folds, output_path="."):
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
    plt.savefig(os.path.join(output_path,"losses_per_fold.png"))  # Save the plot
    plt.show()

# K-Fold Cross-Validation
def cross_validate(model_class, dataset, optimizer_class, loss_fn, k_folds=5, epochs=10, device="cpu", output_path=".", freeze_encoder=False):
    if k_folds < 2:
        train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.1, shuffle=True)
        kfold_splits = [(train_idx, val_idx)]  # Wrapping in a list to keep consistent with KFold
    else:
        kfold = KFold(n_splits=k_folds, shuffle=True)
        kfold_splits = kfold.split(dataset)

    all_train_losses = []
    all_val_losses = []

    fold = 1

    for train_idx, val_idx in kfold_splits:
        print(f"Fold {fold}/{k_folds}")
        
        # Create data loaders for training and validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn_naive)
        val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn_naive)
        
        # Load model and move to the device
        model = model_class()
        model = model.to(device)

        # Freeze the encoder parameters
        if args.freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False

        model.train()

        # Define optimizer
        optimizer = optimizer_class(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

        # Train the model on the current fold
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=epochs, device=device)
        #evaluate_model(model,val_loader)

        # Save the model for this fold
        torch.save(model.state_dict(), os.path.join(output_path,f"{model_class.__name__}_fold_{fold}_model.pth"))

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        #evaluate()

        fold += 1

    all_train_losses = np.array(all_train_losses)
    all_val_losses = np.array(all_val_losses)
    
    np.save(os.path.join(output_path,"all_train_losses.npy"), all_train_losses)
    np.save(os.path.join(output_path, "all_val_losses.npy"), all_val_losses)

    plot_losses_per_fold(all_train_losses, all_val_losses, k_folds, output_path=output_path)
    # save losses


KNOWN_SOLUTIONS: dict[str, Callable[[], torch.nn.Module]] = {
    "dccrnet": load_dccrnet_model,
}

LOSS_FUNCTIONS: dict[str, Callable[[], torch.nn.Module]] = {
    "snr": ScaleInvariantSignalNoiseRatio,
    "sdr": si_sdr,
    "spec": spectral_convergence_loss,
    "comb": combined_loss
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", choices=KNOWN_SOLUTIONS.keys())
    parser.add_argument("--task", default="1")
    parser.add_argument("--level", default="1")
    parser.add_argument("--data-path", default="data", help="Directory containing downloaded data from the challenge.")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--ir", type=bool, default=False, help="Use IR data.")
    parser.add_argument("--freeze-encoder", type=bool, default=False, help="Freeze the encoder parameters.")
    parser.add_argument("--loss", choices=LOSS_FUNCTIONS.keys())

    args = parser.parse_args()
    data_path = create_data_path(args.data_path, args.task, args.level)
    print(f"Data path: {data_path}")
    output_path = os.path.join(args.data_path, "ml_models", datetime.now().strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(output_path, exist_ok=True)

    start = time.time()

    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    print("Loading dataset...")
    #print(f"Using IR data: {args.ir}")
    dataset = AudioDataset(data_path, aligned=True, ir=args.ir)

    # Define the loss function (SI-SNR)
    loss_fn = LOSS_FUNCTIONS[args.loss]
    if args.loss == "snr":
        loss_fn = ScaleInvariantSignalNoiseRatio().to(device)

    # Cross-validation with K-Folds
    cross_validate(
        model_class=KNOWN_SOLUTIONS[args.model.lower()],
        dataset=dataset,
        optimizer_class=optim.Adam,
        loss_fn=loss_fn,
        k_folds=args.k_folds,
        epochs=args.epochs,
        device=device,
        output_path=output_path,
        freeze_encoder=args.freeze_encoder
    )

    end = time.time()
    print(f"Time taken: {end-start} seconds")
