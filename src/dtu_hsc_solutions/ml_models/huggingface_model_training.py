from argparse import ArgumentParser
from typing import Callable
from torch.utils.data import DataLoader, Subset, ConcatDataset
from .hsc_dataset import AudioDataset, create_aligned_data, collate_fn_naive
from .utils import load_dccrnet_model, create_data_path, si_sdr, spectral_convergence_loss, combined_loss
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
import torch
import torch.optim as optim
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os
import torchaudio
from pathlib import Path
#from ...hsc_given_code.evaluate import evaluate
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
    args.verbose = 0
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

def plot_losses_per_hyperparam(train_losses, val_losses, batch_sizes, learning_rates, output_path="."):
    """Plot the loss curves for each combination of batch size and learning rate."""
    num_combinations = len(train_losses)
    plt.figure(figsize=(12, 8))

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i in range(num_combinations):
        train_loss = train_losses[i]
        val_loss = val_losses[i]
        batch_size = batch_sizes[i]
        learning_rate = learning_rates[i]

        color = colors[i % len(colors)]

        plt.plot(train_loss, label=f"Train BS={batch_size}, LR={learning_rate}", color=color)
        plt.plot(val_loss, '--', label=f"Val BS={batch_size}, LR={learning_rate}", color=color)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses for Hyperparameter Combinations")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "hyperparameter_tuning_loss_curves.png"))

def grid_search_hyperparams(model_class, dataset, optimizer_class, loss_fn, batch_sizes, learning_rates, k_folds=5, epochs=10, device="cpu", output_path="."):
    """Perform a grid search over batch sizes and learning rates."""
    
    all_train_losses = []
    all_val_losses = []
    hyperparam_combinations = []
    
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            print(f"Running with Batch Size = {batch_size}, Learning Rate = {learning_rate}")
            hyperparam_combinations.append((batch_size, learning_rate))
            
            train_losses, val_losses = [], []
            
            if k_folds < 2:
                train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.1, shuffle=True)
                kfold_splits = [(train_idx, val_idx)]
            else:
                kfold = KFold(n_splits=k_folds, shuffle=True)
                kfold_splits = kfold.split(dataset)

            fold = 1

            for train_idx, val_idx in kfold_splits:
                print(f"Fold {fold}/{k_folds}")
                
                # Create data loaders for training and validation
                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, val_idx)
                
                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn_naive)
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn_naive)
                
                # Load model and move to the device
                model = model_class()
                model = model.to(device)

                model.train()

                # Define optimizer
                optimizer = optimizer_class(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

                # Train the model on the current fold
                model, fold_train_losses, fold_val_losses = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=epochs, device=device)

                train_losses.append(fold_train_losses)
                val_losses.append(fold_val_losses)

                fold += 1

            # Store the losses for each combination of hyperparameters
            all_train_losses.append(np.mean(train_losses, axis=0))  # Average over the folds
            all_val_losses.append(np.mean(val_losses, axis=0))
            plot_losses_per_hyperparam(all_train_losses, all_val_losses, [x[0] for x in hyperparam_combinations], [x[1] for x in hyperparam_combinations], output_path=output_path)


    # Plot the loss curves for all hyperparameter combinations
    plot_losses_per_hyperparam(all_train_losses, all_val_losses, [x[0] for x in hyperparam_combinations], [x[1] for x in hyperparam_combinations], output_path=output_path)

    return all_train_losses, all_val_losses

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
    parser.add_argument("--level", default="1", help="If 'all' the dataset is combined with all the levels in the task.")
    parser.add_argument("--data-path", default="data", help="Directory containing downloaded data from the challenge.")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--ir", type=bool, default=False, help="Use IR data.")
    parser.add_argument("--freeze-encoder", type=bool, default=False, help="Freeze the encoder parameters.")
    parser.add_argument("--loss", choices=LOSS_FUNCTIONS.keys())
    parser.add_argument("--hyperparam", type=bool, default=False, help="Tune hyperparameters.")
    parser.add_argument("--name", default=datetime.now().strftime("%Y-%m-%d-%H-%M"))

    args = parser.parse_args()

    output_path = os.path.join(args.data_path, "ml_models", args.name)
    os.makedirs(output_path, exist_ok=True)

    data_paths = []
    if args.level == "all":
        data_path = Path(create_data_path(args.data_path, args.task, "1"))
        parent_folder = data_path.parent.absolute()
        #for folder in the parent folder that starts with Task_{task}
        for folder in os.listdir(parent_folder):
            if folder.startswith(f"Task_{args.task}"):
                data_path = os.path.join(parent_folder, folder)
                data_paths.append(data_path)
    else:
        data_path = create_data_path(args.data_path, args.task, args.level)
        data_paths.append(data_path)
    print("Data paths:")
    for data_path in data_paths:
        print(data_path)

    start = time.time()

    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    print("Loading dataset...")
    datasets = []
    for i, data_path in enumerate(data_paths):
        if os.path.exists(os.path.join(data_path, "Aligned")):
            dataset = AudioDataset(data_path, aligned=True, ir=args.ir)
        else:
            # if aligned folder does not exist, set aligned to False and create aligned data
            print("Aligned data did not exist, so creating aligned data...")
            dataset = AudioDataset(data_path, aligned=False, ir=False)
            create_aligned_data(dataset)
            dataset = AudioDataset(data_path, aligned=True, ir=args.ir)
        datasets.append(dataset)
    comb_dataset = ConcatDataset(datasets)

    # Define the loss function (SI-SNR)
    loss_fn = LOSS_FUNCTIONS[args.loss]
    if args.loss == "snr":
        loss_fn = ScaleInvariantSignalNoiseRatio().to(device)

    # Cross-validation with K-Folds
    if not args.hyperparam:
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
    else:
        # Hyperparameter tuning
        batch_sizes = [4, 8, 16]
        learning_rates = [1e-6]
        grid_search_hyperparams(
            model_class=KNOWN_SOLUTIONS[args.model.lower()],
            dataset=dataset,
            optimizer_class=optim.Adam,
            loss_fn=loss_fn,
            batch_sizes=batch_sizes,
            learning_rates=learning_rates,
            k_folds=args.k_folds,
            epochs=args.epochs,
            device=device,
            output_path=output_path
        )

    end = time.time()
    print(f"Time taken: {end-start} seconds")
