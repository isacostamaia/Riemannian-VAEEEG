#%%
%load_ext autoreload
%autoreload 2

import sys
import os

# Adjust this to point to the folder containing 'playground'
project_root = "/localdata/costamai/Desktop/Work/Code/Riemannian-VAEEEG"
sys.path.append(project_root)

# Identify whether a CUDA-enabled GPU is available
from pathlib import Path
import math
import os
import random
from joblib import Parallel, delayed

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.functional import l1_loss
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset
from braindecode.models import EEGNeX
from eegdash import EEGChallengeDataset

import torch.nn.functional as F
from umap import UMAP

from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from braindecode.datasets import BaseConcatDataset
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    msg ='CUDA-enabled GPU found. Training should be faster.'
else:
    msg = (
        "No GPU found. Training will be carried out on CPU, which might be "
        "slower.\n\nIf running on Google Colab, you can request a GPU runtime by"
        " clicking\n`Runtime/Change runtime type` in the top bar menu, then "
        "selecting \'T4 GPU\'\nunder \'Hardware accelerator\'."
    )
print(msg)

DATA_DIR = Path("/localdata/costamai/Apps/LibData/data_eegdash")


# Creating the path if it does not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# We define the list of releases to load.
# Here, only release 5 is loaded.
release_list = ["R5"]

all_datasets_list = [
    EEGChallengeDataset(
        release=release,
        task="contrastChangeDetection",
        mini=True,
        description_fields=[
            "subject",
            "session",
            "run",
            "task",
            "age",
            "gender",
            "sex",
            "p_factor",
        ],
        cache_dir=DATA_DIR,
        download=False,
    )
    for release in release_list
]
print("Datasets loaded")
sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
          "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]

all_datasets = BaseConcatDataset(all_datasets_list)
print(all_datasets.description)

raws = Parallel(n_jobs=os.cpu_count())(
    delayed(lambda d: d.raw)(d) for d in all_datasets.datasets
)

SFREQ = 100

class DatasetWrapper(BaseDataset):
    def __init__(
        self,
        dataset: EEGWindowsDataset,
        crop_size_samples: int,
        target_name: str = "externalizing",
        seed=None,
    ):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.target_name = target_name
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]

        target = self.dataset.description[self.target_name]
        target = float(target)

        # Additional information:
        infos = {
            "subject": self.dataset.description["subject"],
            "sex": self.dataset.description["sex"],
            "age": float(self.dataset.description["age"]),
            "task": self.dataset.description["task"],
            "session": self.dataset.description.get("session", None) or "",
            "run": self.dataset.description.get("run", None) or "",
        }

        # Randomly crop the signal to the desired length:
        i_window_in_trial, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples, f"{i_stop=} {i_start=}"
        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        i_start = i_start + start_offset
        i_stop = i_start + self.crop_size_samples
        X = X[:, start_offset : start_offset + self.crop_size_samples]

        return X, target, (i_window_in_trial, i_start, i_stop), infos
    
# Filter out recordings that are too short
all_datasets = BaseConcatDataset(
    [
        ds
        for ds in all_datasets.datasets
        if not ds.description.subject in sub_rm
        and ds.raw.n_times >= 4 * SFREQ
        and len(ds.raw.ch_names) == 129
        and not math.isnan(ds.description["p_factor"])
    ]
)

# Create 4-seconds windows with 2-seconds stride
windows_ds = create_fixed_length_windows(
    all_datasets,
    window_size_samples=4 * SFREQ,
    window_stride_samples=2 * SFREQ,
    drop_last_window=True,
)

# Wrap each sub-dataset in the windows_ds
windows_ds = BaseConcatDataset(
    [DatasetWrapper(ds, crop_size_samples=2 * SFREQ) for ds in windows_ds.datasets]
)

x_shape = (129,200)

# Initialize model
# model = EEGNeX(n_chans=129, n_outputs=1, n_times=2 * SFREQ).to(device)
from playground.vanilla_vae import VariationalAutoencoder
latent_dims = 4
n_targets = 1 #regression scalar     #len(data.dataset.targets.unique())
model = VariationalAutoencoder(latent_dims, x_shape, n_targets).to(device) # GPU

# Specify optimizer
optimizer = optim.Adamax(params=model.parameters(), lr=0.002)

print(model)
# Create PyTorch Dataloader

num_workers = 1  # We are using a single worker, but you can increase this for faster data loading
dataloader = DataLoader(windows_ds, batch_size=128, shuffle=True, num_workers=num_workers)


from typing import Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler

# Define a method for training one epoch
def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    device,
    print_batch_stats: bool = True,
):
    model.train()

    total_loss = 0.0
    sum_sq_err = 0.0
    n_samples = 0

    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats
    )

    for batch_idx, batch in progress_bar:
        # Support datasets that may return (X, y) or (X, y, ...)
        X, y, crop_inds, infos = batch
        X = X.to(dtype=torch.float32, device=device)
        y = y.to(dtype=torch.float32, device=device).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Flatten to 1D for regression metrics and accumulate squared error
        preds_flat = preds.detach().view(-1)
        y_flat = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()

        if print_batch_stats:
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            progress_bar.set_description(
                f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}, RMSE: {running_rmse:.6f}"
            )

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
    return avg_loss, rmse

import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from tqdm import tqdm

@torch.no_grad()
def valid_model(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    device,
    print_batch_stats: bool = True,
):
    model.eval()

    total_loss = 0.0
    sum_sq_err = 0.0
    n_batches = len(dataloader)
    n_samples = 0

    iterator = tqdm(
        enumerate(dataloader),
        total=n_batches,
        disable=not print_batch_stats
    )

    for batch_idx, batch in iterator:
        # Supports (X, y) or (X, y, ...)
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()
        # casting X to float32

        preds = model(X)
        batch_loss = loss_fn(preds, y).item()
        total_loss += batch_loss

        preds_flat = preds.detach().view(-1)
        y_flat = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()

        if print_batch_stats:
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            iterator.set_description(
                f"Val Batch {batch_idx + 1}/{n_batches}, "
                f"Loss: {batch_loss:.6f}, RMSE: {running_rmse:.6f}"
            )

    avg_loss = total_loss / n_batches if n_batches else float("nan")
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5

    print(f"Val RMSE: {rmse:.6f}, Val Loss: {avg_loss:.6f}\n")
    return avg_loss, rmse

# Defining training parameters
lr = 1E-3
weight_decay = 1E-5
n_epochs = 15
early_stopping_patience = 50

import copy

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
# loss_fn = torch.nn.MSELoss()
loss_fn = l1_loss

patience = 5
min_delta = 1e-4
best_rmse = float("inf")
epochs_no_improve = 0
best_state, best_epoch = None, None

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}: ", end="")

    train_loss, train_rmse = train_one_epoch(
        dataloader, model, loss_fn, optimizer, scheduler, epoch, device
    )
    # val_loss, val_rmse = valid_model(test_loader, model, loss_fn, device)

    print(
        f"Train RMSE: {train_rmse:.6f}, "
        f"Average Train Loss: {train_loss:.6f}, "
        # f"Val RMSE: {val_rmse:.6f}, "
        # f"Average Val Loss: {val_loss:.6f}"
    )

    # if val_rmse < best_rmse - min_delta:
    #     best_rmse = val_rmse
    #     best_state = copy.deepcopy(model.state_dict())
    #     best_epoch = epoch
    #     epochs_no_improve = 0
    # else:
    #     epochs_no_improve += 1
    #     if epochs_no_improve >= patience:
    #         print(f"Early stopping at epoch {epoch}. Best Val RMSE: {best_rmse:.6f} (epoch {best_epoch})")
    #         break

if best_state is not None:
    model.load_state_dict(best_state)
# %%
