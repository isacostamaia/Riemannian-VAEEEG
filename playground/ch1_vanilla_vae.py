#%%
%load_ext autoreload
%autoreload 2

import sys
import os

# Adjust this to point to the folder containing 'playground'
project_root = "/localdata/costamai/Desktop/Work/Code/Riemannian-VAEEEG"
sys.path.append(project_root)

# Identify whether a CUDA-enabled GPU is available
import numpy as np
import torch
from pathlib import Path
from eegdash.dataset import EEGChallengeDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)
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


dataset_ccd = EEGChallengeDataset(task="contrastChangeDetection",
                                  release="R5", cache_dir=DATA_DIR,
                                  mini=True,
                                  download=False)


EPOCH_LEN_S = 2.0
SFREQ = 100 # by definition here

transformation_offline = [
    Preprocessor(
        annotate_trials_with_target,
        target_field="rt_from_stimulus", epoch_length=EPOCH_LEN_S,
        require_stimulus=True, require_response=True,
        apply_on_array=False,
    ),
    Preprocessor(add_aux_anchors, apply_on_array=False),
]
preprocess(dataset_ccd, transformation_offline, n_jobs=1)

ANCHOR = "stimulus_anchor"

SHIFT_AFTER_STIM = 0.5
WINDOW_LEN       = 2.0

# Keep only recordings that actually contain stimulus anchors
dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)

# Create single-interval windows (stim-locked, long enough to include the response)
single_windows = create_windows_from_events(
    dataset,
    mapping={ANCHOR: 0},
    trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),                 # +0.5 s
    trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),   # +2.5 s
    window_size_samples=int(EPOCH_LEN_S * SFREQ),
    window_stride_samples=SFREQ,
    preload=True,
)

# Injecting metadata into the extra mne annotation.
single_windows = add_extras_columns(
    single_windows,
    dataset,
    desc=ANCHOR,
    keys=("target", "rt_from_stimulus", "rt_from_trialstart",
          "stimulus_onset", "response_onset", "correct", "response_type")
          )

meta_information = single_windows.get_metadata()

valid_frac = 0.1
test_frac = 0.1
seed = 2025

subjects = meta_information["subject"].unique()
sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
          "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]
subjects = [s for s in subjects if s not in sub_rm]

train_subj, valid_test_subject = train_test_split(
    subjects, test_size=(valid_frac + test_frac), random_state=check_random_state(seed), shuffle=True
)

valid_subj, test_subj = train_test_split(
    valid_test_subject, test_size=test_frac, random_state=check_random_state(seed + 1), shuffle=True
)
# sanity check
assert (set(valid_subj) | set(test_subj) | set(train_subj)) == set(subjects)

# and finally using braindecode split function, we can do:
subject_split = single_windows.split("subject")

train_set = []
valid_set = []
test_set = []

for s in subject_split:
    if s in train_subj:
        train_set.append(subject_split[s])
    elif s in valid_subj:
        valid_set.append(subject_split[s])
    elif s in test_subj:
        test_set.append(subject_split[s])

train_set = BaseConcatDataset(train_set)
valid_set = BaseConcatDataset(valid_set)
test_set = BaseConcatDataset(test_set)

print("Number of examples in each split in the minirelease")
print(f"Train:\t{len(train_set)}")
print(f"Valid:\t{len(valid_set)}")
print(f"Test:\t{len(test_set)}")

# Create datasets and dataloaders
from torch.utils.data import DataLoader

batch_size = 64
num_workers = 1 # We are using a single worker, but you can increase this for faster data loading

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

x_shape = (129,200)

# def train(model, data, device, epochs=20):
#     opt = torch.optim.Adam(model.parameters())
#     for epoch in range(epochs):
#         for s in data:
#             x = s[0].to(device).float() # GPU
#             y = s[1].to(device).float()
#             opt.zero_grad()
#             pred = model(x)
#             recon_loss = ((x - model.decoder.x_hat)**2).sum()
#             # pred_loss = F.cross_entropy(pred, y, reduction='sum')
#             pred_loss = F.mse_loss(pred, y, reduction='sum')
#             loss = recon_loss + model.encoder.kl + pred_loss
#             loss.backward()
#             opt.step()

#             # #batch acc
#             # preds = pred.argmax(dim=-1)
#             # bacc = (preds == y).sum()/preds.shape[0]

#         print(f"btrain loss: {loss.item()}, brecon_loss: {recon_loss.item()}, bkl loss: {model.encoder.kl}, bpred_loss:{pred_loss.item()}")
#     return model

def plot_latent_by_subject(model, loader):
    #get subject ids from metadata
    meta = loader.dataset.get_metadata()
    subjects = np.array(meta["subject"])
    unique_subjects = np.unique(subjects)
    subject_to_int = {s: i for i, s in enumerate(unique_subjects)}

    # choose a colored colormap
    cmap = plt.cm.get_cmap("tab20", len(unique_subjects))

    # collect latent embeddings for a few batches
    all_z, all_subj = [], []
    with torch.no_grad():
        for batch_indices, s in zip(loader.batch_sampler, loader):
            x = s[0]
            y = s[1]
            mu, z = model.encoder(x.to(device))
            mu = mu.cpu().detach().numpy()

            batch_subjects = [subject_to_int[subjects[j]] for j in batch_indices]
            all_z.append(mu)
            all_subj.append(batch_subjects)
            # if len(all_z) > 10:
            #     break

    all_z = np.concatenate(all_z)
    all_subj = np.concatenate(all_subj)

    if z.shape[-1] > 2:
        all_z = UMAP(n_components=2, random_state=0).fit_transform(all_z)

    plt.figure(figsize=(6,5))
    sc = plt.scatter(all_z[:,0], all_z[:,1], c=all_subj, cmap=cmap, alpha=0.8)
    cbar = plt.colorbar(sc, ticks=range(len(unique_subjects)))
    cbar.ax.set_yticklabels(unique_subjects)
    plt.title("Latent space by subject")
    plt.show()

# def evaluate(model, test_data, device):
#     model.eval()
#     mse = 0
#     n_samples = 0
#     with torch.no_grad():
#         for s in test_data:
#             x = s[0].to(device).float()
#             y = s[1].to(device).float()
#             pred = model(x)
#             mse += F.mse_loss(pred, y, reduction='sum')
#             n_samples += pred.shape[0]
            
#     mse /= n_samples
#     print(f"test MSE: {mse}")
#     return mse

from playground.vanilla_vae import VariationalAutoencoder
from playground.eegnex_vae import EEGNeXVariationalAutoencoder
from playground.higgs_vae import HiggsVariationalAutoencoder

latent_dims = 45
n_targets = 1 #regression scalar     #len(data.dataset.targets.unique())
# model = VariationalAutoencoder(latent_dims, x_shape, n_targets).to(device) # GPU
model = EEGNeXVariationalAutoencoder(latent_dims, x_shape, n_targets).to(device) # GPU
# model = HiggsVariationalAutoencoder(latent_dims, x_shape, n_targets).to(device) # GPU


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
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = loss_fn(preds, y)
        recon_loss = ((X - model.decoder.x_hat)**2).sum()
        regul_loss = model.encoder.kl
        loss = loss + recon_loss + regul_loss

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
n_epochs = 100
early_stopping_patience = 50

import copy

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
loss_fn = torch.nn.MSELoss()

patience = 5
min_delta = 1e-4
best_rmse = float("inf")
epochs_no_improve = 0
best_state, best_epoch = None, None

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}: ", end="")

    train_loss, train_rmse = train_one_epoch(
        train_loader, model, loss_fn, optimizer, scheduler, epoch, device
    )
    val_loss, val_rmse = valid_model(test_loader, model, loss_fn, device)

    print(
        f"Train RMSE: {train_rmse:.6f}, "
        f"Average Train Loss: {train_loss:.6f}, "
        f"Val RMSE: {val_rmse:.6f}, "
        f"Average Val Loss: {val_loss:.6f}"
    )

    if val_rmse < best_rmse - min_delta:
        best_rmse = val_rmse
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val RMSE: {best_rmse:.6f} (epoch {best_epoch})")
            break

if best_state is not None:
    model.load_state_dict(best_state)


from sklearn.metrics import root_mean_squared_error as rmse
@torch.no_grad()
def test_model(
    dataloader: DataLoader,
    model: Module,
    # loss_fn,
    device,
    print_batch_stats: bool = True,
):
    model.eval()
    
    n_batches = len(dataloader)

    iterator = tqdm(
        enumerate(dataloader),
        total=n_batches,
        disable=not print_batch_stats
    )
    preds_ = []
    y_= []

    for batch_idx, batch in iterator:
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()
        # casting X to float32

        preds = model(X)
        preds_.append(preds.detach().cpu().numpy())
        y_.append(y.detach().cpu().numpy())
    preds_ = np.concatenate(preds_)
    y_ = np.concatenate(y_)
    score = rmse(y_, preds_) / np.std(y_)
    print(f"Test score: {score:.6f}")
    return score

test_score = test_model(valid_loader, model, device)


# %%
plot_latent_by_subject(model, train_loader)
plt.show()
plt.close()
# %%
