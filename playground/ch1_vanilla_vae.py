#%%
%load_ext autoreload
%autoreload 2

import sys
import os

# Adjust this to point to the folder containing 'playground'
project_root = "/localdata/costamai/Desktop/Work/Code/_RiemannianVAE_"
sys.path.append(project_root)

# Identify whether a CUDA-enabled GPU is available
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

from playground.vanilla_vae import VariationalAutoencoder

def train(model, data, device, epochs=20):
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        for s in data:
            x = s[0].to(device).float() # GPU
            y = s[1].to(device).float()
            opt.zero_grad()
            pred = model(x)
            recon_loss = ((x - model.decoder.x_hat)**2).sum()
            # pred_loss = F.cross_entropy(pred, y, reduction='sum')
            pred_loss = F.mse_loss(pred, y, reduction='sum')
            loss = recon_loss + model.encoder.kl + pred_loss
            loss.backward()
            opt.step()

            # #batch acc
            # preds = pred.argmax(dim=-1)
            # bacc = (preds == y).sum()/preds.shape[0]

        print(f"btrain loss: {loss.item()}, brecon_loss: {recon_loss.item()}, bkl loss: {model.encoder.kl}, bpred_loss:{pred_loss.item()}")
    return model

def plot_latent(model, data, num_batches=100):
    for i, s in enumerate(data):
        z = model.encoder(s[0].to(device))
        z = z.to('cpu').detach().numpy()
        if z.shape[-1] > 2:
            z = UMAP(n_components=2, init='random', random_state=0).fit_transform(z)
        plt.scatter(z[:, 0], z[:, 1], c=s[1], cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.title("Latent space projection colored by regression target value")

def evaluate(model, test_data, device):
    model.eval()
    mse = 0
    n_samples = 0
    with torch.no_grad():
        for s in test_data:
            x = s[0].to(device).float()
            y = s[1].to(device).float()
            pred = model(x)
            mse += F.mse_loss(pred, y, reduction='sum')
            
    mse /= pred.shape[0]
    print(f"test MSE: {mse}")
    return mse


latent_dims = 4
n_targets = 1 #regression scalar     #len(data.dataset.targets.unique())
vae = VariationalAutoencoder(latent_dims, x_shape, n_targets).to(device) # GPU
vae = train(vae, train_loader, device)
plot_latent(vae, train_loader)
plt.show()
plt.close()
# plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
test_acc = evaluate(vae, test_loader, device)


# def train(model, data, device, epochs=50):
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

# def plot_latent(model, data, num_batches=100):
#     for i, s in enumerate(data):
#         z = model.encoder(s[0].to(device).float())
#         z = z.to('cpu').detach().numpy()
#         if z.shape[-1] > 2:
#             z = UMAP(n_components=2, init='random', random_state=0).fit_transform(z)
#         plt.scatter(z[:, 0], z[:, 1], c=s[1], cmap='tab10')
#         if i > num_batches:
#             plt.colorbar()
#             break


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
            
#     mse /= pred.shape[0]
#     print(f"MSE: {mse}")
#     return mse


# latent_dims = 4
# n_targets = 1 #regression scalar     #len(data.dataset.targets.unique())
# vae = VariationalAutoencoder(latent_dims, n_targets).to(device) # GPU
# vae = train(vae, train_loader, device)
# plot_latent(vae, train_loader)
# plt.show()
# plt.close()
# # plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
# test_acc = evaluate(vae, test_loader, device)
# %%
