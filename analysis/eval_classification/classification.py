import os
import sys
from pathlib import Path

# Automatically get the root of the project (one level up from this script)
project_root = Path(__file__).resolve().parents[1]

#Add root to sys.path so you can import rvae, datasets, etc.
sys.path.append(str(project_root))

#Change working directory to root (for file paths)
os.chdir(project_root)

import numpy as np
import torch

def m1_infer(m1, data_loader, m2=None):
    """
    if m2 not provided, preds=[]
    """
    m1.eval()
    if m2:
        m2.eval()
    x_ = []
    z_ = []
    z_mean_ = []
    labels = []
    domains = []
    preds = []
    with torch.no_grad():
        for x, y, dom in data_loader:
            x = x.to(m1.device)
            mean, logvar = m1.encode(x, dom)
            z = m1.reparametrize(mean, logvar)
            if m2:
                # logits = m2.qy_z1_logits(mean)
                #TODO: write a function in m2-kingma that does this
                preds_ = m2.infer(mean) #batch_size
                preds.extend(preds_.cpu().numpy().tolist())
            x_.append(x.squeeze().cpu().numpy())
            z_.append(z.squeeze().cpu().numpy())
            z_mean_.append(mean.squeeze().cpu().numpy())
            labels.extend(y)
            domains.extend(dom)

    x_ = np.vstack(x_) #shape = n_samples, N, N 
    z_ = np.vstack(z_) #will be of shape (n_samples, latent_dim)
    z_mean_ = np.vstack(z_mean_) #will be of shape (n_samples, latent_dim)
    labels = np.array(labels)
    domains = np.array(domains)
    preds = np.array(preds)

    return x_, z_, z_mean_, labels, domains, preds