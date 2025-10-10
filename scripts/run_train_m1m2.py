#%%
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import geoopt, pickle, matplotlib.pyplot as plt

# --- My imports ---
from utils.config_utils import load_config, set_seed, create_exp_folder, get_beta_schedule, get_dataloader, build_var, ensure_float
from utils.train_functions import train_m1m2
from models.rvae import VAE
from models.m2 import VAEM2

# ===========================================================
# Main script
# ===========================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    #1 - load config
    config = load_config(args.config)
    set_seed(config["seed"])

    exp_params = config["experiment"]
    m1_params = config["model"]["m1"]
    m2_params = config["model"]["m2"]
    optimizer_cfg = config["optimizer"]
    dataloader_cfg = config["dataloader"]["params"]
    var_cfgs = config["variables"]
    log_cfg = config["logging"]

    #2 - schedule
    exp_params["beta"] = get_beta_schedule(exp_params["beta_schedule"])

    # #3 - dataset
    # train_loader, val_loader, test_loader, test_loader_off = get_dataloader(dataloader_cfg)

    #3 - dataset
    #this already computes dataset hash to identify cache version
    train_loader, val_loader, test_loader, test_loader_off, ds_metadata = get_dataloader(dataloader_cfg)


    #4 - infer shape and variables
    data_sample = next(iter(train_loader))
    n = data_sample[0][0].shape[-1]

    xVar = build_var(var_cfgs["xVar"], inferred_shape=(n, n))
    z1Var = build_var(var_cfgs["z1Var"])

    #5 - instantiate models
    domain_keys = train_loader.dataset.metadata.domain_id.unique()
    m1 = VAE(xVar=xVar, zVar=z1Var, domain_keys=domain_keys, **m1_params)
    m2 = VAEM2(**m2_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1, m2 = m1.to(device), m2.to(device)

    #6 - optimizers
    raw_std_params = [layer.raw_std for layer in m1.spd_bn_layers.values()]
    rot_mat_params = [layer.rot_mat for layer in m1.spd_bn_layers.values()]
    all_params = list(m1.parameters()) + list(m2.parameters())
    euc_reg_params = [p for p in all_params if all(id(p) != id(rp) for rp in raw_std_params + rot_mat_params)] #non-fast(regular) euc params

    base_lr = ensure_float(optimizer_cfg["base_lr"])
    fast_lr_mult = ensure_float(optimizer_cfg["fast_lr_mult"])
    riem_lr_mult = ensure_float(optimizer_cfg["riem_lr_mult"])

    optimizer = torch.optim.Adam([
        {'params': euc_reg_params, 'lr': base_lr},
        {'params': raw_std_params, 'lr': base_lr * fast_lr_mult}
    ])
    optimizer_riem = geoopt.optim.RiemannianAdam(rot_mat_params, lr=base_lr * riem_lr_mult)

    #7 - training loop
    loss_dict = {k: [] for k in ['train_loss', 'train_recon_m2', 'train_kl_m2',
                                 'train_yprior', 'train_yclass', 'train_recon_m1', 'train_kl_m1']}
    for epoch in range(m1_params["num_epochs"]):
        train_m1m2(m1, m2, train_loader, exp_params["beta"][epoch],
                   exp_params["alpha"], epoch, loss_dict, optimizer, optimizer_riem)

    #8 - save
    exp_path = create_exp_folder(log_cfg["save_path"], config["exp_prefix"], config["exp_name"])

    #save dataset metadata to experiment folder
    # with open(os.path.join(exp_path, "dataset_metadata.yaml"), "w") as f:
    #     yaml.safe_dump(dataset_metadata, f, sort_keys=False)
    # print(f"[INFO] Saved dataset metadata to {exp_path}/dataset_metadata.yaml")

    metadata_path = os.path.join(exp_path, "dataset_metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(ds_metadata, f)
    print(f"[INFO] Dataset metadata saved to: {metadata_path}")


    #save loss dict
    with open(os.path.join(exp_path, 'loss_dict.p'), 'wb') as fp:
        pickle.dump(loss_dict, fp)
    
    #save loss curve
    if log_cfg["plot_loss"]:
        for l, t in loss_dict.items():
            plt.plot(t, label=l)
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(os.path.join(exp_path, "training_loss.png"))

    #save models
    if log_cfg["save_model"]:
        torch.save(m1.state_dict(), os.path.join(exp_path, "m1.pth"))
        torch.save(m2.state_dict(), os.path.join(exp_path, "m2.pth"))

    print(f"Experiment completed and saved at {exp_path}")

# %%
