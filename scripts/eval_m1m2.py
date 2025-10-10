#%%
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score
from umap import UMAP

# --- My imports ---
from utils.config_utils import load_config, get_dataloader
from utils.vars import SPDVar, EucVecVar
from models.rvae import VAE
from models.m2 import VAEM2
from analysis.eval_classification.classification import m1_infer
from analysis.visualizations.plots import plot_2d, plot_3d
from analysis.eval_generation.iwae import inference_xyIWAE_m1m2_vec
from analysis.eval_generation.betaprime_twosamples import betaprime, eff_kernel_test, gen_bblock_mats


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_path(path):
    if not os.path.isabs(path):
        return os.path.join(PROJECT_ROOT, path)
    return path


# ===========================================================
# Main evaluation
# ===========================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_path", type=str, required=True, help="Path to trained experiment folder (from training output)")


    args = parser.parse_args()
    args.config = resolve_path(args.config)
    args.exp_path = resolve_path(args.exp_path)

    # 1 - Load config
    config = load_config(args.config)
    dataloader_cfg = config["dataloader"]["params"]
    var_cfgs = config["variables"]
    m1_params = config["model"]["m1"]
    m2_params = config["model"]["m2"]
    log_cfg = config["logging"]

    # 2 - Dataset
    #this already computes dataset hash to identify cache version
    train_loader, val_loader, test_loader, test_loader_off, ds_metadata = get_dataloader(dataloader_cfg)


    # 3 - Infer shapes and build variables
    data_sample = next(iter(train_loader))
    n = data_sample[0][0].shape[-1]
    target_names = list(train_loader.dataset.classes)
    n_classes = len(target_names)

    from utils.distributions import RiemannianNormal

    xVar = SPDVar(
        distribution=RiemannianNormal,
        shape=(n, n),
        center_at=torch.eye(n),
        min_eigenval=1e-6
    )
    z1Var = EucVecVar(
        distribution="EucGaussian",
        shape=m2_params["z1_dim"],
        center_at=None,
        batchNorm=None
    )

    # 4 - Load models
    domain_keys = train_loader.dataset.metadata.domain_id.unique()
    m1 = VAE(xVar=xVar, zVar=z1Var, domain_keys=domain_keys, **m1_params)
    m2 = VAEM2(y_dim=n_classes, **m2_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1, m2 = m1.to(device), m2.to(device)

    # Load trained weights
    m1.load_state_dict(torch.load(os.path.join(args.exp_path, "m1.pth"), map_location=device))
    m2.load_state_dict(torch.load(os.path.join(args.exp_path, "m2.pth"), map_location=device))
    m1.eval(); m2.eval()

    print(f"Models loaded from {args.exp_path}")

    # ===========================================================
    # 5 - Evaluation
    # ===========================================================
    exp_path = args.exp_path

    # --- M1+M2 train: variables and plots ---
    x_train, z_train, z_mean_train, labels_train, domains_train, preds_train = m1_infer(m1, train_loader, m2)
    z_train_proj = UMAP(n_components=2, init='random', random_state=0).fit_transform(z_train)
    plot_2d(z_train_proj[:,0], z_train_proj[:,1], labels_train, exp_path=exp_path, name="train_stochastic_latent_space")
    z_mean_train_proj = UMAP(n_components=2, init='random', random_state=0).fit_transform(z_mean_train)
    plot_2d(z_mean_train_proj[:,0], z_mean_train_proj[:,1], labels_train, exp_path=exp_path, name="train_embeddings_space")
    z_mean_train_proj_3d = UMAP(n_components=3, init='random', random_state=0).fit_transform(z_mean_train)
    plot_3d(z_mean_train_proj_3d[:,0], z_mean_train_proj_3d[:,1], z_mean_train_proj_3d[:,2], labels_train, exp_path=exp_path, name="train_embeddings_space_3D")

    # --- M1+M2 test ---
    x_test, z_test, z_mean_test, labels_test, domains_test, preds_test = m1_infer(m1, test_loader, m2)
    z_mean_test_proj = UMAP(n_components=2, init='random', random_state=0).fit_transform(z_mean_test)
    z_mean_test_proj_3d = UMAP(n_components=3, init='random', random_state=0).fit_transform(z_mean_test)
    plot_2d(z_mean_test_proj[:,0], z_mean_test_proj[:,1], labels_test, pred_labels=preds_test, exp_path=exp_path, name="M2_predictions_on_test_embeddings_space")
    plot_3d(z_mean_test_proj_3d[:,0], z_mean_test_proj_3d[:,1], z_mean_test_proj_3d[:,2], labels_test, pred_labels=preds_test, exp_path=exp_path, name="M2_predictions_on_test_embeddings_space_3D")

    #1 - Classification report
    with open(os.path.join(exp_path, 'report.txt'), "a") as f:
        cm = sklearn.metrics.confusion_matrix(labels_test, preds_test)
        print("Classification on test set (M2 predictions):", file=f)
        print(cm, file=f)
        print(classification_report(labels_test, preds_test, target_names=target_names), file=f)
        print("Balanced Accuracy:", balanced_accuracy_score(labels_test, preds_test), file=f)
        print("\n", file=f)

    #2 - Linear Probe
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(z_mean_train, labels_train)
    preds_test_lp = clf.predict(z_mean_test)
    cm = sklearn.metrics.confusion_matrix(labels_test, preds_test_lp)
    with open(os.path.join(exp_path, 'report.txt'), "a") as f:
        print("Linear Probe on M1 embeddings:", file=f)
        print(cm, file=f)
        print(classification_report(labels_test, preds_test_lp, target_names=target_names), file=f)
        print("Balanced Accuracy:", balanced_accuracy_score(labels_test, preds_test_lp), file=f)
        print("\n", file=f)

    #3: νϕ and Rotation matrices
    with open(os.path.join(exp_path, 'report.txt'), "a") as f:
        print("Rotation matrices stats:", file=f)
        for dom_key, layer in m1.spd_bn_layers.items():
            if hasattr(layer, "rot_mat"):
                rot = layer.rot_mat.detach()
                dist = torch.norm(rot - torch.eye(rot.shape[-1], device=rot.device), p='fro')
                print(f"{dom_key}: Frobenius distance from I = {dist.item():.6f}", file=f)
        print("\n", file=f)
        print("νϕ matrices stats:", file=f) 
        for dom_key, layer in m1.spd_bn_layers.items():
            if hasattr(layer, "raw_std"):
                std = torch.nn.functional.softplus(layer.raw_std.detach()) + layer.min_std
                print(f"{dom_key}: νϕ = {std.item():.6f}", file=f)

    #4 - betaprime kernel 2-samples test
    from analysis.eval_generation.betaprime_twosamples import betaprime, eff_kernel_test, gen_bblock_mats
    with torch.no_grad():
        x_test_np, recons_x_test_np = [], []
        for x, y, dom in test_loader:
            m1_varparams = m1(x.to(device), dom)
            x_test_np.append(x.cpu().numpy())
            recons_x_test_np.append(m1_varparams["mu_theta_zi"].cpu().numpy())
    x_test_np = np.vstack(x_test_np)
    recons_x_test_np = np.vstack(recons_x_test_np)
    block_size = 5
    block_recon = gen_bblock_mats(recons_x_test_np, l=block_size).reshape(-1, block_size, block_size)
    block_truepop = gen_bblock_mats(x_test_np, l=block_size).reshape(-1, block_size, block_size)
    result, test_stat, acceptance, upper = eff_kernel_test(betaprime, level=0.05, sampleA=block_recon, sampleB=block_truepop)
    with open(os.path.join(exp_path, 'report.txt'), "a") as f:
        print("Kernel 2-sample test:", file=f)
        print(f"blocks={block_size}, result={result}, test_stat={test_stat}, acceptance={acceptance}, upper={upper}", file=f)
        print("\n", file=f)

    #5 - IWAE
    iwae = inference_xyIWAE_m1m2_vec(m1, m2, test_loader, K=512)
    with open(os.path.join(exp_path, 'report.txt'), "a") as f:
        print("IWAE mean:", np.mean(iwae), "std:", np.std(iwae), file=f)
        print("\n", file=f)

    print(f"Evaluation completed. Results saved to {exp_path}")
