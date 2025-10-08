#%%
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random, numpy as np, torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)  # or  torch.set_deterministic(True)

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import time
from datetime import datetime
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score
from umap import UMAP

from moabb.datasets import BI2013a
from moabb.paradigms import P300

from utils.distributions import RiemannianNormal
from models.rvae import VAE
from utils.vars import SPDVar, EucVecVar
from models.m2 import VAEM2_CCVAE
from datasets.eeg_datasets import get_eeg_dataloader_treated_by_domain
from utils.utils import frange_cycle_linear
from analysis.eval_generation.betaprime_twosamples import betaprime, eff_kernel_test, gen_bblock_mats
from utils.train_functions import train_m1m2_CCVAE
from analysis.eval_classification.classification import m1_infer
from analysis.visualizations.plots import plot_2d, plot_3d



if __name__ == "__main__":
    #Set seeds
    seed = 0
    torch.manual_seed(seed)

    # 2
    m1_params = {"num_epochs": 30,
                "hidden_dim": 8, #16,
                "device":torch.device("cuda" if torch.cuda.is_available() else "cpu")
                }
    
    # 1
    exp_params = {"seed": seed,
                  "db_name": "EEG", #"MNIST",
                  "alpha": 1, 
                  "beta": frange_cycle_linear(n_iter=m1_params["num_epochs"], stop=1.0),
                  "MCsamples": 100,
                  #list(reversed(monotonic_schedule(n_iter=m1_params["num_epochs"], stop=1))),
                  #
    #beta_schedule = [beta]*model.num_epochs
    #.5,
                  }
    


    # 3
    optimizer_params ={"learning_rate": 1e-4}

    # 4
    dataloader_params = {"batch_size": 45}
    
    if exp_params["db_name"] == "EEG":
        print("***\nEEG database\n***")
        eeg_dataloader_params = {"moabb_dataset": BI2013a(),  
                                "paradigm": P300(), 
                                "batch_size": dataloader_params["batch_size"],
                                "min_batch_size": 5,
                                }
        eeg_transform_params = {"xdawn_filters":0,
                                "normalize": False, 
                                "estimator": "lwf", 
                                "epsilon": 1.1618
                                }
        train_loader, val_loader, test_loader, test_loader_off = get_eeg_dataloader_treated_by_domain(**eeg_dataloader_params,**eeg_transform_params)

    iterable = iter(train_loader)
    data_sample = next(iterable)
    n = data_sample[0][0].shape[-1] #n dimension of cov mat
    classes = train_loader.dataset.classes
    train_domain_keys = train_loader.dataset.metadata.domain_id.unique()


    #Model and Optimizer
    # 5
    xVar_params = {"distribution":RiemannianNormal,
                "shape": (n,n),
                "center_at": torch.eye(n),
                "min_eigenval": 1e-6
                }
    # 6
    z1Var_params = {"distribution":"EucGaussian",
                "shape": 4, #2, #Latent dim
                "center_at": None, #not needed since for Euclidean vars exponential and log map are skipped
                "batchNorm": None  #only for (truly) observed variables
                }
    
    # 7
    m2_varparams = {"z1_dim": z1Var_params["shape"],
                 "zc_dim": 4, #2,
                 "zbc_dim": 4, #2,
                 "y_dim": len(classes), #number of classes used to do one-hot encoding
                 "hidden_dim": 8,
                 "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")} 

    xVar = SPDVar(**xVar_params)
    z1Var = EucVecVar(**z1Var_params)

    m1 = VAE(xVar = xVar,
             zVar=z1Var,
             domain_keys=train_domain_keys,
             **m1_params).to(m1_params["device"])

 
    m2 = VAEM2_CCVAE(**m2_varparams).to(m2_varparams["device"])
    
    #give increased lr to batch norm params (\nu_\phi) and classif head
    #raw_std parameters (\nu_\phi)
    raw_std_params = [layer.raw_std for layer in m1.spd_bn_layers.values()]
    #classif head
    # cls_params = list(m2.qy_z1_logits.parameters())
    #fastly updated params
    fast_up_params = raw_std_params

    #other parameters
    all_params = list(m1.parameters()) + list(m2.parameters())
    other_params = [p for p in all_params if all(id(p) != id(rp) for rp in fast_up_params)]

    #optimizer with two groups
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': optimizer_params["learning_rate"]},
        {'params': fast_up_params, 'lr': optimizer_params["learning_rate"] * 10}  #bigger LR
    ])

    #loss dictionary
    loss_dict = {'train_loss': [], #total loss
                 'train_recon_m2': [], 'train_m_pzy_m2': [], #m2 loss
                 'train_qzc_m2': [], 'train_yz1_m2': [], 
                 'train_class': [], #m2 loss
                 'train_recon_m1': [],'train_kl_m1': []} #m1 loss
    
    m1m2_varparams = {"m1": [],
                     "m2": []}

    #Training loop
    start = time.time()
    for epoch in range(m1_params["num_epochs"]):
        train_m1m2_CCVAE(m1, m2, train_loader, exp_params["beta"][epoch], exp_params["alpha"], epoch, exp_params["MCsamples"], loss_dict, optimizer, m1m2_varparams)
    end = time.time()
    exp_params["training_time"] = end - start


    #Save results
    params = {"exp_params": exp_params,
            "m1_params": m1_params, 
            "m2_varparams": m2_varparams,
            "optimizer_params": optimizer_params,
            "dataloader_params": dataloader_params,
            "xVar_params": xVar_params,
            "z1Var_params": z1Var_params}
    
    #experience's path
    exp_prefix = "M1+M2_CCVAE"
    def create_exp_folder(db_name):
        os.makedirs(db_name, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(db_name, f"{exp_prefix}_exp_{timestamp}")
        os.makedirs(folder)
        print(f"Created folder: {folder}")
        return folder

    exp_path = create_exp_folder(os.path.join("experiences", exp_params["db_name"]))

    #save loss
    with open(os.path.join(exp_path, 'loss_dict.p'), 'wb') as fp:
        pickle.dump(loss_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    #save training params    
    with open(os.path.join(exp_path, 'training_params.p'), 'wb') as fp:
        pickle.dump(params, fp, protocol=pickle.HIGHEST_PROTOCOL)

    #save training params for facilitating reading
    with open(os.path.join(exp_path, 'report.txt'), "a") as f:
        print("Training params: ", file = f)
        print(params, file = f)
        print("\n\n", file = f)

    #save loss curve
    [plt.plot(t, label=l)  for l,t in loss_dict.items()]
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(os.path.join(exp_path, "training_loss.png"))

    #save model
    torch.save(m1.state_dict(), os.path.join(exp_path, "m1.pth"))
    torch.save(m2.state_dict(), os.path.join(exp_path, "m2.pth"))
    
    m1.eval()
    m2.eval()

    ##M1+M2 train: variables and plots
    x_train, z_train, z_mean_train, labels_train, domains_train, preds_train = m1_infer(m1, train_loader, m2)
    #project and plot 2D train stochastic latent space
    z_train_proj = UMAP(n_components=2, init='random', random_state=0).fit_transform(z_train)
    plot_2d(z_train_proj[:,0], z_train_proj[:,1], labels_train, pred_labels=None, exp_path=exp_path, name="train stochastic latent space classes")
    #project and plot 2D and 3D train embedding space
    z_mean_train_proj = UMAP(n_components=2, init='random', random_state=0).fit_transform(z_mean_train)
    plot_2d(z_mean_train_proj[:,0], z_mean_train_proj[:,1], labels_train, pred_labels=None, exp_path=exp_path, name="train embeddings space classes")
    z_mean_train_proj_3d = UMAP(n_components=3, init='random', random_state=0).fit_transform(z_mean_train)
    plot_3d(z_mean_train_proj_3d[:,0], z_mean_train_proj_3d[:,1], z_mean_train_proj_3d[:,2], labels_train, pred_labels=None, exp_path=exp_path, name="train embeddings space classes")
    
    ##M1+M2 test: variables and plots
    x_test, z_test, z_mean_test, labels_test, domains_test, preds_test = m1_infer(m1, test_loader, m2)
    #project and plot 2D test stochastic latent space
    z_test_proj = UMAP(n_components=2, init='random', random_state=0).fit_transform(z_test)
    plot_2d(z_test_proj[:,0], z_test_proj[:,1], labels_test, pred_labels=None, exp_path=exp_path, name="test stochastic latent space classes")
    #project and plot 2D and 3D test embedding space & include M2 predictions
    z_mean_test_proj = UMAP(n_components=2, init='random', random_state=0).fit_transform(z_mean_test)
    plot_2d(z_mean_test_proj[:,0], z_mean_test_proj[:,1], labels_test, pred_labels=preds_test, exp_path=exp_path, name="M2 predictions on M1 test embeddings space classes")
    z_mean_test_proj_3d = UMAP(n_components=3, init='random', random_state=0).fit_transform(z_mean_test)
    plot_3d(z_mean_test_proj_3d[:,0], z_mean_test_proj_3d[:,1], z_mean_test_proj_3d[:,2], labels_test, pred_labels=preds_test, exp_path=exp_path, name="M2 predictions on M1 test embeddings space classes")

    #1 - Acc, Recall, Precision of y test predictions from M2
    cm = sklearn.metrics.confusion_matrix(labels_test, preds_test)
    with open(os.path.join(exp_path, 'report.txt'), "a") as f:
        print("Classification on test set using y variabe from M2:", file = f)
        print("Confusion Matrix: ", file = f)
        print(cm, file = f)
        print("\n", file=f)
        print(classification_report(labels_test, preds_test, target_names=["NonTarget", "Target"]), file=f)
        print("Balanced Accuracy: ", balanced_accuracy_score(labels_test, preds_test), file=f)
        print("\n\n", file=f)

    #2 - Acc, Recall, Precision of y test predictions from linear probe (classifier trained on z1 train embeddings) 
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')  
    clf.fit(z_mean_train, labels_train)
    preds_test_lp = clf.predict(z_mean_test)
    cm = sklearn.metrics.confusion_matrix(labels_test, preds_test_lp)
    with open(os.path.join(exp_path, 'report.txt'), "a") as f:
        print("Classification using linear probe on M1 train embeddings (and testing on test set)", file = f)
        print("Confusion Matrix: ", file = f)
        print(cm, file = f)
        print("\n", file=f)
        print(classification_report(labels_test, preds_test_lp, target_names=["NonTarget", "Target"]), file=f)
        print("Balanced Accuracy: ", balanced_accuracy_score(labels_test, preds_test_lp), file=f)
        print("\n\n", file=f)
    #plot 2D and 3D test embedding space with Linear probe predictions
    plot_2d(z_mean_test_proj[:,0], z_mean_test_proj[:,1], labels_test, pred_labels=preds_test_lp, exp_path=exp_path, name="Linear Probe predictions on M1 test embeddings space classes")
    plot_3d(z_mean_test_proj_3d[:,0], z_mean_test_proj_3d[:,1], z_mean_test_proj_3d[:,2], labels_test, pred_labels=preds_test_lp, exp_path=exp_path, name="Linear Probe predictions on M1 test embeddings space classes")

    #4 - Betaprime kernel 2-samples test
    with torch.no_grad(): #reconstructed samples from test
        x_test = []
        recons_x_test = []
        for x, y, dom in test_loader:
            m1_varparams = m1(x.to(m1.device), dom) 
            x_test.append(x.cpu().numpy())
            recons_x_test.append(m1_varparams["mu_theta_zi"].cpu().numpy())

    recon_x_test = np.vstack(recons_x_test)
    x_test = np.vstack(x_test)
    #create block matrices:
    l = 5 #size of block covariance matrices
    block_recon = gen_bblock_mats(recon_x_test, l=l).reshape(-1,l,l)
    block_truepop = gen_bblock_mats(x_test, l=l).reshape(-1,l,l)

    result, test_stat, acceptance, upper = eff_kernel_test(betaprime, level=0.05, sampleA=block_recon, sampleB=block_truepop)
    
    with open(os.path.join(exp_path, 'report.txt'), "a") as f:
        print("Kernel 2-samples test:", file=f)
        print("blocks of size=", l, file=f)
        print("result: {}, \n acceptance/test: {}, \n test_stat: {}, acceptance; {}, upper: {}".format(result, acceptance/test_stat, test_stat, acceptance, upper), file = f)
        print("\n\n", file=f)
# %%
