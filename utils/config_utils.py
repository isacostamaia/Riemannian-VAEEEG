#####################################
# Load config
####################################

import yaml
import re
import os

def _convert_numeric(value):
    if isinstance(value, str):
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?\d*\.?\d+([eE][-+]?\d+)?", value):
            return float(value)
    return value

def _convert_recursive(obj):
    if isinstance(obj, dict):
        return {k: _convert_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_recursive(v) for v in obj]
    else:
        return _convert_numeric(obj)

# def load_config(path):
#     """Load YAML config and auto-convert numeric strings to floats/ints.

#     Paths are always resolved relative to the current working directory.
#     """
#     # Make path absolute relative to cwd if it isn't already
#     if not os.path.isabs(path):
#         path = os.path.abspath(path)  # cwd + relative path

#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Config file not found: {path}")

#     with open(path, "r") as f:
#         cfg = yaml.safe_load(f)
#     return _convert_recursive(cfg)

def load_config(path):
    """Load YAML config and auto-convert numeric strings to floats/ints.

    All relative paths are made absolute relative to the project root,
    so caching and logging directories stay consistent.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = _convert_recursive(cfg)

    # Determine project root (two levels up from this file)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def make_abs(p):
        if isinstance(p, str) and not os.path.isabs(p):
            return os.path.join(project_root, p)
        return p

    # Resolve relevant paths to absolute ones
    if "dataloader" in cfg and "params" in cfg["dataloader"]:
        params = cfg["dataloader"]["params"]
        if "cache_dir" in params:
            params["cache_dir"] = make_abs(params["cache_dir"])
    if "logging" in cfg and "save_path" in cfg["logging"]:
        cfg["logging"]["save_path"] = make_abs(cfg["logging"]["save_path"])

    return cfg



# ===========================================================
# Utility functions
# ===========================================================
import os, yaml, torch, random, numpy as np
from datetime import datetime

from utils.utils import frange_cycle_linear
from utils.vars import SPDVar, EucVecVar
from datasets.eeg_datasets import get_eeg_dataloader_treated_by_domain


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def create_exp_folder(base_path, exp_prefix, exp_name):
    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(base_path, f"{exp_prefix}_{exp_name}_{timestamp}")
    os.makedirs(folder)
    return folder

def get_beta_schedule(schedule_cfg):
    if schedule_cfg["type"] == "frange_cycle_linear":
        return frange_cycle_linear(n_iter=schedule_cfg["n_iter"], stop=schedule_cfg["stop"])
    else:
        raise ValueError(f"Unknown schedule type {schedule_cfg['type']}")

# def get_dataloader(cfg):
#     """Generalized dataloader loader — could extend for other datasets."""
#     ds_cfg = cfg["dataset"]
#     transform = cfg.get("transform", {})

#     if ds_cfg["name"] == "BI2013a":
#         from moabb.datasets import BI2013a
#         from moabb.paradigms import P300
#         return get_eeg_dataloader_treated_by_domain(
#             moabb_dataset=BI2013a(),
#             paradigm=P300(),
#             **{k: v for k, v in cfg.items() if k not in ["dataset", "transform"]},
#             **transform
#         )
#     else:
#         raise NotImplementedError(f"Dataset {ds_cfg['name']} not supported yet.")

import os, yaml, hashlib, pickle
from datasets.eeg_datasets import get_eeg_dataloader_treated_by_domain

def get_config_hash(cfg):
    """Create a unique hash based on dataset parameters."""
    cfg_str = yaml.dump(cfg, sort_keys=True)
    return hashlib.md5(cfg_str.encode('utf-8')).hexdigest()

import os, yaml, hashlib, pickle
from datasets.eeg_datasets import get_eeg_dataloader_treated_by_domain


def get_dataset_cache_path(cache_dir, dataset_cfg):
    """Compute deterministic cache path for this dataset config."""
    config_bytes = str(dataset_cfg).encode("utf-8")
    hash_id = hashlib.md5(config_bytes).hexdigest()[:10]
    cache_path = os.path.join(cache_dir, f"{dataset_cfg['name']}_{dataset_cfg['paradigm']}_{hash_id}.pkl")
    return cache_path

def get_dataloader(cfg):
    """
    Load EEG dataloaders with caching and metadata tracking.
    Compatible with BI2013a / P300 datasets and project conventions.
    """
    ds_cfg = cfg["dataset"]
    transform = cfg.get("transform", {})
    use_cache = cfg.get("use_cache", False)
    cache_dir = cfg.get("cache_dir", "datasets/cache")
    os.makedirs(cache_dir, exist_ok=True)

    # ---- compute deterministic cache path ----
    cache_path = get_dataset_cache_path(cache_dir, ds_cfg)

    # ---- try loading from cache ----
    if use_cache and os.path.exists(cache_path):
        print(f"[INFO] Loading dataset from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            dataset_dict = pickle.load(f)
        metadata = dataset_dict["metadata"]
        print(f"[INFO] Loaded cached dataset ({metadata['dataset_name']} / {metadata['paradigm']})")
        return (
            dataset_dict["train_loader"],
            dataset_dict["val_loader"],
            dataset_dict["test_loader"],
            dataset_dict["test_loader_off"],
            metadata,
        )

    # ---- otherwise build from scratch ----
    print("[INFO] Building dataset from scratch...")

    if ds_cfg["name"] == "BI2013a":
        from moabb.datasets import BI2013a
        from moabb.paradigms import P300

        train_loader, val_loader, test_loader, test_loader_off = get_eeg_dataloader_treated_by_domain(
            moabb_dataset=BI2013a(),
            paradigm=P300(),
            **{k: v for k, v in cfg.items() if k not in ["dataset", "transform", "use_cache", "cache_dir"]},
            **transform,
        )
    else:
        raise NotImplementedError(f"Dataset {ds_cfg['name']} not supported yet.")

    # ---- build metadata dict ----
    metadata = {
        "dataset_name": ds_cfg["name"],
        "paradigm": ds_cfg["paradigm"],
        "transform": transform,
        "cache_used": use_cache,
        "cache_file": cache_path,
    }

    # ---- save to cache ----
    if use_cache:
        print(f"[INFO] Saving dataset to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                    "test_loader": test_loader,
                    "test_loader_off": test_loader_off,
                    "metadata": metadata,
                },
                f,
            )

    return train_loader, val_loader, test_loader, test_loader_off, metadata


def build_var(var_cfg, inferred_shape=None):
    """Instantiate SPDVar or EucVecVar dynamically."""
    cls_name = var_cfg["class"]
    params = var_cfg["params"].copy()
    if isinstance(params.get("distribution"), str):
        from utils.distributions import RiemannianNormal
        name = params["distribution"]
        if name == "RiemannianNormal":
            params["distribution"] = RiemannianNormal
        elif name == "EucGaussian":
            params["distribution"] = name #TODO: fix that later
        else:
            raise ValueError(f"Unknown distribution: {name}")
    if params.get("shape") is None and inferred_shape is not None:
        params["shape"] = inferred_shape
    if params.get("center_at") == "eye" and inferred_shape is not None:
        params["center_at"] = torch.eye(inferred_shape[0])
    if cls_name == "SPDVar":
        return SPDVar(**params)
    elif cls_name == "EucVecVar":
        return EucVecVar(**params)
    else:
        raise ValueError(f"Unknown variable class {cls_name}")
    
def ensure_float(value):
    if isinstance(value, str):
        return float(value)
    return value