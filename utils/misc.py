
import os
import random
import numpy as np
import torch
import yaml

def set_seed(seed: int = 42):
    """Fix random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_yaml_config(path: str):
    with open(path, "r", encoding="utf‑8") as f:
        return yaml.safe_load(f)

def save_checkpoint(model, optimizer, epoch, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(ckpt, save_path)

def load_checkpoint(model, ckpt_path, optimizer=None, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"]