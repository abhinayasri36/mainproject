# src/utils.py
import os
import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(state: dict, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(filepath: str, model, optimizer=None, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint
