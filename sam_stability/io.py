from pathlib import Path

import torch

from .config import ROOT


def save_pt(data, name: str, root: Path | None = None) -> Path:
    path = (root or ROOT) / name
    torch.save(data, path)
    return path


def load_pt(name: str, root: Path | None = None):
    return torch.load((root or ROOT) / name, weights_only=False)


def save_landscape(data, root: Path | None = None) -> Path:
    return save_pt(data, "exp_settings.pt", root)


save_experiment = save_pt
load_landscape = lambda root=None: load_pt("exp_settings.pt", root)
load_results = load_pt
