"""Beale function benchmark used in the paper experiments."""

import numpy as np
import torch

from ..config import BEALE_BOUNDS, BEALE_LR, BEALE_MAX_STEPS, X_INIT, Y_INIT
from .problem import Problem, to_tensor


def beales(tensor) -> torch.Tensor:
    x, y = tensor
    x, y = to_tensor(x), to_tensor(y)
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y**2) ** 2 + (2.625 - x + x * y**3) ** 2


def dbeales(tensor) -> torch.Tensor:
    x, y = tensor
    x, y = to_tensor(x), to_tensor(y)
    dx = (
        2 * (x * y**3 - x + 2.625) * (y**3 - 1)
        + 2 * (x * y**2 - x + 2.25) * (y**2 - 1)
        + 2 * (x * y - x + 1.5) * (y - 1)
    )
    dy = (
        6 * (x * y**3 - x + 2.625) * x * y**2
        + 4 * (x * y**2 - x + 2.25) * x * y
        + 2 * (x * y - x + 1.5) * x
    )
    return torch.stack([dx, dy], 1)[0]


def beale_problem() -> Problem:
    return Problem(
        f=beales,
        df=dbeales,
        minima=np.array([3.0, 0.5]),
        bounds=BEALE_BOUNDS,
        x0=[X_INIT, Y_INIT],
        steps=BEALE_MAX_STEPS,
        lr=BEALE_LR,
    )
