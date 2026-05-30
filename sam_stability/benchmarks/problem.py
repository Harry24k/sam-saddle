"""2D optimization problem container."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Union

import numpy as np
import torch
from torch.autograd import Variable

Number = Union[float, int, np.generic]
TensorFn = Callable[[Sequence], torch.Tensor]


def to_tensor(x, dtype=np.float64) -> Variable:
    if isinstance(x, np.ndarray):
        return Variable(torch.Tensor(x.astype(dtype)))
    if isinstance(x, list):
        if x and isinstance(x[0], np.ndarray):
            return Variable(torch.from_numpy(np.stack(x).astype(dtype)))
        return Variable(torch.Tensor(x))
    if isinstance(x, (float, int, np.generic)):
        return Variable(torch.Tensor([float(x)]))
    return x


class Problem:
    """Binds objective, gradient, bounds, and default optimizer hyperparameters."""

    def __init__(
        self,
        f: TensorFn,
        df: TensorFn,
        minima: np.ndarray,
        x0: Sequence[float],
        bounds: List[List[float]] = None,
        lr: float = 1e-3,
        steps: int = 3000,
        noise: Dict[str, float] = None,
    ):
        noise = noise or dict(m=0, c=0)

        def f_noise(t):
            t = to_tensor(t)
            m = 1 + Variable(torch.rand(t[0].size()) * noise["m"])
            c = Variable(torch.rand(t[0].size()) * noise["c"])
            return m * f(t) + c

        self._f = f
        self.f = f_noise
        self.df = df
        self.x0 = list(x0)
        self.bounds = bounds or [[-5, 5], [-5, 5]]
        self.minima = np.asarray(minima)
        self.lr = lr
        self.steps = steps

        self.xmin, self.xmax = self.bounds[0]
        self.ymin, self.ymax = self.bounds[1]
