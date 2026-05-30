"""Optimization loop for 2D benchmark problems."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Type

import numpy as np
import torch
from torch.autograd import Variable

from .sam import SAM, SamMetrics

TensorFn = Callable[[Sequence], torch.Tensor]
Constructor = Callable[[List], torch.optim.Optimizer]


def run_optimizer(
    f: TensorFn,
    constructor: Constructor,
    *,
    steps: Optional[int] = None,
    x0: Sequence[float] = (-4, -1),
    minima: Optional[np.ndarray] = None,
    until_converged: bool = False,
    max_steps: int = 100_000,
    pos_tol: float = 0.15,
    pos_stall_tol: float = 0.55,
    loss_patience: int = 80,
    loss_rtol: float = 1e-8,
    pos_stall_window: int = 500,
    pos_stall_eps: float = 1e-5,
    minimizer: Optional[Type[SAM]] = None,
    minimizer_kwargs: Optional[dict] = None,
    metrics: Optional[SamMetrics] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    minimizer_kwargs = minimizer_kwargs or {}
    params = Variable(torch.Tensor(x0), requires_grad=True)
    optimizer = constructor([params])
    sam_wrapper = None
    if minimizer is not None:
        sam_wrapper = minimizer(optimizer, [params], metrics=metrics, **minimizer_kwargs)

    def eval_step():
        loss = f(params)
        loss.backward()
        return loss

    def step_once():
        if sam_wrapper is not None:
            loss = eval_step()
            sam_wrapper.ascent_step()
            if metrics is not None:
                metrics.record_ascent_loss(sam_wrapper.rho, loss)

            loss = eval_step()
            if metrics is not None:
                metrics.record_descent_loss(sam_wrapper.rho, loss)
            sam_wrapper.descent_step()
        else:
            optimizer.zero_grad()
            loss = optimizer.step(eval_step)
        return loss

    data: List[np.ndarray] = []
    dist: List[float] = []
    stall = 0
    last_loss: Optional[float] = None
    minima_arr = np.asarray(minima) if minima is not None else None

    for t in range(max_steps):
        loss_val = float(step_once().squeeze().data.numpy())
        dist.append(loss_val)
        data.append(params.data.numpy().copy())

        if until_converged and minima_arr is not None:
            dist_now = float(np.linalg.norm(data[-1] - minima_arr))
            if dist_now < pos_tol:
                break
            if dist_now < pos_stall_tol and len(data) >= pos_stall_window:
                window = np.array(data[-pos_stall_window:])
                moves = np.linalg.norm(np.diff(window, axis=0), axis=1)
                if np.max(moves) < pos_stall_eps:
                    break
            if last_loss is not None:
                rel = abs(loss_val - last_loss) / (abs(last_loss) + 1e-12)
                stall = stall + 1 if rel < loss_rtol else 0
                if stall >= loss_patience:
                    break
            last_loss = loss_val

        if steps is not None and t + 1 >= steps:
            break

    return np.array(data), np.array(dist)
