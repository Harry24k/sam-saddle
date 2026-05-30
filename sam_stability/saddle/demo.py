"""Saddle function f(x,y)=x²−y²: GD vs SAM trajectories (Figure traj.pdf)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..config import ROOT

X0, Y0 = -3.0, -0.01
GRID = dict(xmin=-4.5, xmax=4.5, ystep=0.2)


def saddle(x: float, y: float) -> float:
    return x**2 - y**2


def get_trajectory(
    *,
    epochs: int = 500,
    lr: float = 0.01,
    gamma: float = 0.0,
    rho: float = 0.0,
    x0: float = X0,
    y0: float = Y0,
) -> Tuple[List[float], List[float], List[float]]:
    path_x, path_y, path_fxy = [], [], []
    x_torch = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    y_torch = torch.tensor(y0, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([x_torch, y_torch], lr=lr, momentum=gamma)

    for _ in range(epochs):
        loss = x_torch**2 - y_torch**2
        loss.backward()

        if rho > 0:
            with torch.no_grad():
                x_grad = x_torch.grad.clone().detach()
                y_grad = y_torch.grad.clone().detach()
                norm = torch.norm(torch.stack([x_grad, y_grad]), p=2)
                x_eps = rho / norm * x_grad
                y_eps = rho / norm * y_grad
                x_torch.add_(x_eps)
                y_torch.add_(y_eps)
            loss_p = x_torch**2 - y_torch**2
            loss_p.backward()

        optimizer.step()

        with torch.no_grad():
            if rho > 0:
                x_torch.sub_(x_eps)
                y_torch.sub_(y_eps)
            x_numpy = x_torch.detach().numpy()
            y_numpy = y_torch.detach().numpy()
            path_x.append(float(x_numpy))
            path_y.append(float(y_numpy))
            path_fxy.append(float(saddle(x_numpy, y_numpy)))

        x_torch.grad.zero_()
        y_torch.grad.zero_()

    return path_x, path_y, path_fxy


def plot_trajectory_figure(
    output: str | Path = "traj.pdf",
    *,
    root: Path | None = None,
) -> Path:
    plt.rcParams.update({"font.size": 15})
    xmin, xmax, ystep = GRID["xmin"], GRID["xmax"], GRID["ystep"]
    x, y = np.meshgrid(
        np.arange(xmin, xmax + ystep, ystep),
        np.arange(xmin, xmax + ystep, ystep),
    )
    z = saddle(x, y)

    sgd = get_trajectory(gamma=0.0, lr=0.01, rho=0.0)
    sam = get_trajectory(gamma=0.0, lr=0.01, rho=0.5)

    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis", alpha=0.9, zorder=6)
    ax.set_box_aspect(aspect=(1.6, 1.6, 1))

    ax.plot([X0], [Y0], [saddle(X0, Y0)], "+", color="k", markersize=23, zorder=4, markeredgewidth=1.5)
    for (path, color, label) in zip(
        (sgd, sam),
        ("b", "r"),
        ("GD", "SAM"),
    ):
        px, py, pf = map(np.array, path)
        ax.plot(px, py, pf, linewidth=2.5, color=color, zorder=3, label=label)
        ax.plot([px[-1]], [py[-1]], [pf[-1]], "o", markersize=8, color=color, zorder=5)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$f(x,y)$")
    plt.legend(bbox_to_anchor=(0.95, 0.88), framealpha=1)
    plt.tight_layout()
    out = Path(root or ROOT) / output
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
