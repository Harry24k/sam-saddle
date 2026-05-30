"""Log-shifted contour grid for Beale landscape plots."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .benchmarks.problem import Problem


def build_landscape_grid(
    problem: Problem,
    *,
    grid_points: int = 200,
    zeps: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Build (x, y, z) mesh for contour plots with log-scale shifting.

    Returns:
        x, y, z meshes, logzmax, minima column vector
    """
    xmin, xmax = problem.xmin, problem.xmax
    ymin, ymax = problem.ymin, problem.ymax
    xstep = (xmax - xmin) / grid_points
    ystep = (ymax - ymin) / grid_points

    z_min = problem.f(problem.minima).data.numpy()
    x, y = np.meshgrid(
        np.arange(xmin, xmax + xstep, xstep),
        np.arange(ymin, ymax + ystep, ystep),
    )
    z = problem.f([x, y]).data.numpy()

    if z.min() < z_min:
        z_min = z.min()

    z = z + (-z_min + zeps)
    logzmax = np.log(z.max() - z.min() + zeps)
    minima_ = problem.minima.reshape(-1, 1)
    return x, y, z, logzmax, minima_
