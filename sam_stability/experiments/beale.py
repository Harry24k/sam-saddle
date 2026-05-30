"""Run Beale SAM experiments and persist results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.optim as optim
from tqdm import tqdm

from ..benchmarks.beale import beale_problem
from ..config import (
    BASELINE_RHOS,
    BEALE_MAX_STEPS,
    GD_LOSS_PATIENCE,
    GD_LOSS_RTOL,
    GD_POS_STALL_EPS,
    GD_POS_STALL_TOL,
    GD_POS_STALL_WINDOW,
    GD_POS_TOL,
    ROOT,
    WP_RHO,
)
from ..io import load_results, save_experiment, save_landscape
from ..landscape import build_landscape_grid
from ..optim.runner import run_optimizer
from ..optim.sam import SAM, SamMetrics


def _constructor(lr: float):
    return lambda params: optim.SGD(params, lr=lr / 80, momentum=0.0)


def _run_one(problem, rho: float, steps, metrics: SamMetrics, until_converged: bool):
    return run_optimizer(
        problem.f,
        _constructor(problem.lr),
        x0=problem.x0,
        steps=steps,
        minima=problem.minima,
        until_converged=until_converged,
        max_steps=BEALE_MAX_STEPS,
        pos_tol=GD_POS_TOL,
        loss_patience=GD_LOSS_PATIENCE,
        loss_rtol=GD_LOSS_RTOL,
        pos_stall_tol=GD_POS_STALL_TOL,
        pos_stall_window=GD_POS_STALL_WINDOW,
        pos_stall_eps=GD_POS_STALL_EPS,
        minimizer=SAM,
        minimizer_kwargs={"rho": rho},
        metrics=metrics,
    )


def run_sam_experiment(
    rhos: Iterable[float],
    *,
    optimizer_name: str = "momentum",
    show_progress: bool = True,
) -> Tuple[dict, dict, SamMetrics, int]:
    torch.set_default_dtype(torch.float64)
    problem = beale_problem()
    results, distance = {}, {}
    metrics = SamMetrics()
    gd_steps = None

    rho_list = list(rhos)
    iterator = tqdm(rho_list, desc="SAM") if show_progress else rho_list
    for rho in iterator:
        until = rho == 0.0 and gd_steps is None
        data, dist = _run_one(
            problem,
            rho,
            steps=None if until else gd_steps,
            metrics=metrics,
            until_converged=until,
        )
        if until:
            gd_steps = len(data)

        key = f"{optimizer_name}_SAM($\\rho={rho:.1f}$)"
        results[key] = data
        distance[key] = dist

    return results, distance, metrics, gd_steps or 0


def _bundle(results, distance, metrics: SamMetrics, gd_steps: int):
    return (
        results,
        distance,
        gd_steps,
        metrics.loss_descent,
        metrics.loss_ascent,
        {},
        metrics.grad_cosine,
    )


def _gd_steps_from_baseline(root: Path | None) -> int | None:
    try:
        bundle = load_results("momentum_00.pt", root)
        gd_steps = bundle[2]
        return int(gd_steps) if gd_steps else None
    except FileNotFoundError:
        return None


def save_baseline_bundle(root: Path | None = None) -> Path:
    problem = beale_problem()
    x, y, z, logzmax, minima_ = build_landscape_grid(problem)
    results, distance, metrics, gd_steps = run_sam_experiment(BASELINE_RHOS)
    save_experiment(_bundle(results, distance, metrics, gd_steps), "momentum_00.pt", root=root)
    save_landscape([x, y, z, logzmax, minima_, problem.x0], root=root)
    return (root or ROOT) / "momentum_00.pt"


def save_wp_bundle(root: Path | None = None) -> Path:
    problem = beale_problem()
    metrics = SamMetrics()
    gd_steps = _gd_steps_from_baseline(root)
    if gd_steps is None:
        gd_data, _ = _run_one(problem, 0.0, None, metrics, until_converged=True)
        gd_steps = len(gd_data)
    data, dist = _run_one(problem, WP_RHO, gd_steps, metrics, until_converged=False)
    key = f"momentum_SAM($\\rho={WP_RHO:.1f}$)"
    return save_experiment(
        _bundle({key: data}, {key: dist}, metrics, gd_steps),
        "wp.pt",
        root=root,
    )
