"""Publication figures for Beale experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

from ..benchmarks.beale import beales
from ..config import (
    INTRO_GIF_CONTOUR_LEVELS,
    INTRO_GIF_DURATION_SEC,
    INTRO_GIF_DPI,
    INTRO_GIF_FIGSIZE,
    INTRO_GIF_FRAMES,
    INTRO_METHOD_INDICES,
    INTRO_XLIM,
    INTRO_YLIM,
    METHOD_COLORS,
    METHOD_LABELS,
    ROOT,
    WP_ANNOTATE_T,
    WP_RHO,
    WP_TRAJECTORY_KEY,
)
from ..io import load_landscape, load_results


def _rho_from_key(name: str) -> float:
    return float(name.split("=")[-1].split("$")[0])


def _compute_wp_trajectory(wt_array: np.ndarray, rho: float) -> np.ndarray:
    """Return w^p_t = w_t + rho * grad(w_t) / ||grad(w_t)|| for each row of wt_array."""
    wp_list = []
    for wt in wt_array:
        wt_t = torch.tensor(wt, dtype=torch.float64, requires_grad=True)
        loss = beales(wt_t)
        loss.backward()
        grad = wt_t.grad.clone()
        gnorm = torch.norm(grad) + 1e-16
        wpt = (wt_t + rho * grad / gnorm).detach().numpy()
        wp_list.append(wpt)
    return np.array(wp_list)


def _load_baseline(root: Path | None = None):
    x, y, z, logzmax, minima_, x0 = load_landscape(root=root)
    bundle = load_results("momentum_00.pt", root=root)
    results = bundle[0]
    gd_steps = int(bundle[2]) if len(bundle) > 2 and bundle[2] else None
    grads_cosine_records = bundle[6] if len(bundle) > 6 else bundle[-1]
    return x, y, z, logzmax, minima_, x0, results, grads_cosine_records, gd_steps


def _wp_slice(n_steps: int, half_window: int = 49) -> slice:
    mid = n_steps // 2
    return slice(mid - half_window, mid + half_window, 2)


def _contour(ax, x, y, z, logzmax, *, levels=40):
    ax.contour(
        x,
        y,
        z,
        levels=np.logspace(0, logzmax // 2.05, levels),
        norm=LogNorm(),
        cmap=plt.cm.jet,
        alpha=0.5,
        zorder=-4,
    )


def plot_grad_cosine(output: str | Path = "grad_cos.pdf", *, root: Path | None = None) -> Path:
    plt.rcParams.update({"font.size": 13})
    *land, results, grads_cosine, _ = _load_baseline(root)
    fig, ax = plt.subplots(figsize=(6, 2))

    for i, name in enumerate(results):
        if i != 2:
            continue
        rho = _rho_from_key(name)
        ax.plot(
            grads_cosine[rho][:50_000],
            color=METHOD_COLORS[i],
            linewidth=1.3,
        )

    ax.yaxis.grid(True, color="gray", linestyle="dashed", linewidth=0.5)
    ax.set_ylabel(r"$\cos(\nabla \ell(w_t), \nabla \ell(w^p_t))$")
    ax.set_xlabel(r"$t$")
    ax.set_ylim(-1.2, 1.2)
    out = Path(root or ROOT) / output
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_intro(output: str | Path = "intro.pdf", *, root: Path | None = None) -> Path:
    plt.rcParams.update({"font.size": 13})
    x, y, z, logzmax, minima_, x0, results, _, _ = _load_baseline(root)
    fig, ax = plt.subplots(figsize=(6, 3))
    _contour(ax, x, y, z, logzmax)
    ax.plot(*minima_, marker="*", c="yellow", markersize=25, markeredgecolor="k", markeredgewidth=1.5, zorder=-1)
    ax.plot(*x0, "+", c="orange", markersize=15, markeredgecolor="k", markeredgewidth=3, zorder=-1)

    for i, name in enumerate(results):
        if i not in INTRO_METHOD_INDICES:
            continue
        path = results[name]
        ax.plot(*path.T, label=METHOD_LABELS[i], linewidth=3, c=METHOD_COLORS[i])
        ax.plot(*path[-1].T, METHOD_COLORS[i] + "o", markersize=8, markeredgecolor="k", zorder=10)

    ax.legend()
    ax.set_xlim(*INTRO_XLIM)
    ax.set_ylim(*INTRO_YLIM)
    out = Path(root or ROOT) / output
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_intro_gif(output: str | Path = "intro.gif", *, root: Path | None = None) -> Path:
    """Intro animation: high-quality contour + blitted paths, 4 s total."""
    x, y, z, logzmax, minima_, x0, results, _, gd_steps = _load_baseline(root)

    paths, colors, labels = [], [], []
    for i, name in enumerate(results):
        if i not in INTRO_METHOD_INDICES:
            continue
        paths.append(results[name])
        colors.append(METHOD_COLORS[i])
        labels.append(METHOD_LABELS[i])

    n_frames = INTRO_GIF_FRAMES
    max_len = gd_steps or max(len(p) for p in paths)
    frame_indices = np.linspace(0, max_len - 1, n_frames, dtype=int)
    fps = n_frames / INTRO_GIF_DURATION_SEC

    fig, ax = plt.subplots(figsize=INTRO_GIF_FIGSIZE, dpi=INTRO_GIF_DPI)
    _contour(ax, x, y, z, logzmax, levels=INTRO_GIF_CONTOUR_LEVELS)
    ax.plot(
        *minima_,
        marker="*",
        c="yellow",
        markersize=25,
        markeredgecolor="k",
        markeredgewidth=1.5,
        zorder=-1,
    )
    ax.plot(*x0, "+", c="orange", markersize=15, markeredgecolor="k", markeredgewidth=3, zorder=-1)
    ax.set_xlim(*INTRO_XLIM)
    ax.set_ylim(*INTRO_YLIM)

    lines, heads = [], []
    for color, label in zip(colors, labels):
        (ln,) = ax.plot([], [], lw=3, c=color, label=label, animated=True)
        (hd,) = ax.plot([], [], color + "o", ms=8, markeredgecolor="k", animated=True)
        lines.append(ln)
        heads.append(hd)
    ax.legend(loc="upper right")

    artists = lines + heads

    def _update(frame_idx: int):
        step = frame_indices[frame_idx]
        for path, ln, hd in zip(paths, lines, heads):
            end = min(step + 1, len(path))
            seg = path[:end]
            ln.set_data(seg[:, 0], seg[:, 1])
            if end:
                hd.set_data([path[end - 1, 0]], [path[end - 1, 1]])
            else:
                hd.set_data([], [])
        return artists

    anim = FuncAnimation(fig, _update, frames=n_frames, blit=True, repeat=False)
    out = Path(root or ROOT) / output
    anim.save(out, writer=PillowWriter(fps=fps), dpi=INTRO_GIF_DPI)
    plt.close(fig)
    return out


def plot_wp(output: str | Path = "wp.pdf", *, root: Path | None = None) -> Path:
    plt.rcParams.update({"font.size": 13})
    x, y, z, logzmax, _minima, _x0, _results, _grads, gd_steps = _load_baseline(root)
    results_wp, *_ = load_results("wp.pt", root=root)

    # `results_wp` stores the w_t trajectory (current weights).  The figure must
    # show w^p_t (perturbed weights), which oscillate visibly between basins even
    # when w_t is essentially pinned at the saddle.  We re-derive w^p_t here using
    # the normalised SAM perturbation formula:  w^p_t = w_t + rho * grad / ||grad||
    # Window: 8 consecutive steps starting at the annotated t=WP_ANNOTATE_T so
    # the alternating basin-crossing behaviour is captured with clear arrows.
    t0 = min(WP_ANNOTATE_T, len(results_wp[WP_TRAJECTORY_KEY]) - 8)
    wt_window = results_wp[WP_TRAJECTORY_KEY][t0: t0 + 8]
    segment = _compute_wp_trajectory(wt_window, rho=WP_RHO)

    fig, ax = plt.subplots(figsize=(3, 3))
    _contour(ax, x, y, z, logzmax)
    ax.axhline(1, c="gray", linestyle="--")
    ax.axvline(0, c="gray", linestyle="--")

    x_, y_ = segment[:, 0], segment[:, 1]
    ax.quiver(
        x_[:-1],
        y_[:-1],
        x_[1:] - x_[:-1],
        y_[1:] - y_[:-1],
        scale_units="xy",
        angles="xy",
        scale=1.1,
        color="k",
        headwidth=10,
        headlength=7,
    )
    ax.plot(x_[1:], y_[1:], "ro", ms=8, markeredgecolor="k", label=r"$w^p_t$")
    ax.plot(x_[0], y_[0], "ro", ms=8, markeredgecolor="k")
    ax.text(
        x_[0] - 0.6,
        y_[0] - 0.35,
        r"$t=5000$",
        fontdict={"weight": "heavy"},
        size=12,
        style="italic",
        bbox={"facecolor": "white", "edgecolor": "darkviolet", "linewidth": 2},
    )
    ax.quiver(
        x_[0] - 0.5,
        y_[0] - 0.35,
        0.5,
        0.35,
        scale_units="xy",
        angles="xy",
        scale=1.1,
        color="darkviolet",
        headwidth=7,
        headlength=7,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 2)
    ax.legend(labelspacing=0.01, borderpad=0.3, handlelength=0.8, prop={"size": 14})
    out = Path(root or ROOT) / output
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_all_paper_figures(*, root: Path | None = None) -> None:
    plot_grad_cosine(root=root)
    plot_intro(root=root)
    plot_intro_gif(root=root)
    plot_wp(root=root)
