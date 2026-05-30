#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam_stability.saddle.demo import plot_trajectory_figure
from sam_stability.viz.beale import plot_all_paper_figures

if __name__ == "__main__":
    plot_trajectory_figure()
    plot_all_paper_figures()
    print("Wrote traj.pdf, grad_cos.pdf, intro.pdf, intro.gif, wp.pdf")
