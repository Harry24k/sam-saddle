#!/usr/bin/env python3
"""Run Beale baseline (all SAM ρ) and save momentum_00.pt + exp_settings.pt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam_stability.experiments.beale import save_baseline_bundle

if __name__ == "__main__":
    path = save_baseline_bundle()
    print(f"Saved {path}")
