#!/usr/bin/env python3
"""Run Beale SAM with ρ=0.5 and save wp.pt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam_stability.experiments.beale import save_wp_bundle

if __name__ == "__main__":
    path = save_wp_bundle()
    print(f"Saved {path}")
