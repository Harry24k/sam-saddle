from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

X_INIT, Y_INIT = 1.0, 2.0
BEALE_LR = 1e-3
BEALE_BOUNDS = [[-4.5, 4.5], [-4.5, 4.5]]
BEALE_MAX_STEPS = 100_000

# GD (ρ=0) runs until position / loss plateau; other ρ use the same step count
GD_POS_TOL = 0.15          # stop when close to (3, 0.5)
GD_POS_STALL_TOL = 0.55    # allow stall stop only near the minimum
GD_LOSS_PATIENCE = 80
GD_LOSS_RTOL = 1e-8
GD_POS_STALL_WINDOW = 500
GD_POS_STALL_EPS = 1e-5

BASELINE_RHOS = (0.0, 0.1, 0.5, 1.0)
WP_RHO = 0.5
WP_ANNOTATE_T = 5000  # first step shown in wp.pdf; matches paper annotation "t=5000"

METHOD_COLORS = ["b", "g", "r", "y"]
METHOD_LABELS = ["GD", "SAM(0.1)", "SAM", "SAM"]
INTRO_METHOD_INDICES = (0, 2)

WP_TRAJECTORY_KEY = r"momentum_SAM($\rho=0.5$)"

INTRO_XLIM = (-1.5, 3.7)
INTRO_YLIM = (-0.5, 2.7)

# intro.gif: high quality (like original), fewer frames, 4 s total
INTRO_GIF_FRAMES = 16
INTRO_GIF_DURATION_SEC = 4.0
INTRO_GIF_FIGSIZE = (6, 3)
INTRO_GIF_DPI = 100
INTRO_GIF_CONTOUR_LEVELS = 80
