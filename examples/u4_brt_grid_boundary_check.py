"""Check whether the BRT zero-contour is clipped by the exploration grid.

The value function is negative inside the backward reachable tube, zero on its
boundary (the contour), and positive outside. If the value is still negative on
a grid boundary face, the reachable set fills the grid up to that edge and the
true zero-contour lies *outside* the grid, i.e. the grid is too small along
that axis.

For every `.npy` in `examples/u4_outputs/` this reports, per state axis and per
face (low/high), the minimum boundary value and how many boundary cells are
inside the tube (value <= 0).
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path

import numpy as np
import yaml

from hj_reachability.systems.u4_linear import AXIS_SPEC

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "u4_outputs"

with open(REPO_ROOT / "config" / "u4_analysis_config.yml") as config_file:
    cfg = yaml.safe_load(config_file)


def axis_grid(axis):
    """Returns (state_names, coordinate_vectors) for the deviation grid of `axis`."""
    spec = AXIS_SPEC[axis]
    axis_cfg = cfg[spec["cfg_key"]]
    names = spec["state_names"]
    coords = [
        np.linspace(axis_cfg[f"grid_min_{name}"],
                    axis_cfg[f"grid_max_{name}"],
                    axis_cfg[f"grid_number_{name}"])
        for name in names
    ]
    return names, coords


def parse_axis(stem):
    for axis in AXIS_SPEC:
        if f"_{axis.upper()}_" in stem:
            return axis
    raise ValueError(f"cannot determine axis from filename stem {stem!r}")


def check(npy_path):
    stem = npy_path.stem
    axis = parse_axis(stem)
    names, coords = axis_grid(axis)
    values = np.load(npy_path)

    if values.ndim != len(names):
        raise ValueError(f"{stem}: value ndim {values.ndim} != {len(names)} state dims")

    # If nothing is inside the tube at all, there is no contour to clip.
    if values.min() > 0:
        print(f"{stem}: value function is entirely positive (empty BRT) -- no contour.")
        return

    breached = []
    print(f"\n=== {stem}  (axis={axis}, shape={values.shape}) ===")
    print(f"    global min value = {values.min():.4g}")
    for dim, name in enumerate(names):
        for side, idx, edge in (("low", 0, coords[dim][0]), ("high", -1, coords[dim][-1])):
            face = np.take(values, idx, axis=dim)
            fmin = float(face.min())
            inside = int(np.count_nonzero(face <= 0.0))
            total = face.size
            flag = "  <-- BRT reaches this face (contour clipped)" if fmin <= 0.0 else ""
            print(f"    {name:>6} {side:>4} (= {edge:+.3g}): "
                  f"min={fmin:+.4g}, inside {inside}/{total} cells{flag}")
            if fmin <= 0.0:
                breached.append(f"{name}:{side}")

    if breached:
        print(f"    RESULT: contour extends beyond grid on faces -> {', '.join(breached)}")
    else:
        print("    RESULT: BRT fully contained (contour inside grid on all faces).")


if __name__ == "__main__":
    npy_files = sorted(OUTPUT_DIR.glob("*.npy"))
    if not npy_files:
        raise SystemExit(f"no .npy files found in {OUTPUT_DIR}")
    for npy_path in npy_files:
        check(npy_path)
