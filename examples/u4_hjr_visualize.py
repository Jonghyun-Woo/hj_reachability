"""Visualize U4 HJ reachability results as 2D contours over all state pairs.

Each `.npy` file in `examples/u4_outputs/` holds a 4D value function on the
lon/lat deviation grid. For every pair of the 4 state dimensions (4C2 = 6
pairs) this slices the remaining two dimensions at the grid center (i.e. the
other states held at their trim deviation of 0) and draws a 2D contour of the
value function, marking the zero level set (the reachable set boundary).
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import itertools
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

import hj_reachability as hj
from hj_reachability.systems.u4_linear import AXIS_SPEC

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "u4_outputs"

with open(REPO_ROOT / "config" / "u4_analysis_config.yml") as config_file:
    cfg = yaml.safe_load(config_file)


def build_grid(axis):
    """Rebuilds the deviation grid for `axis` from the analysis config."""
    spec = AXIS_SPEC[axis]
    axis_cfg = cfg[spec["cfg_key"]]
    names = spec["state_names"]
    grid_lo = jnp.array([axis_cfg[f"grid_min_{name}"] for name in names])
    grid_hi = jnp.array([axis_cfg[f"grid_max_{name}"] for name in names])
    grid_shape = tuple(axis_cfg[f"grid_number_{name}"] for name in names)
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(grid_lo, grid_hi), grid_shape)
    return grid, grid_shape, names


def parse_axis(stem):
    """Extracts the axis ('lon'/'lat') from a stem like 'U4_LAT_BRT_TILT90'."""
    for axis in AXIS_SPEC:
        if f"_{axis.upper()}_" in stem:
            return axis
    raise ValueError(f"cannot determine axis from filename stem {stem!r}")


def visualize(npy_path):
    """Draws all 6 state-pair contours of one value function into one figure."""
    stem = npy_path.stem
    axis = parse_axis(stem)
    grid, grid_shape, names = build_grid(axis)
    values = np.load(npy_path)

    pairs = list(itertools.combinations(range(len(names)), 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for (x_dim, y_dim), ax in zip(pairs, axes.ravel()):
        # Hold the other two states at their center index (trim deviation ~ 0).
        slicer = tuple(slice(None) if dim in (x_dim, y_dim) else n // 2
                       for dim, n in enumerate(grid_shape))
        slice_2d = np.asarray(values[slicer]).T
        xs = grid.coordinate_vectors[x_dim]
        ys = grid.coordinate_vectors[y_dim]
        mesh = ax.pcolormesh(xs, ys, slice_2d, cmap="viridis", shading="gouraud",
                             vmin=slice_2d.min(), vmax=0)
        fig.colorbar(mesh, ax=ax)
        ax.contour(xs, ys, slice_2d, levels=[0], colors="black", linewidths=2)
        ax.set_xlabel(names[x_dim])
        ax.set_ylabel(names[y_dim])
        ax.set_title(f"{names[x_dim]} - {names[y_dim]}")

    fig.suptitle(stem, fontsize=14)
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{stem}_pairs.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    npy_files = sorted(OUTPUT_DIR.glob("*.npy"))
    if not npy_files:
        raise SystemExit(f"no .npy files found in {OUTPUT_DIR}")
    for npy_path in npy_files:
        visualize(npy_path)
