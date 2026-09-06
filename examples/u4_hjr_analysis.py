"""U4 HJ reachability analysis over the trim corridor (port of helperOC's U4_HJIR.m).

For each trim point (tilt angle 0:5:90 deg) this computes the reachable
set/tube of the linearized lon/lat dynamics in deviation coordinates and
saves the final value function to `examples/u4_outputs/`.
"""

import itertools
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.io
import yaml

import hj_reachability as hj
from hj_reachability.systems.u4_linear import AXIS_SPEC


def to_cell(items):
    """Pack a Python sequence into a 1xN object array so it becomes a MATLAB cell."""
    cell = np.empty((len(items),), dtype=object)
    for i, item in enumerate(items):
        cell[i] = item
    return cell


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "u4_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

config_path = REPO_ROOT / "config" / "u4_analysis_config.yml"
with open(config_path) as config_file:
    cfg = yaml.safe_load(config_file)
cfg["mat_path"] = str(REPO_ROOT / cfg["mat_path"])

hj_cfg = cfg["hj_analysis_config"]
axis = hj_cfg["axis"]
spec = AXIS_SPEC[axis]
axis_cfg = cfg[spec["cfg_key"]]
state_names = spec["state_names"]

# Grid (deviation coordinates about the trim state)
grid_lo = jnp.array([axis_cfg[f"grid_min_{name}"] for name in state_names])
grid_hi = jnp.array([axis_cfg[f"grid_max_{name}"] for name in state_names])
grid_shape = tuple(axis_cfg[f"grid_number_{name}"] for name in state_names)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(grid_lo, grid_hi), grid_shape)

# Target set: rectangle by corners (signed distance; negative inside),
# equivalent to shapeRectangleByCorners in ToolboxLS.
target_lo = jnp.array([axis_cfg[f"target_min_{name}"] for name in state_names])
target_hi = jnp.array([axis_cfg[f"target_max_{name}"] for name in state_names])
target_center = (target_hi + target_lo) / 2
target_half = (target_hi - target_lo) / 2
target_dist = jnp.abs(grid.states - target_center) - target_half
values = (jnp.linalg.norm(jnp.maximum(target_dist, 0.), axis=-1)
          + jnp.minimum(jnp.max(target_dist, axis=-1), 0.))
initial_values = jnp.asarray(values, dtype=jnp.float32)

# Analysis mode: backward (brs/brt) -> control minimizes, disturbance maximizes;
# forward (frs/frt) -> control maximizes, disturbance minimizes.
mode = hj_cfg["mode"]
if mode in ("brs", "brt"):
    control_mode, disturbance_mode, time_sign = "min", "max", -1.
elif mode in ("frs", "frt"):
    control_mode, disturbance_mode, time_sign = "max", "min", 1.
else:
    raise ValueError(f"mode must be one of 'brs', 'frs', 'brt', 'frt', got {mode!r}")

# Tube modes restrict the Hamiltonian, equivalent to 'minVOverTime' in helperOC.
solver_kwargs = {}
if mode in ("brt"):
    solver_kwargs["hamiltonian_postprocessor"] = hj.solver.backwards_reachable_tube
if mode in ("frt"):
    solver_kwargs["hamiltonian_postprocessor"] = hj.solver.forwards_reachable_tube
solver_settings = hj.SolverSettings.with_accuracy(hj_cfg["accuracy"], **solver_kwargs)

time = 0.
target_time = time_sign * hj_cfg["time"]

for trim_idx in range(hj_cfg["trim_idx_start"], hj_cfg["trim_idx_end"] + 1):
    u4_dynamics = hj.systems.U4Linear(cfg, trim_idx, axis, control_mode, disturbance_mode)

    target_values = hj.step(solver_settings, u4_dynamics, grid, time, initial_values, target_time)

    tilt_deg = u4_dynamics.tilt_deg
    stem = f"U4_{axis.upper()}_{mode.upper()}_TILT{tilt_deg}"
    scipy.io.savemat(
        os.path.join(OUTPUT_DIR, f"{stem}.mat"), {
            "values": np.asarray(target_values, dtype=np.float32),
            "grid_min": np.asarray(grid_lo, dtype=np.float64),
            "grid_max": np.asarray(grid_hi, dtype=np.float64),
            "grid_N": np.asarray(grid_shape, dtype=np.float64),
            "grid_axes": to_cell([np.asarray(cv, dtype=np.float64) for cv in grid.coordinate_vectors]),
            "state_names": to_cell(list(state_names)),
            "tau": float(abs(target_time)),
            "mode": mode,
            "axis": axis,
            "tilt_deg": float(tilt_deg),
        })

    pairs = list(itertools.combinations(range(len(state_names)), 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for (x_dim, y_dim), ax in zip(pairs, axes.ravel()):
        slicer = tuple(slice(None) if dim in (x_dim, y_dim) else n // 2
                       for dim, n in enumerate(grid_shape))
        slice_2d = np.asarray(target_values[slicer]).T
        xs = grid.coordinate_vectors[x_dim]
        ys = grid.coordinate_vectors[y_dim]
        mesh = ax.pcolormesh(xs, ys, slice_2d, cmap="viridis", shading="gouraud",
                             vmin=slice_2d.min(), vmax=0)
        fig.colorbar(mesh, ax=ax)
        ax.contour(xs, ys, slice_2d, levels=[0], colors="black", linewidths=2)
        ax.set_xlabel(state_names[x_dim])
        ax.set_ylabel(state_names[y_dim])
        ax.set_title(f"{state_names[x_dim]} - {state_names[y_dim]}")
    fig.suptitle(stem, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{stem}_pairs.png"), dpi=150)
    plt.close(fig)

    print(f"Reachability analysis for U4_{axis.upper()} (tilt {tilt_deg} deg) completed.")
    print(f"Results saved to {os.path.join(OUTPUT_DIR, stem + '.mat')} and {stem + '_pairs.png'}")
