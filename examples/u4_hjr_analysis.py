"""U4 HJ reachability analysis over the trim corridor (port of helperOC's U4_HJIR.m).

For each trim point (tilt angle 0:5:90 deg) this computes the reachable
set/tube of the linearized lon/lat dynamics in deviation coordinates and
saves the final value function to `examples/u4_outputs/`.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

import hj_reachability as hj
from hj_reachability.systems.u4_linear import AXIS_SPEC

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
if mode in ("brt", "frt"):
    solver_kwargs["hamiltonian_postprocessor"] = hj.solver.backwards_reachable_tube
solver_settings = hj.SolverSettings.with_accuracy(hj_cfg["accuracy"], **solver_kwargs)

time = 0.
target_time = time_sign * hj_cfg["time"]

# 2D projection for visualization (remaining dims sliced at the grid center),
# matching the plotDims used in U4_HJIR.m.
plot_dims = (0, 1) if axis == "lon" else (2, 3)

for trim_idx in range(hj_cfg["trim_idx_start"], hj_cfg["trim_idx_end"] + 1):
    u4_dynamics = hj.systems.U4Linear(cfg, trim_idx, axis, control_mode, disturbance_mode)

    target_values = hj.step(solver_settings, u4_dynamics, grid, time, initial_values, target_time)

    tilt_deg = u4_dynamics.tilt_deg
    stem = f"U4_{axis.upper()}_{mode.upper()}_TILT{tilt_deg}"
    np.save(os.path.join(OUTPUT_DIR, f"{stem}.npy"), np.asarray(target_values))
    print(f"Reachability analysis for U4_{axis.upper()} (tilt {tilt_deg} deg) completed.")
    print(f"Results saved to {os.path.join(OUTPUT_DIR, stem + '.npy')}")

    slicer = tuple(slice(None) if dim in plot_dims else n // 2 for dim, n in enumerate(grid_shape))
    x_dim, y_dim = plot_dims
    plt.figure(figsize=(8, 6))
    plt.contourf(grid.coordinate_vectors[x_dim], grid.coordinate_vectors[y_dim],
                    np.asarray(target_values[slicer]).T)
    plt.imshow(np.asarray(target_values[slicer]).T, extent=[grid_lo[0], grid_hi[0], grid_lo[1], grid_hi[1],],
            origin='lower', aspect='auto', cmap='viridis')
    plt.pcolormesh(grid.coordinate_vectors[x_dim], grid.coordinate_vectors[y_dim],
                    np.asarray(target_values[slicer]).T, cmap='viridis', shading='gouraud', vmin=np.asarray(target_values[slicer]).T.min(), vmax=0)
    plt.colorbar()
    plt.contour(grid.coordinate_vectors[x_dim], grid.coordinate_vectors[y_dim],
                np.asarray(target_values[slicer]).T, levels=[0], colors="black", linewidths=2)
    plt.xlabel(state_names[x_dim])
    plt.ylabel(state_names[y_dim])
    plt.title(f"U4 {axis.upper()} {mode.upper()}, tilt {tilt_deg} deg")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{stem}.png"), dpi=150)
    plt.close()
