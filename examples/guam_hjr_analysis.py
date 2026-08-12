"""GUAM HJ reachability analysis over the trim table (port of helperOC's GUAM_HJIR.m).

For each horizontal-speed trim point (uh_idx, at a fixed wh_idx) this computes
the reachable set/tube of the linearized lon/lat dynamics in deviation
coordinates and saves the final value function to `examples/guam_outputs/`.
"""

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
from hj_reachability.systems.guam_linear import AXIS_SPEC

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guam_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

config_path = REPO_ROOT / "config" / "guam_analysis_config.yml"
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
# matching the plotDims used in GUAM_HJIR.m: lon -> (u, w), lat -> (r, phi).
plot_dims = (0, 1) if axis == "lon" else (2, 3)

# GUAM wind/gust disturbance bounds per uh_idx trim point (column j <-> uh_idx
# j+1), from disturbance_lb_ub_10mps.mat. Rows are body axes: dF (ft/s^2) ->
# [ax, ay, az], dM (rad/s^2) -> [roll, pitch, yaw]. Each state is driven by the
# body force/moment acting on it; the attitude states (theta/phi) have none.
dist_mat = scipy.io.loadmat(REPO_ROOT / "hj_reachability" / "systems" / "guam_disturbance_lb_ub_10mps.mat")
dist_source = {
    "lon": (("dF", 0), ("dF", 2), ("dM", 1), None),  # u, w, q, theta
    "lat": (("dF", 1), ("dM", 0), ("dM", 2), None),  # v, p, r, phi
}[axis]

wh_idx = hj_cfg["wh_idx"]
for uh_idx in range(hj_cfg["uh_idx_start"], hj_cfg["uh_idx_end"] + 1):
    # uh_idx = 20
    col = uh_idx - 1
    dist_lb = [0. if src is None else dist_mat[f"min_{src[0]}"][src[1], col] for src in dist_source]
    dist_ub = [0. if src is None else dist_mat[f"max_{src[0]}"][src[1], col] for src in dist_source]
    disturbance_space = hj.sets.Box(jnp.asarray(dist_lb, dtype=jnp.float32),
                                    jnp.asarray(dist_ub, dtype=jnp.float32))
    guam_dynamics = hj.systems.GuamLinear(cfg, uh_idx, wh_idx, axis, control_mode, disturbance_mode,
                                            disturbance_space=disturbance_space)

    target_values = hj.step(solver_settings, guam_dynamics, grid, time, values, target_time)

    stem = f"GUAM_{axis.upper()}_{mode.upper()}_UH{uh_idx}_WH{wh_idx}"
    np.save(os.path.join(OUTPUT_DIR, f"{stem}.npy"), np.asarray(target_values))
    print(f"Reachability analysis for GUAM_{axis.upper()} "
            f"(UH {guam_dynamics.uh:.1f} ft/s, WH {guam_dynamics.wh:.1f} ft/s) completed.")
    print(f"Results saved to {os.path.join(OUTPUT_DIR, stem + '.npy')}")
