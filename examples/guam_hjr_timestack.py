"""GUAM HJ reachability time-stack for Monte-Carlo BRT verification.

Produces value-function snapshots at N_STEPS+1 uniform time points by calling
hj.step() in a Python loop (GPU-memory-safe: each slice is transferred to CPU
before the next step). Output layout under examples/guam_timestack/:

  guam_analysis_config.yml          -- copy for MATLAB brt_setup(read_yml(...))
  {AXIS}_NPY/
    {stem}_stack.npy  float32 (K, n1, n2, n3, n4)
                      index 0 = tau=0 (most evolved BRT)
                      index K-1 = tau=T (initial set)
    {stem}.png        6-subplot all-pairs visualization of the tau=0 slice

The reversed time ordering matches MATLAB value_grad_tv, which expects
  Vslices{1}  -> tau = 0  (most evolved)
  Vslices{K}  -> tau = T  (initial set)
"""

import itertools
import os
import shutil

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

import hj_reachability as hj
from hj_reachability.systems.guam_linear import AXIS_SPEC

# Number of time intervals; total slices = N_STEPS + 1.
# Memory per trim point (lon): (N_STEPS+1) * 47*67*49*53 * 4 B
#   N_STEPS=20 -> ~690 MB,  N_STEPS=50 -> ~1.7 GB.
N_STEPS = 30

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "guam_timestack"

config_path = REPO_ROOT / "config" / "guam_analysis_config.yml"
with open(config_path) as config_file:
    cfg = yaml.safe_load(config_file)
cfg["mat_path"] = str(REPO_ROOT / cfg["mat_path"])
if "quadfit_mat" in cfg:
    cfg["quadfit_mat"] = str(REPO_ROOT / cfg["quadfit_mat"])

hj_cfg = cfg["hj_analysis_config"]
axis = hj_cfg["axis"]
spec = AXIS_SPEC[axis]
axis_cfg = cfg[spec["cfg_key"]]
state_names = spec["state_names"]

grid_lo = jnp.array([axis_cfg[f"grid_min_{name}"] for name in state_names])
grid_hi = jnp.array([axis_cfg[f"grid_max_{name}"] for name in state_names])
grid_shape = tuple(axis_cfg[f"grid_number_{name}"] for name in state_names)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(grid_lo, grid_hi), grid_shape)

target_lo = jnp.array([axis_cfg[f"target_min_{name}"] for name in state_names])
target_hi = jnp.array([axis_cfg[f"target_max_{name}"] for name in state_names])
target_center = (target_hi + target_lo) / 2
target_half = (target_hi - target_lo) / 2
target_dist = jnp.abs(grid.states - target_center) - target_half
values = (jnp.linalg.norm(jnp.maximum(target_dist, 0.), axis=-1)
          + jnp.minimum(jnp.max(target_dist, axis=-1), 0.))

mode = hj_cfg["mode"]
if mode in ("brs", "brt"):
    control_mode, disturbance_mode, time_sign = "min", "max", -1.
elif mode in ("frs", "frt"):
    control_mode, disturbance_mode, time_sign = "max", "min", 1.
else:
    raise ValueError(f"mode must be one of 'brs', 'frs', 'brt', 'frt', got {mode!r}")

solver_kwargs = {}
if mode == "brt":
    solver_kwargs["hamiltonian_postprocessor"] = hj.solver.backwards_reachable_tube
elif mode == "frt":
    solver_kwargs["hamiltonian_postprocessor"] = hj.solver.forwards_reachable_tube
solver_settings = hj.SolverSettings.with_accuracy(hj_cfg["accuracy"], **solver_kwargs)

times = np.linspace(0., time_sign * hj_cfg["time"], N_STEPS + 1)

npy_dir = OUTPUT_DIR / f"{axis.upper()}_NPY"
npy_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(config_path, OUTPUT_DIR / "guam_analysis_config.yml")

wh_idx = hj_cfg["wh_idx"]
for uh_idx in range(hj_cfg["uh_idx_start"], hj_cfg["uh_idx_end"] + 1):
    guam_dynamics = hj.systems.GuamLinear(cfg, uh_idx, wh_idx, axis, control_mode, disturbance_mode)

    # Step through time, transferring each slice to CPU immediately.
    v = values
    slices = [np.asarray(v, dtype=np.float32)]   # index 0 = t=0, tau=T (initial)
    for k in range(N_STEPS):
        v = hj.step(solver_settings, guam_dynamics, grid, times[k], v, times[k + 1],
                    progress_bar=False)
        slices.append(np.asarray(v, dtype=np.float32))
        print(f"  step {k + 1}/{N_STEPS} done", end="\r", flush=True)
    print()

    # Reverse so index 0 = tau=0 (most evolved) to match MATLAB Vslices{1}=tau=0.
    stack = np.stack(slices[::-1])  # (K, n1, n2, n3, n4)

    stem = f"GUAM_{axis.upper()}_{mode.upper()}_UH{uh_idx}_WH{wh_idx}"
    np.save(npy_dir / f"{stem}_stack.npy", stack)

    final_values = slices[-1]   # tau=0, most evolved
    pairs = list(itertools.combinations(range(len(state_names)), 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for (x_dim, y_dim), ax in zip(pairs, axes.ravel()):
        slicer = tuple(slice(None) if dim in (x_dim, y_dim) else n // 2
                       for dim, n in enumerate(grid_shape))
        slice_2d = final_values[slicer].T
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
    fig.savefig(npy_dir / f"{stem}.png", dpi=150)
    plt.close(fig)

    print(f"GUAM_{axis.upper()} UH{uh_idx} WH{wh_idx}: stack {stack.shape} saved to "
          f"{npy_dir / (stem + '_stack.npy')}")
