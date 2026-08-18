import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("TkAgg")  # interactive backend so plt.show() opens a window (WSLg)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import yaml

import hj_reachability as hj
from hj_reachability.systems.u4_linear import AXIS_SPEC, load_trim_corridor

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "u4_outputs"

with open(REPO_ROOT / "config" / "u4_analysis_config.yml") as config_file:
    cfg = yaml.safe_load(config_file)
cfg["mat_path"] = str(REPO_ROOT / cfg["mat_path"])

RAD2DEG = 180.0 / np.pi
# U4 states are already in SI (m/s); only angular states need rad->deg for display.
# Lon states: [u (m/s), w (m/s), q (rad/s), theta (rad)].
AXIS_SCALE = {"lon": np.array([1.0, 1.0, RAD2DEG, RAD2DEG])}

CONTOUR_COLOR = "b"      # single color for every tube contour
TARGET_COLOR = "green"   # rectangular target set

# U4 considers only the longitudinal BRT (no lateral analysis). Each panel is its
# own figure; 'dims' are the two kept state dims, 'xyz' maps (kept_x, kept_y,
# trim-airspeed height 'h') onto the plot axes. The u-w panel is a plain 2D plot;
# the q-theta panel is 3D with the trim airspeed ('h') on the middle axis,
# stretched to a 1:5:1 box aspect (like MATLAB's pbaspect([1,5,1])).
PANELS = [
    {"title": "Lon: u vs w", "kind": "2d", "axis": "lon", "dims": (0, 1), "file": "u_w",
     "xyz": ("x", "y"), "labels": ("u (m/s)", "w (m/s)")},
    {"title": "Lon: q vs u vs theta", "kind": "3d", "axis": "lon", "dims": (2, 3), "file": "q_u_theta",
     "xyz": ("x", "h", "y"), "labels": ("q (deg/s)", "u_trim (m/s)", "theta (deg)")},
]

# Trim points 1..19 <-> tilt angle 0:5:90 deg (tilt = (trim_idx - 1) * 5).
TRIM_RANGE = range(1, 20)
MODE = "brt"


def build_grid(axis):
    """Rebuilds the deviation grid and its shape for `axis` from the config."""
    spec = AXIS_SPEC[axis]
    axis_cfg = cfg[spec["cfg_key"]]
    names = spec["state_names"]
    grid_lo = jnp.array([axis_cfg[f"grid_min_{name}"] for name in names])
    grid_hi = jnp.array([axis_cfg[f"grid_max_{name}"] for name in names])
    grid_shape = tuple(axis_cfg[f"grid_number_{name}"] for name in names)
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(grid_lo, grid_hi), grid_shape)
    return grid, grid_shape, names


def zero_segments(xa, xb, slice_2d):
    """Zero-level contour of `slice_2d` (dims a,b) as a list of (N,2) arrays."""
    tmp_fig = plt.figure()
    tmp_ax = tmp_fig.add_subplot()
    segs = []
    if slice_2d.min() < 0 < slice_2d.max():
        cs = tmp_ax.contour(xa, xb, slice_2d.T, levels=[0.0])
        segs = [np.asarray(s) for s in cs.allsegs[0] if len(s) > 1]
    plt.close(tmp_fig)
    return segs


def place(kept_x, kept_y, height, xyz):
    """Maps (kept_x, kept_y, trim height) onto the panel's plot axes (2 or 3)."""
    src = {"x": kept_x, "y": kept_y, "h": np.full_like(kept_x, height)}
    return tuple(src[key] for key in xyz)


def target_corners(axis):
    """Target rectangle corners (min, max) in deviation coords, from the config."""
    spec = AXIS_SPEC[axis]
    axis_cfg = cfg[spec["cfg_key"]]
    lo = np.array([axis_cfg[f"target_min_{name}"] for name in spec["state_names"]])
    hi = np.array([axis_cfg[f"target_max_{name}"] for name in spec["state_names"]])
    return lo, hi


def main():
    trim = load_trim_corridor(cfg["mat_path"])["X_TRIM"]
    grid, grid_shape, _ = build_grid("lon")
    tgt_lo, tgt_hi = target_corners("lon")
    scale = AXIS_SCALE["lon"]
    x_state_idx = list(AXIS_SPEC["lon"]["state_idx"])

    # One separate figure (with a single axes) per panel.
    figs, axes = [], []
    for panel in PANELS:
        fig = plt.figure(figsize=(20, 10) if panel["kind"] == "2d" else (12, 12))
        ax = fig.add_subplot(projection="3d") if panel["kind"] == "3d" else fig.add_subplot()
        figs.append(fig)
        axes.append(ax)

    for trim_idx in TRIM_RANGE:
        tilt_deg = (trim_idx - 1) * 5
        npy_path = OUTPUT_DIR / f"U4_LON_{MODE.upper()}_TILT{tilt_deg}.npy"
        if not npy_path.exists():
            print(f"skipping tilt {tilt_deg}: missing {npy_path.name}")
            continue

        values = np.load(npy_path)
        # Lon trim state [u, w, q, theta]; u_trim (x_trim[0], m/s) is the corridor height.
        x_trim = np.asarray(trim[trim_idx - 1].squeeze()[x_state_idx])
        u_trim_mps = float(x_trim[0])

        for ax, panel in zip(axes, PANELS):
            a, b = panel["dims"]
            slicer = tuple(slice(None) if dim in (a, b) else n // 2
                           for dim, n in enumerate(grid_shape))
            xa = np.asarray(grid.coordinate_vectors[a])
            xb = np.asarray(grid.coordinate_vectors[b])
            lines = []
            for seg in zero_segments(xa, xb, np.asarray(values[slicer])):
                kept_x = (seg[:, 0] + x_trim[a]) * scale[a]
                kept_y = (seg[:, 1] + x_trim[b]) * scale[b]
                coords = place(kept_x, kept_y, u_trim_mps, panel["xyz"])
                lines.append(np.column_stack(coords))
            if lines:
                collection = (Line3DCollection(lines, colors=[CONTOUR_COLOR]) if panel["kind"] == "3d"
                              else LineCollection(lines, colors=[CONTOUR_COLOR]))
                (ax.add_collection3d if panel["kind"] == "3d" else ax.add_collection)(collection)

            # Rectangular target set for this trim point (shifted to trim, scaled).
            la, ha, lb, hb = tgt_lo[a], tgt_hi[a], tgt_lo[b], tgt_hi[b]
            rect_a = (np.array([la, ha, ha, la, la]) + x_trim[a]) * scale[a]
            rect_b = (np.array([lb, lb, hb, hb, lb]) + x_trim[b]) * scale[b]
            ax.plot(*place(rect_a, rect_b, u_trim_mps, panel["xyz"]),
                    color=TARGET_COLOR, linewidth=1.0)

    for fig, ax, panel in zip(figs, axes, PANELS):
        ax.set_title(f"Transition Corridor {panel['title']} ({MODE.upper()})")
        if panel["kind"] == "3d":
            ax.set_box_aspect((1, 5, 1))
            ax.autoscale()
            ax.view_init(elev=15, azim=-50)
        else:
            ax.autoscale()
            ax.grid(True)
        out_path = OUTPUT_DIR / f"U4_transition_corridor_{panel['file']}_{MODE.upper()}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
        print(f"Saved {out_path}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
