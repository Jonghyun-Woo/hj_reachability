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
from hj_reachability.systems.guam_linear import AXIS_SPEC, load_trim_table

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "guam_outputs"

with open(REPO_ROOT / "config" / "guam_analysis_config.yml") as config_file:
    cfg = yaml.safe_load(config_file)

FT2M = 0.3048
RAD2DEG = 180.0 / np.pi
AXIS_SCALE = {"lon": np.array([FT2M, FT2M, RAD2DEG, RAD2DEG]),
              "lat": np.array([FT2M, RAD2DEG, RAD2DEG, RAD2DEG])}

CONTOUR_COLOR = "b"      # single color for every tube contour
TARGET_COLOR = "green"   # rectangular target set

# XU0_interp rows [u v w p q r ax ay az phi theta psi]; pick the axis states.
TRIM_ROWS = {"lon": (0, 2, 4, 10), "lat": (1, 3, 5, 9)}

UH_RANGE = range(1, 21)
WH_IDX = 3

# Each panel is its own figure. 'dims' are the two kept state dims; 'xyz' maps
# (kept_x, kept_y, trim-airspeed height 'h') onto the plot axes. The u-w panel
# is a plain 2D plot; the others are 3D with the u_trim ('h') axis in the
# middle, stretched to a 1:5:1 box aspect (like MATLAB's pbaspect([1,5,1])).
PANELS = [
    {"title": "Lon: u vs w", "kind": "2d", "axis": "lon", "dims": (0, 1), "file": "u_w",
     "xyz": ("x", "y"), "labels": ("u (m/s)", "w (m/s)")},
    {"title": "Lon: q vs u vs theta", "kind": "3d", "axis": "lon", "dims": (2, 3), "file": "q_u_theta",
     "xyz": ("x", "h", "y"), "labels": ("q (deg/s)", "u_trim (m/s)", "theta (deg)")},
    {"title": "Lat: v vs u vs p", "kind": "3d", "axis": "lat", "dims": (0, 1), "file": "v_u_p",
     "xyz": ("x", "h", "y"), "labels": ("v (m/s)", "u_trim (m/s)", "p (deg/s)")},
    {"title": "Lat: r vs u vs phi", "kind": "3d", "axis": "lat", "dims": (2, 3), "file": "r_u_phi",
     "xyz": ("x", "h", "y"), "labels": ("r (deg/s)", "u_trim (m/s)", "phi (deg)")},
]

FIGURE_CFG = [{"font_size": 14}]

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
    trim = load_trim_table(str(REPO_ROOT / cfg["mat_path"]))["XU0_interp"]
    grids = {axis: build_grid(axis) for axis in ("lon", "lat")}
    targets = {axis: target_corners(axis) for axis in ("lon", "lat")}

    # One separate figure (with a single axes) per panel.
    figs, axes = [], []
    for panel in PANELS:
        if panel["kind"] == "2d":
            fig = plt.figure(figsize=(20, 10))
        else:
            fig = plt.figure(figsize=(12, 12))

        ax = fig.add_subplot(projection="3d") if panel["kind"] == "3d" else fig.add_subplot()
        figs.append(fig)
        axes.append(ax)

    for uh_idx in UH_RANGE:
        col = uh_idx - 1
        u_trim_mps = float(trim[0, col, WH_IDX - 1]) * FT2M

        data = {}  # axis -> (values, grid, grid_shape, x_trim, scale)
        for axis in ("lon", "lat"):
            npy_path = OUTPUT_DIR / f"GUAM_{axis.upper()}_BRT_UH{uh_idx}_WH{WH_IDX}.npy"
            if not npy_path.exists():
                break
            grid, grid_shape, _ = grids[axis]
            x_trim = np.asarray(trim[TRIM_ROWS[axis], col, WH_IDX - 1])
            data[axis] = (np.load(npy_path), grid, grid_shape, x_trim, AXIS_SCALE[axis])
        if len(data) < 2:
            print(f"skipping UH{uh_idx}: missing lon/lat npy")
            continue

        for ax, panel in zip(axes, PANELS):
            values, grid, grid_shape, x_trim, scale = data[panel["axis"]]
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
            tgt_lo, tgt_hi = targets[panel["axis"]]
            la, ha, lb, hb = tgt_lo[a], tgt_hi[a], tgt_lo[b], tgt_hi[b]
            rect_a = (np.array([la, ha, ha, la, la]) + x_trim[a]) * scale[a]
            rect_b = (np.array([lb, lb, hb, hb, lb]) + x_trim[b]) * scale[b]
            ax.plot(*place(rect_a, rect_b, u_trim_mps, panel["xyz"]),
                    color=TARGET_COLOR, linewidth=1.0)

    # sm = plt.cm.ScalarMappable(cmap=cmap,
    #                            norm=plt.Normalize(UH_RANGE.start, UH_RANGE.stop - 1))
    for fig, ax, panel in zip(figs, axes, PANELS):
        ax.set_title(f"Transition Corridor {panel['title']} (BRT)")
        # ax.set_xlabel(panel["labels"][0])
        # ax.set_ylabel(panel["labels"][1])
        if panel["kind"] == "3d":
            # ax.set_zlabel(panel["labels"][2])
            ax.set_box_aspect((1, 5, 1))
            ax.autoscale()
            ax.view_init(elev=15, azim=-50)
        else:
            ax.autoscale()
            ax.grid(True)
        # fig.colorbar(sm, ax=ax, shrink=0.6, label="uh_idx")
        out_path = OUTPUT_DIR / f"GUAM_transition_corridor_{panel['file']}_BRT.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
        print(f"Saved {out_path}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
