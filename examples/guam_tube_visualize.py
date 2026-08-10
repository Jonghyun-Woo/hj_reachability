import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import itertools
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

import hj_reachability as hj
from hj_reachability.systems.guam_linear import AXIS_SPEC

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "guam_outputs"

with open(REPO_ROOT / "config" / "guam_analysis_config.yml") as config_file:
    cfg = yaml.safe_load(config_file)

# Unit conversions and per-axis display scaling / labels (from Config.m).
FT2M = 0.3048
RAD2DEG = 180.0 / np.pi
AXIS_DISPLAY = {
    "lon": {"scale": np.array([FT2M, FT2M, RAD2DEG, RAD2DEG]),
            "labels": ["u (m/s)", "w (m/s)", "q (deg/s)", "theta (deg)"]},
    "lat": {"scale": np.array([FT2M, RAD2DEG, RAD2DEG, RAD2DEG]),
            "labels": ["v (m/s)", "p (deg/s)", "r (deg/s)", "phi (deg)"]},
}
# brt -> red 'BRT', frt -> blue 'FRT' (contour_color / contour_legend in MATLAB).
TUBE_STYLE = {"BRT": ("red", "BRT"), "FRT": ("blue", "FRT")}


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


def target_corners(axis):
    """Target rectangle corners for `axis`, scaled to display units."""
    spec = AXIS_SPEC[axis]
    axis_cfg = cfg[spec["cfg_key"]]
    scale = AXIS_DISPLAY[axis]["scale"]
    lo = np.array([axis_cfg[f"target_min_{name}"] for name in spec["state_names"]]) * scale
    hi = np.array([axis_cfg[f"target_max_{name}"] for name in spec["state_names"]]) * scale
    return lo, hi


def parse_axis(stem):
    """Extracts the axis ('lon'/'lat') from a stem like 'GUAM_LON_BRT_UH20_WH3'."""
    for axis in AXIS_SPEC:
        if f"_{axis.upper()}_" in stem:
            return axis
    raise ValueError(f"cannot determine axis from filename stem {stem!r}")


def parse_tube(stem):
    """Returns (color, legend) for the tube type encoded in the stem."""
    for tag, style in TUBE_STYLE.items():
        if f"_{tag}_" in stem:
            return style
    raise ValueError(f"cannot determine tube type (BRT/FRT) from stem {stem!r}")


def visualize_2d(npy_path):
    """Draws all 6 state-pair tube contours of one value function into one figure."""
    stem = npy_path.stem
    axis = parse_axis(stem)
    color, legend = parse_tube(stem)
    grid, grid_shape, names = build_grid(axis)
    scale = AXIS_DISPLAY[axis]["scale"]
    labels = AXIS_DISPLAY[axis]["labels"]
    tgt_lo, tgt_hi = target_corners(axis)
    values = np.load(npy_path)

    pairs = list(itertools.combinations(range(len(names)), 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for (x_dim, y_dim), ax in zip(pairs, axes.ravel()):
        # Hold the other two states at their center index (trim deviation ~ 0).
        slicer = tuple(slice(None) if dim in (x_dim, y_dim) else n // 2
                       for dim, n in enumerate(grid_shape))
        slice_2d = np.asarray(values[slicer]).T
        xs = np.asarray(grid.coordinate_vectors[x_dim]) * scale[x_dim]
        ys = np.asarray(grid.coordinate_vectors[y_dim]) * scale[y_dim]
        # Shade the tube interior (value <= 0) and draw its zero-level boundary.
        if slice_2d.min() < 0 < slice_2d.max():
            ax.contourf(xs, ys, slice_2d, levels=[slice_2d.min(), 0.0],
                        colors=[color], alpha=0.25)
            ax.contour(xs, ys, slice_2d, levels=[0.0], colors=color, linewidths=2)
        ax.add_patch(Rectangle((tgt_lo[x_dim], tgt_lo[y_dim]),
                               tgt_hi[x_dim] - tgt_lo[x_dim], tgt_hi[y_dim] - tgt_lo[y_dim],
                               edgecolor="green", facecolor="none", linewidth=1.5, label="Target"))
        ax.set_xlabel(labels[x_dim])
        ax.set_ylabel(labels[y_dim])
        ax.set_title(f"{names[x_dim]} - {names[y_dim]}")
        ax.grid(True)

    fig.suptitle(f"{stem}  ({legend} tube vs Target)", fontsize=14)
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{stem}_2d.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _box_edges(lo, hi):
    """Line coordinates (x, y, z) tracing the 12 edges of an axis-aligned box."""
    c = np.array(list(itertools.product(*zip(lo, hi))))  # 8 corners
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if bin(i ^ j).count("1") == 1]  # corners differing in one axis
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [c[i, 0], c[j, 0], None]
        ys += [c[i, 1], c[j, 1], None]
        zs += [c[i, 2], c[j, 2], None]
    return xs, ys, zs


def visualize_3d(npy_path):
    """Draws all 4 state-triple tube isosurfaces of one value function into one HTML."""
    stem = npy_path.stem
    axis = parse_axis(stem)
    color, legend = parse_tube(stem)
    grid, grid_shape, names = build_grid(axis)
    scale = AXIS_DISPLAY[axis]["scale"]
    labels = AXIS_DISPLAY[axis]["labels"]
    tgt_lo, tgt_hi = target_corners(axis)
    values = np.load(npy_path)

    triples = list(itertools.combinations(range(len(names)), 3))
    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scene"}] * 2] * 2,
                        subplot_titles=[", ".join(names[d] for d in t) for t in triples])
    for idx, (a, b, c) in enumerate(triples):
        row, col = idx // 2 + 1, idx % 2 + 1
        # Hold the remaining state at its center index (trim deviation ~ 0).
        slicer = tuple(slice(None) if dim in (a, b, c) else n // 2
                       for dim, n in enumerate(grid_shape))
        slice_3d = np.asarray(values[slicer])
        xa = np.asarray(grid.coordinate_vectors[a]) * scale[a]
        xb = np.asarray(grid.coordinate_vectors[b]) * scale[b]
        xc = np.asarray(grid.coordinate_vectors[c]) * scale[c]
        X, Y, Z = np.meshgrid(xa, xb, xc, indexing="ij")
        if slice_3d.min() < 0 < slice_3d.max():
            fig.add_trace(go.Isosurface(
                x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=slice_3d.ravel(),
                isomin=0.0, isomax=0.0, surface_count=1, opacity=0.5,
                colorscale=[[0, color], [1, color]], showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False), name=legend),
                row=row, col=col)
        bx, by, bz = _box_edges([tgt_lo[a], tgt_lo[b], tgt_lo[c]],
                                [tgt_hi[a], tgt_hi[b], tgt_hi[c]])
        fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="lines",
                                   line=dict(color="green", width=3), name="Target"),
                      row=row, col=col)
        scene_name = "scene" if idx == 0 else f"scene{idx + 1}"
        fig.update_layout({scene_name: dict(xaxis_title=labels[a], yaxis_title=labels[b],
                                            zaxis_title=labels[c])})

    fig.update_layout(title=f"{stem}  ({legend} tube vs Target)", showlegend=False)
    out_path = OUTPUT_DIR / f"{stem}_3d.html"
    fig.write_html(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proj", choices=["2d", "3d"], default="2d",
                        help="projection mode: '2d' (state pairs) or '3d' (state triples)")
    args = parser.parse_args()

    npy_files = sorted(OUTPUT_DIR.glob("*.npy"))
    if not npy_files:
        raise SystemExit(f"no .npy files found in {OUTPUT_DIR}")
    render = visualize_2d if args.proj == "2d" else visualize_3d
    for npy_path in npy_files:
        render(npy_path)
