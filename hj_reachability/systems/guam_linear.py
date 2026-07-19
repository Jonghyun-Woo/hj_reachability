"""Linearized GUAM dynamics.

`trim_table_Poly_ConcatVer4p0.mat` is expected to contain the trim points and
linearized system dynamics:

Ap_lat_interp, Bp_lat_interp / Ap_lon_interp, Bp_lon_interp:
A_lat (4 x 4 ) : Ap_lat_interp(:, :, UH, WH)
B_lat (4 x 10) : Bp_lat_interp(:, :, UH, WH)
A_lon (4 x 4 ) : Ap_lon_interp(:, :, UH, WH)
B_lon (4 x 11) : Bp_lon_interp(:, :, UH, WH)

X0 (12 x 1) : XU0_interp(1:12, UH, WH) <= [u, v, w, p, q, r, ax, ay, az, phi, theta, psi]
U0 (13 x 1) : XU0_interp(13:end, UH, WH) <= [flap, aileron, elevator, rudder, Pi_lifts (8), Pi_pusher]

The dynamics are expressed in deviation coordinates about the selected trim
point, i.e. the grid state is `dx = x - x_trim` and the control is
`du = u - u_trim`, so that

    d(dx)/dt = A @ dx + B @ du + d
"""

import jax.numpy as jnp
import numpy as np
import scipy.io

from hj_reachability import dynamics
from hj_reachability import sets


TRIM_KEYS = ("Ap_lat_interp", "Bp_lat_interp", "Ap_lon_interp", "Bp_lon_interp",
             "XU0_interp", "UH", "WH")

# X_TRIM order: [u v w p q r ax ay az phi theta psi]
# U_TRIM order: [flap, aileron, elevator, rudder, Pi_1, ..., Pi_8, Pi_pusher]
AXIS_SPEC = {
    "lon": {
        # States: [u (ft/s), w (ft/s), q (rad/s), theta (rad)]
        # Inputs: [Pi_1 (rad/s), ..., Pi_8 (rad/s), Pi_p (rad/s), delta_e (rad), delta_f (rad)]
        "A_key": "Ap_lon_interp",
        "B_key": "Bp_lon_interp",
        "state_idx": (0, 2, 4, 10),
        "input_idx": (4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0),
        "input_names": ("Pi",) * 8 + ("Pi_p", "delta_e", "delta_f"),
        "cfg_key": "longitudinal",
        "state_names": ("u", "w", "q", "theta"),
    },
    "lat": {
        # States: [v (ft/s), p (rad/s), r (rad/s), phi (rad)]
        # Inputs: [Pi_1 (rad/s), ..., Pi_8 (rad/s), delta_a (rad), delta_r (rad)]
        "A_key": "Ap_lat_interp",
        "B_key": "Bp_lat_interp",
        "state_idx": (1, 3, 5, 9),
        "input_idx": (4, 5, 6, 7, 8, 9, 10, 11, 1, 3),
        "input_names": ("Pi",) * 8 + ("delta_a", "delta_r"),
        "cfg_key": "lateral",
        "state_names": ("v", "p", "r", "phi"),
    },
}


def load_trim_table(mat_path):
    """Loads `trim_table_Poly_ConcatVer4p0.mat` and extracts the trim tables.

    Returns a dict mapping each key in `TRIM_KEYS` to a numpy array; the
    interpolated tables are indexed by the trailing (UH, WH) axes, e.g.
    `Ap_lon_interp` has shape (4, 4, n_UH, n_WH).

    Note: `scipy.io.loadmat` supports MAT files up to v7.2; if the file was
    saved with `-v7.3`, load it with h5py instead.
    """
    mat = scipy.io.loadmat(mat_path)
    missing = [key for key in TRIM_KEYS if key not in mat]
    if missing:
        raise KeyError(f"missing keys in {mat_path}: {missing}")
    return {key: np.asarray(mat[key]) for key in TRIM_KEYS}


class GuamLinear(dynamics.ControlAndDisturbanceAffineDynamics):
    """Linear dynamics `d(dx)/dt = A @ dx + B @ du + d` about one trim point.

    Attributes:
        A: State matrix at the selected trim point (4x4).
        B: Input matrix at the selected trim point (lon: 4x11, lat: 4x10).
        x_trim: Trim state for the selected axis (lon: [u, w, q, theta],
            lat: [v, p, r, phi]).
        u_trim: Trim input for the selected axis in B-column order
            (lon: [Pi_1, ..., Pi_8, Pi_p, delta_e, delta_f],
            lat: [Pi_1, ..., Pi_8, delta_a, delta_r]).
        ctrl_lb, ctrl_ub: Control bounds in deviation coordinates, i.e. the
            allowed perturbation about the trim input clipped to the physical
            actuator limits (same construction as `init_input_bounds` in
            helperOC's `GUAM_LON` / `GUAM_LAT`).
    """

    def __init__(self,
                 cfg,
                 uh_idx=1,
                 wh_idx=3,
                 axis="lon",
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):
        """
        Args:
            cfg: Configuration dict loaded from `config/guam_analysis_config.yml`;
                provides the .mat file path (`cfg["mat_path"]`), the physical
                actuator limits (`cfg["dynamics"]`) and the per-axis
                perturbation limits and disturbance magnitude
                (`cfg["longitudinal"]` / `cfg["lateral"]`).
            uh_idx: Horizontal-speed grid index, 1..n_UH (MATLAB 1-based);
                the trim airspeed is `UH[uh_idx - 1]` (ft/s).
            wh_idx: Vertical-speed grid index, 1..n_WH (MATLAB 1-based);
                the trim vertical speed is `WH[wh_idx - 1]` (ft/s).
            axis: "lon" for (A_lon, B_lon) or "lat" for (A_lat, B_lat).
        """
        if axis not in AXIS_SPEC:
            raise ValueError(f"axis must be 'lon' or 'lat', got {axis!r}")
        spec = AXIS_SPEC[axis]

        data = load_trim_table(cfg["mat_path"])
        uh_grid = data["UH"].ravel()
        wh_grid = data["WH"].ravel()
        if not 1 <= uh_idx <= uh_grid.size:
            raise ValueError(f"uh_idx must be in [1, {uh_grid.size}], got {uh_idx}")
        if not 1 <= wh_idx <= wh_grid.size:
            raise ValueError(f"wh_idx must be in [1, {wh_grid.size}], got {wh_idx}")
        i, j = uh_idx - 1, wh_idx - 1

        self.A = jnp.asarray(data[spec["A_key"]][:, :, i, j], dtype=jnp.float32)
        self.B = jnp.asarray(data[spec["B_key"]][:, :, i, j], dtype=jnp.float32)
        xu0 = data["XU0_interp"][:, i, j]
        self.x_trim = jnp.asarray(xu0[:12][list(spec["state_idx"])], dtype=jnp.float32)
        self.u_trim = jnp.asarray(xu0[12:][list(spec["input_idx"])], dtype=jnp.float32)
        self.uh_idx = uh_idx
        self.wh_idx = wh_idx
        self.uh = float(uh_grid[i])
        self.wh = float(wh_grid[j])
        self.axis = axis

        self.ctrl_lb, self.ctrl_ub = self.control_bound(cfg)
        if control_space is None:
            control_space = sets.Box(self.ctrl_lb, self.ctrl_ub)
        if disturbance_space is None:
            # Additive per-state disturbance |d_i| <= dist_max, matching
            # `GUAM_LON/optDstb.m` in helperOC; dist_max = 0 means no disturbance.
            dist_max = jnp.broadcast_to(jnp.asarray(cfg[spec["cfg_key"]]["dist_max"], dtype=jnp.float32),
                                        (self.A.shape[0],))
            disturbance_space = sets.Box(-dist_max, dist_max)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def control_bound(self, cfg):
        """Control bounds in deviation coordinates about the trim input.

        Following `init_input_bounds` in helperOC's `GUAM_LON` / `GUAM_LAT`:
            lb = max(physical_lb, u_trim - Delta) - u_trim
            ub = min(physical_ub, u_trim + Delta) - u_trim
        where the physical limits come from `cfg["dynamics"]` and the
        perturbation limits (+-Delta) from the per-axis config section. The
        eight lift rotors share the single `Pi` config entry.
        """
        spec = AXIS_SPEC[self.axis]
        phys = cfg["dynamics"]
        axis_cfg = cfg[spec["cfg_key"]]
        names = spec["input_names"]

        phys_lb = jnp.array([phys[f"input_min_{name}"] for name in names], dtype=jnp.float32)
        phys_ub = jnp.array([phys[f"input_max_{name}"] for name in names], dtype=jnp.float32)
        delta_lb = jnp.array([axis_cfg[f"input_min_{name}"] for name in names], dtype=jnp.float32)
        delta_ub = jnp.array([axis_cfg[f"input_max_{name}"] for name in names], dtype=jnp.float32)

        ctrl_lb = jnp.maximum(phys_lb, self.u_trim + delta_lb) - self.u_trim
        ctrl_ub = jnp.minimum(phys_ub, self.u_trim + delta_ub) - self.u_trim
        return ctrl_lb, ctrl_ub

    def open_loop_dynamics(self, state, time):
        return self.A @ state

    def control_jacobian(self, state, time):
        return self.B

    def disturbance_jacobian(self, state, time):
        # The disturbance enters each state equation directly (dx/dt += d),
        # matching `GUAM_LON/dynamics.m` in helperOC.
        return jnp.eye(self.A.shape[0])
