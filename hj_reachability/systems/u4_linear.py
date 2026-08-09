"""Linearized U4 dynamics built from MATLAB trim corridor results.

`Trim_Corridor_Results.mat` is expected to contain the following 1x19 cell
arrays (one entry per trim point, MATLAB indices 1..19):

    A_LAT_all, A_LON_all : lateral / longitudinal state matrices
    B_LAT_all, B_LON_all : lateral / longitudinal input matrices
    X_TRIM, U_TRIM       : trim state / trim input at each point

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

TRIM_KEYS = ("A_LAT_all", "A_LON_all", "B_LAT_all", "B_LON_all", "U_TRIM", "X_TRIM")

# X_TRIM order: [u v w p q r phi theta psi x y z]
# U_TRIM order: [Pi (%), delta_a (deg), delta_e (deg), delta_r (deg), tilt (deg), ...]
AXIS_SPEC = {
    "lon": {
        # States: [u (m/s), w (m/s), q (rad/s), theta (rad)]
        # Inputs: [Pi (%), delta_e (deg), tilt (deg)]
        "A_key": "A_LON_all",
        "B_key": "B_LON_all",
        "state_idx": (0, 2, 4, 7),
        "input_idx": (0, 2, 4),
        "input_names": ("Pi", "delta_e", "tilt"),
        "cfg_key": "longitudinal",
        "state_names": ("u", "w", "q", "theta"),
    },
    "lat": {
        # States: [v (m/s), p (rad/s), r (rad/s), phi (rad)]
        # Inputs: [delta_a (deg), delta_r (deg), tilt (deg)]
        "A_key": "A_LAT_all",
        "B_key": "B_LAT_all",
        "state_idx": (1, 3, 5, 6),
        "input_idx": (1, 3, 4),
        "input_names": ("delta_a", "delta_r", "tilt"),
        "cfg_key": "lateral",
        "state_names": ("v", "p", "r", "phi"),
    },
}


def load_trim_corridor(mat_path):
    """Loads `Trim_Corridor_Results.mat` and unpacks each 1x19 cell array.

    Returns a dict mapping each key in `TRIM_KEYS` to a list of 19 numpy
    arrays (list index 0 corresponds to MATLAB index 1).

    Note: `scipy.io.loadmat` supports MAT files up to v7.2; if the file was
    saved with `-v7.3`, load it with h5py instead.
    """
    mat = scipy.io.loadmat(mat_path)
    missing = [key for key in TRIM_KEYS if key not in mat]
    if missing:
        raise KeyError(f"missing keys in {mat_path}: {missing}")
    return {key: [np.asarray(cell) for cell in mat[key].ravel()] for key in TRIM_KEYS}


class U4Linear(dynamics.ControlAndDisturbanceAffineDynamics):
    """Linear dynamics `d(dx)/dt = A @ dx + B @ du + d` about one trim point.

    Attributes:
        A: State matrix at the selected trim point (4x4).
        B: Input matrix at the selected trim point (4x3).
        x_trim: Trim state for the selected axis (lon: [u, w, q, theta],
            lat: [v, p, r, phi]).
        u_trim: Trim input for the selected axis (lon: [Pi, delta_e, tilt],
            lat: [delta_a, delta_r, tilt]).
        ctrl_lb, ctrl_ub: Control bounds in deviation coordinates, i.e. the
            allowed perturbation about the trim input clipped to the physical
            actuator limits (same construction as `U4_Config.init_input_bounds`
            in helperOC).
    """

    def __init__(self,
                 cfg,
                 trim_idx=1,
                 axis="lon",
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):
        """
        Args:
            cfg: Configuration dict loaded from `config/u4_analysis_config.yml`;
                provides the .mat file path (`cfg["mat_path"]`), the physical
                actuator limits (`cfg["dynamics"]`) and the per-axis
                perturbation limits and disturbance magnitude
                (`cfg["longitudinal"]` / `cfg["lateral"]`).
            trim_idx: Trim point index, 1..19 (MATLAB 1-based); the tilt angle
                is `(trim_idx - 1) * 5` degrees.
            axis: "lon" for (A_LON, B_LON) or "lat" for (A_LAT, B_LAT).
        """
        if axis not in AXIS_SPEC:
            raise ValueError(f"axis must be 'lon' or 'lat', got {axis!r}")
        spec = AXIS_SPEC[axis]

        data = load_trim_corridor(cfg["mat_path"])
        if not 1 <= trim_idx <= len(data["X_TRIM"]):
            raise ValueError(f"trim_idx must be in [1, {len(data['X_TRIM'])}], got {trim_idx}")
        i = trim_idx - 1

        self.A          = jnp.asarray(data[spec["A_key"]][i], dtype=jnp.float32)
        self.B          = jnp.asarray(data[spec["B_key"]][i], dtype=jnp.float32)
        self.x_trim     = jnp.asarray(data["X_TRIM"][i].squeeze()[list(spec["state_idx"])], dtype=jnp.float32)
        self.u_trim     = jnp.asarray(data["U_TRIM"][i].squeeze()[list(spec["input_idx"])], dtype=jnp.float32)
        self.trim_idx   = trim_idx
        self.tilt_deg   = (trim_idx - 1) * 5
        self.axis = axis

        self.ctrl_lb, self.ctrl_ub = self.control_bound(cfg)
        if control_space is None:
            control_space = sets.Box(self.ctrl_lb, self.ctrl_ub)
        if disturbance_space is None:
            # Additive per-state disturbance |d_i| <= dist_max, matching
            # `optDstb` in helperOC; dist_max = 0 means no disturbance.
            dist_max = jnp.broadcast_to(jnp.asarray(cfg[spec["cfg_key"]]["dist_max"], dtype=jnp.float32),
                                        (self.A.shape[0],))
            disturbance_space = sets.Box(-dist_max, dist_max)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def control_bound(self, cfg):
        """Control bounds in deviation coordinates about the trim input.

        Following `U4_Config.init_input_bounds` in helperOC:
            lb = max(physical_lb, u_trim - Delta) - u_trim
            ub = min(physical_ub, u_trim + Delta) - u_trim
        where the physical limits come from `cfg["dynamics"]` and the
        perturbation limits (+-Delta) from the per-axis config section.
        """
        spec        = AXIS_SPEC[self.axis]
        phys        = cfg["dynamics"]
        axis_cfg    = cfg[spec["cfg_key"]]
        names       = spec["input_names"]

        phys_lb     = jnp.array([phys[f"input_min_{name}"] for name in names], dtype=jnp.float32)
        phys_ub     = jnp.array([phys[f"input_max_{name}"] for name in names], dtype=jnp.float32)
        delta_lb    = jnp.array([axis_cfg[f"input_min_{name}"] for name in names], dtype=jnp.float32)
        delta_ub    = jnp.array([axis_cfg[f"input_max_{name}"] for name in names], dtype=jnp.float32)

        ctrl_lb = jnp.maximum(phys_lb, self.u_trim + delta_lb) - self.u_trim
        ctrl_ub = jnp.minimum(phys_ub, self.u_trim + delta_ub) - self.u_trim
        return jnp.deg2rad(ctrl_lb), jnp.deg2rad(ctrl_ub)  # convert to radians for the angle inputs

    def open_loop_dynamics(self, state, time):
        return self.A @ state

    def control_jacobian(self, state, time):
        return self.B

    def disturbance_jacobian(self, state, time):
        # The disturbance enters each state equation directly (dx/dt += d),
        # matching `dynamics.m` in helperOC.
        return jnp.eye(self.A.shape[0])
