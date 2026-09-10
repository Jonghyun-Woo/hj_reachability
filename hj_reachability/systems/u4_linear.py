"""Linearized U4 dynamics built from MATLAB trim corridor results.

`Trim_Corridor_Results.mat` is expected to contain the following 1x19 cell
arrays (one entry per trim point, MATLAB indices 1..19):

    A_LAT_all, A_LON_all : lateral / longitudinal state matrices
    B_LAT_all, B_LON_all : lateral / longitudinal input matrices
    X_TRIM               : trim state at each point
    U_TRIM_IND           : trim input at each point [Thrust (FR, RR, RL, FL), ail, rvL, rvR, Tilt (FR, RR, RL, FL)]

The dynamics are expressed in deviation coordinates about the selected trim
point, i.e. the grid state is `dx = x - x_trim` and the control is
`du = u - u_trim`, so that

    d(dx)/dt = A @ dx + B @ du + d
"""

import os

import jax.numpy as jnp
import numpy as np
import scipy.io

from hj_reachability import dynamics
from hj_reachability import sets

TRIM_KEYS = ("A_LAT_all", "A_LON_all", "B_LAT_all", "B_LON_all", "U_TRIM_IND", "X_TRIM")

# X_TRIM order: [u v w p q r phi theta psi x y z]
AXIS_SPEC = {
    "lon": {
        # States: [u (m/s), w (m/s), q (rad/s), theta (rad)]
        "A_key": "A_LON_all",
        "B_key": "B_LON_all",
        "state_idx": (0, 2, 4, 7),
        "cfg_key": "longitudinal",
        "state_names": ("u", "w", "q", "theta"),
    },
    "lat": {
        # States: [v (m/s), p (rad/s), r (rad/s), phi (rad)]
        "A_key": "A_LAT_all",
        "B_key": "B_LAT_all",
        "state_idx": (1, 3, 5, 6),
        "cfg_key": "lateral",
        "state_names": ("v", "p", "r", "phi"),
    },
}

PHYS_INPUT_NAMES = ("thr_FR", "thr_RR", "thr_RL", "thr_FL", "d_ail", "d_rvL", "d_rvR", "tilt_FR", "tilt_RR", "tilt_RL", "tilt_FL")
INPUT_TYPES = ("thr", "thr", "thr", "thr", "ail", "rv", "rv", "tilt", "tilt", "tilt", "tilt")
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
        B: Input matrix at the selected trim point (4x11)
        x_trim: Trim state for the selected axis (lon: [u, w, q, theta],
            lat: [v, p, r, phi]).
        u_trim: Trim input (11 physical inputs, `PHYS_INPUT_NAMES` order).
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
        self.u_trim     = jnp.asarray(data["U_TRIM_IND"][i].squeeze(), dtype=jnp.float32)
        self.trim_idx   = trim_idx
        self.tilt_deg   = (trim_idx - 1) * 5
        self.axis = axis

        self.ctrl_lb, self.ctrl_ub = self.control_bound(cfg)
        if control_space is None:
            control_space = sets.Box(self.ctrl_lb, self.ctrl_ub)

        self._beta = None
        if disturbance_space is None:
            # Base additive per-state disturbance d_box in [min, max], keyed by
            # state name under `dist` in the per-axis config section (a
            # min == max == 0 state means no base disturbance).
            dist_cfg = cfg[spec["cfg_key"]]["dist"]
            dist_lb = jnp.array([dist_cfg[name]["min"] for name in spec["state_names"]], dtype=jnp.float32)
            dist_ub = jnp.array([dist_cfg[name]["max"] for name in spec["state_names"]], dtype=jnp.float32)

            # If `quadfit_mat` is present and the file exists, a state-dependent
            # model-mismatch envelope (quadratic fit) acts *in addition* to the
            # base disturbance: stacked as d = [d_box (4); d_mismatch (4)],
            # d_mismatch in [-1, 1]^4, shaped by diag(e_max(state)) in
            # disturbance_jacobian. beta_{axis}: (15, 4, n_trim); slice for this
            # trim_idx (0-based). Otherwise only the base box disturbance is used.
            quadfit_mat = cfg.get("quadfit_mat")
            if quadfit_mat and os.path.exists(quadfit_mat):
                qf = scipy.io.loadmat(quadfit_mat)
                self._beta = jnp.asarray(qf[f"beta_{axis}"][:, :, i], dtype=jnp.float32)
                self._half = jnp.asarray(qf[f"half_{axis}"].ravel(), dtype=jnp.float32)
                ones = jnp.ones(4, dtype=jnp.float32)
                disturbance_space = sets.Box(jnp.concatenate([dist_lb, -ones]),
                                             jnp.concatenate([dist_ub, ones]))
            else:
                disturbance_space = sets.Box(dist_lb, dist_ub)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def control_bound(self, cfg):
        """Control bounds in deviation coordinates about the trim input.

        Following `U4_Config.init_input_bounds` in helperOC, per physical input:
            lb = max(physical_lb, u_trim - Delta) - u_trim
            ub = min(physical_ub, u_trim + Delta) - u_trim
        where the physical limits come from `cfg["dynamics"]` and the
        perturbation limits (+-Delta) from the per-axis config section.
        """
        axis_cfg    = cfg[AXIS_SPEC[self.axis]["cfg_key"]]
        phys        = cfg["dynamics"]

        phys_lb     = jnp.array([phys[f"input_min_{t}"] for t in INPUT_TYPES], dtype=jnp.float32)
        phys_ub     = jnp.array([phys[f"input_max_{t}"] for t in INPUT_TYPES], dtype=jnp.float32)
        delta_lb    = jnp.array([axis_cfg[f"input_min_{t}"] for t in INPUT_TYPES], dtype=jnp.float32)
        delta_ub    = jnp.array([axis_cfg[f"input_max_{t}"] for t in INPUT_TYPES], dtype=jnp.float32)

        ctrl_lb = jnp.maximum(phys_lb, self.u_trim + delta_lb) - self.u_trim
        ctrl_ub = jnp.minimum(phys_ub, self.u_trim + delta_ub) - self.u_trim
        return ctrl_lb, ctrl_ub

    def open_loop_dynamics(self, state, time):
        return self.A @ state

    def control_jacobian(self, state, time):
        return self.B

    def disturbance_jacobian(self, state, time):
        # Base additive disturbance enters each state directly (dx/dt += d_box),
        # matching `dynamics.m` / optDstb in helperOC.
        g_box = jnp.eye(self.A.shape[0])
        if self._beta is None:
            return g_box
        # Model-mismatch envelope stacked alongside the base disturbance so both
        # act simultaneously: G_d = [I | diag(e_max(state))], d = [d_box; d_mismatch].
        z = state / self._half
        phi = jnp.array([1., z[0], z[1], z[2], z[3],
                         z[0]**2, z[1]**2, z[2]**2, z[3]**2,
                         z[0]*z[1], z[0]*z[2], z[0]*z[3],
                         z[1]*z[2], z[1]*z[3], z[2]*z[3]])
        e_max = jnp.maximum(phi @ self._beta, 0.)
        return jnp.concatenate([g_box, jnp.diag(e_max)], axis=1)
