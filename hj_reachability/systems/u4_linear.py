"""Linearized U4 dynamics built from MATLAB trim corridor results.

`Trim_Corridor_Results.mat` is expected to contain the following 1x19 cell
arrays (one entry per trim point, MATLAB indices 1..19):

    A_LAT_all, A_LON_all : lateral / longitudinal state matrices
    B_LAT_all, B_LON_all : lateral / longitudinal input matrices
    X_TRIM, U_TRIM       : trim state / trim input at each point

The dynamics are expressed in deviation coordinates about the selected trim
point, i.e. the grid state is `dx = x - x_trim` and the control is
`du = u - u_trim`, so that

    d(dx)/dt = A @ dx + B @ du + G_d @ d
"""

import jax.numpy as jnp
import numpy as np
import scipy.io

from hj_reachability import dynamics
from hj_reachability import sets

TRIM_KEYS = ("A_LAT_all", "A_LON_all", "B_LAT_all", "B_LON_all", "U_TRIM", "X_TRIM")


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
    """Linear dynamics `d(dx)/dt = A @ dx + B @ du` about one trim point.

    Attributes:
        A: State matrix at the selected trim point.
        B: Input matrix at the selected trim point.
        x_trim: Trim state (full state, as stored in X_TRIM).
        u_trim: Trim input (full input, as stored in U_TRIM).
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
            cfg: Configuration object/dict; expected to provide the .mat file
                path (`cfg["mat_path"]`) and actuator limits (see
                `control_bound`). TODO: adjust to the actual cfg structure.
            trim_idx: Trim point index, 1..19 (MATLAB 1-based).
            axis: "lon" for (A_LON, B_LON) or "lat" for (A_LAT, B_LAT).
        """
        data = load_trim_corridor(cfg["mat_path"])
        if not 1 <= trim_idx <= len(data["X_TRIM"]):
            raise ValueError(f"trim_idx must be in [1, {len(data['X_TRIM'])}], got {trim_idx}")
        i = trim_idx - 1

        if axis == "lon":
            A, B = data["A_LON_all"][i], data["B_LON_all"][i]
        elif axis == "lat":
            A, B = data["A_LAT_all"][i], data["B_LAT_all"][i]
        else:
            raise ValueError(f"axis must be 'lon' or 'lat', got {axis!r}")

        self.A = jnp.asarray(A, dtype=jnp.float32)
        self.B = jnp.asarray(B, dtype=jnp.float32)
        self.x_trim = jnp.asarray(data["X_TRIM"][i], dtype=jnp.float32).squeeze()
        self.u_trim = jnp.asarray(data["U_TRIM"][i], dtype=jnp.float32).squeeze()
        self.trim_idx = trim_idx
        self.axis = axis

        n_states, n_controls = self.B.shape

        self.ctrl_lb, self.ctrl_ub = self.control_bound(cfg)
        if control_space is None:
            control_space = sets.Box(self.ctrl_lb, self.ctrl_ub)
        if disturbance_space is None:
            # TODO: model disturbance (e.g. wind) if needed; a zero-radius ball
            # means no disturbance.
            disturbance_space = sets.Ball(jnp.zeros(n_states), 0.)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def control_bound(self, cfg):
        n_controls = self.B.shape[1]
        dyn_lb = jnp.array([cfg['dynamics']['input_min_Pi'],
                            cfg['dynamics']['input_min_delta_e'],
                            cfg['dynamics']['input_min_tilt']])
        
        dyn_ub = jnp.array([cfg['dynamics']['input_max_Pi'],
                            cfg['dynamics']['input_max_delta_e'],
                            cfg['dynamics']['input_max_tilt']])
        if self.axis == "lon":
            
            temp_lb = jnp.array([cfg['longitudinal']['input_min_Pi'],
                                 cfg['longitudinal']['input_min_delta_e'],
                                 cfg['longitudinal']['input_min_tilt']])
            
            temp_ub = jnp.array([cfg['longitudinal']['input_max_Pi'],
                                 cfg['longitudinal']['input_max_delta_e'],
                                 cfg['longitudinal']['input_max_tilt']])
            return 
            
        elif self.axis == "lat":
            temp_lb = jnp.array([cfg['lateral']['input_min_delta_a'],
                                 cfg['lateral']['input_min_delta_r'],
                                 cfg['lateral']['input_min_delta_tilt']])
            temp_ub = jnp.array([cfg['lateral']['input_max_delta_a'],
                                 cfg['lateral']['input_max_delta_r'],
                                 cfg['lateral']['input_max_delta_tilt']])
            return
        
        else:
            ReferenceError(f"The \'axis\' should be selected as \'lon\' or \'lat\' ")

    def open_loop_dynamics(self, state, time):
        return self.A @ state

    def control_jacobian(self, state, time):
        return self.B

    def disturbance_jacobian(self, state, time):
        # TODO: replace with the actual disturbance input matrix; identity means
        # the disturbance enters each state equation directly.
        return jnp.eye(self.A.shape[0])
