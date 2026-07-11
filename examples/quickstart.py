import os

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # For WSL Environment (No-display)
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

import hj_reachability as hj

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_animation(animation, name):
    try:
        animation.save(os.path.join(OUTPUT_DIR, name + ".mp4"), writer="ffmpeg", fps=20)
        print(f"saved {name}.mp4")
    except (RuntimeError, FileNotFoundError):
        animation.save(os.path.join(OUTPUT_DIR, name + ".gif"), writer="pillow", fps=20)
        print(f"ffmpeg not found; saved {name}.gif instead")

dynamics = hj.systems.Air3d()
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([-6., -10., 0.]),
                                                                           np.array([20., 10., 2 * np.pi])),
                                                               (51, 40, 50),
                                                               periodic_dims=2)
values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 5

# def fixed_step_time_integrator(base_integrator, dt):
#     """한 번에 최대 dt까지만 진행하도록 base_integrator를 감싼다.

#     euler_step은 min(CFL 한계, 요청 구간)으로 스텝을 잡으므로, dt가 CFL 한계보다
#     작으면 매 스텝이 정확히 dt인 고정 스텝 적분이 된다.
#     """

#     def integrator(solver_settings, dynamics, grid, time, values, target_time):
#         step_target = time + jnp.sign(target_time - time) * jnp.minimum(dt, jnp.abs(target_time - time))
#         return base_integrator(solver_settings, dynamics, grid, time, values, step_target)

#     return integrator


solver_settings = hj.SolverSettings.with_accuracy("very_high",
                                                  hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
# solver_settings = solver_settings.replace(
#     time_integrator=fixed_step_time_integrator(solver_settings.time_integrator, 0.001))

time = 0.
target_time = -2.8
target_values = hj.step(solver_settings, dynamics, grid, time, values, target_time)

plt.jet()
plt.figure(figsize=(13, 8))
plt.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values[:, :, 30].T)
plt.colorbar()
plt.contour(grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            target_values[:, :, 30].T,
            levels=0,
            colors="black",
            linewidths=3)
plt.savefig(os.path.join(OUTPUT_DIR, "air3d_slice.png"), bbox_inches="tight")
plt.close()
print("saved air3d_slice.png")

go.Figure(data=go.Isosurface(x=grid.states[..., 0].ravel(),
                             y=grid.states[..., 1].ravel(),
                             z=grid.states[..., 2].ravel(),
                             value=target_values.ravel(),
                             colorscale="jet",
                             isomin=0,
                             surface_count=1,
                             isomax=0)).write_html(os.path.join(OUTPUT_DIR, "air3d_isosurface.html"))
print("saved air3d_isosurface.html")


times = np.linspace(0, -2.8, 57)
initial_values = values
all_values = hj.solve(solver_settings, dynamics, grid, times, initial_values)

vmin, vmax = all_values.min(), all_values.max()
levels = np.linspace(round(vmin), round(vmax), round(vmax) - round(vmin) + 1)
fig = plt.figure(figsize=(13, 8))


def render_frame(i, colorbar=False):
    plt.contourf(grid.coordinate_vectors[0],
                 grid.coordinate_vectors[1],
                 all_values[i, :, :, 30].T,
                 vmin=vmin,
                 vmax=vmax,
                 levels=levels)
    if colorbar:
        plt.colorbar()
    plt.contour(grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                target_values[:, :, 30].T,
                levels=0,
                colors="black",
                linewidths=3)


render_frame(0, True)
save_animation(anim.FuncAnimation(fig, render_frame, all_values.shape[0], interval=50), "air3d_solve")
plt.close(fig)

class AccelerationCurvatureCar(hj.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 max_acceleration=1.,
                 max_curvature=1.,
                 max_position_disturbance=0.25,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-max_acceleration, -max_curvature]),
                                        jnp.array([max_acceleration, max_curvature]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(2), max_position_disturbance)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, _, v, q = state
        return jnp.array([v * jnp.cos(q), v * jnp.sin(q), 0., 0.])

    def control_jacobian(self, state, time):
        v = state[2]
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [1., 0.],
            [0., v],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [1., 0.],
            [0., 1.],
            [0., 0.],
            [0., 0.],
        ])


dynamics = AccelerationCurvatureCar()
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(lo=np.array([-5., -5., -1., -np.pi]),
                                                                           hi=np.array([5., 5., 1., np.pi])),
                                                               (40, 40, 50, 50),
                                                               periodic_dims=3)
values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 1

solver_settings = hj.SolverSettings.with_accuracy("low")

time = 0.
target_time = -2.0
target_values = hj.step(solver_settings, dynamics, grid, time, values, target_time)

go.Figure(data=go.Isosurface(x=grid.states[:, :, -1, :, 0].ravel(),
                             y=grid.states[:, :, -1, :, 1].ravel(),
                             z=grid.states[:, :, -1, :, 3].ravel(),
                             value=target_values[:, :, -1, :].ravel(),
                             colorscale="jet",
                             isomin=0,
                             surface_count=1,
                             isomax=0)).write_html(os.path.join(OUTPUT_DIR, "car4d_isosurface.html"))
print("saved car4d_isosurface.html")

print("done; outputs in", OUTPUT_DIR)
