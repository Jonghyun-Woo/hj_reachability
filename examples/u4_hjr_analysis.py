import os
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

import hj_reachability as hj

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "u4_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

config_path = Path(__file__).parent / "config" / "u4_analysis_config.yml"
with open(config_path) as config_file:
    cfg = yaml.safe_load(config_file)

# 사각형 두 코너 정의
# target_lo = jnp.array([])
# target_hi = jnp.array([])
# target_center = (target_hi + target_lo) / 2
# target_half = (target_hi - target_lo) / 2

# # 각 격자점에서 중심까지 측별 거리에서 반너비를 뺀 값
# # d < 0 면 그 축 방향으론 사각형 안쪽
# target_dist = jnp.abs(grid.states[..., :] - target_center) - target_half

# Signed distance 정의
# values = (jnp.linalg.norm(jnp.maximum(target_dist, 0.), axis=-1) # 
        #   + jnp.minimum(jnp.max(target_dist, axis=-1), 0))

for trim_index in range(19):
    u4Dynamics = hj.systems.U4Linear(cfg, trim_index)
    
    # grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    #     hj.sets.Box(
    #         np.array([-5., -5., ])
    #     )
    # )
    # values =  
    solver_settings = hj.SolverSettings.with_accuracy("high",
                                                      hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    time = 0
    target_time = cfg['hj_analysis_config']['time']
    
    target_values = hj.step(solver_settings, u4Dynamics, grid, time, values, target_time)
    