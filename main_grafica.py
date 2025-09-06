import warnings

import pandas as pd
import torch

from backend.src.common.domain.entities.properties import SimulationProps, PhysicalProps
from backend.src.simulator.domain.entities.particle_system import ParticleSystem2DTensor
from plotting.plot_particles2 import plot_data

warnings.warn(
    "This function is deprecated. Please use `cli.py` and `cli_plot.py` instead.",
    DeprecationWarning,
)

RADIO = 6.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ps = ParticleSystem2DTensor(
    pos=ParticleSystem2DTensor.initialize_particles_in_circle(
        n_particles=64,
        R=RADIO,
        device=device,
    ),
    # sim_props=SimulationProperties(
    sim_props=SimulationProps(
        g=9.81,
        k=9e9,
        min_vel=1e3,
        r_confinement=RADIO,
        beta=0.5,
        dt=0.1,
    ),
    phys_props=PhysicalProps(
        q=1.0,
        m=1.0,
    )
)
evolucion = ps.to_dict

for i in range(1000):
    ps.velocity_verlet_step()
    evolucion += ps.to_dict

res_df = pd.DataFrame(evolucion)
print(res_df.head())
plot_data(res_df, x="rx", y="ry", animation_frame="step", hover_name="step",
          # range_x=[-1.2, 1.2],
          # range_y=[-1.2, 1.2],
          range_x=[-10, 10],
          range_y=[-10, 10],
          )
