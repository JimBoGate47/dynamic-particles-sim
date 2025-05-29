import pandas as pd
import torch

from backend.src.domain.entities.particle_system import ParticleSystem2D
from backend.src.domain.entities.properties import SimulationProperties, PhysicalProperties
from plotting.plot_particles2 import plot_data

RADIO = 6.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ps = ParticleSystem2D(
    pos=ParticleSystem2D.initialize_particles_in_circle(
        n_particles=64,
        R=RADIO,
        device=device,
    ),
    sim_props=SimulationProperties(
        r_confin=RADIO,
        beta=0.5,
        dt=0.1,
    ),
    phys_props=PhysicalProperties(
        q=1.0,
        m=1.0,
    )
)
evolucion = ps.to_dict

for i in range(1000):
    ps.velocity_verlet_step()
    evolucion += ps.to_dict

res_df = pd.DataFrame(evolucion)
plot_data(res_df, x="rx", y="ry", animation_frame="step", hover_name="step",
          # range_x=[-1.2, 1.2],
          # range_y=[-1.2, 1.2],
          range_x=[-10, 10],
          range_y=[-10, 10],
          )
