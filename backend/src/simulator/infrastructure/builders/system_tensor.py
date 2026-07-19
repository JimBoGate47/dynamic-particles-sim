import torch

from src.common.domain.entities import Snapshot
from src.common.domain.entities.properties import PhysicalProps
from src.simulator.domain.entities.particle_system import System2DTensor


def build_system_tensor(
        snapshot: Snapshot,
        device=None,
) -> System2DTensor:
    """
    Construye un ParticleSystem2DTensor a partir de ParticleSystem2D.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    particles = snapshot.particles
    n = len(particles)

    # --- Crear tensores vacíos ---
    pos = torch.zeros((n, 2), device=device)
    vel = torch.zeros((n, 2), device=device)
    acc = torch.zeros((n, 2), device=device)
    q = torch.zeros((n, 1), device=device)
    m = torch.zeros((n, 1), device=device)

    # --- Poblarlos desde ParticleSystem2D ---
    for i, p in enumerate(particles):
        pos[i] = torch.tensor([p.r[0], p.r[1]], dtype=torch.float32, device=device)
        vel[i] = torch.tensor([p.v[0], p.v[1]], dtype=torch.float32, device=device)
        acc[i] = torch.tensor([p.a[0], p.a[1]], dtype=torch.float32, device=device)

        # phys_props es un dict {'q': , 'm': }
        q[i] = float(p.phys_props["q"])
        m[i] = float(p.phys_props["m"])

    phys_props = PhysicalProps(q=q, m=m)

    return System2DTensor(
        pos=pos,
        vel=vel,
        acc=acc,
        phys_props=phys_props,
    )
