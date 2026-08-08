import torch
from pydantic import BaseModel, ConfigDict


# TODO tal vez como id en lugar de UUID?
class SimulationID(BaseModel):
    id: float
    name: float


class PhysicalProps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    q: torch.Tensor
    m: torch.Tensor

    @classmethod
    def from_charges(cls, n_particles: int, charges: list[float], device: torch.device = None):
        """
        Creates a PhysicalProps instance by distributing charges among particles.
        """
        num_charge_values = len(charges)
        base_chunk_size = n_particles // num_charge_values
        remainder = n_particles % num_charge_values

        charge_tensors = []
        for i, charge_value in enumerate(charges):
            chunk_size = base_chunk_size + (1 if i < remainder else 0)
            charge_tensors.append(torch.full((chunk_size,), charge_value, dtype=torch.float32, device=device))

        q = torch.cat(charge_tensors).unsqueeze(1)
        m = torch.ones(
            n_particles,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)  # Assuming mass is 1 for all particles

        return cls(q=q, m=m)


class SimulationProps(BaseModel):
    g: float
    k: float
    min_vel: float
    r_confinement: float
    k_confinement: float
    beta: float
    dt: float
    delta_gravity: float | None = None

    @property
    def delta_gravity_exists(self) -> bool:
        if (
                self.delta_gravity and
                self.delta_gravity > 0
        ):
            return True
        return False
