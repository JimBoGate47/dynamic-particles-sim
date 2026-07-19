# Dynamic Confined Particles

This repository contains code for simulating the dynamics of particles confined within a bounded space. The simulation models particle interactions, confinement effects, and dynamic behavior over time.

## Docker Setup (optional)

If you prefer not to install MongoDB manually, you can run the services with Docker Compose:

1. Install [Docker](https://docs.docker.com/engine/install/).
2. Launch in the background:

```bash
docker compose up -d
```

This starts **MongoDB** (`mongo:8.3`) and **mongo-express** (web UI at `http://localhost:8081`).

## Requirements

This project uses [**uv**](https://docs.astral.sh/uv/) for environment and dependency management.

> Make sure you have the Python version specified in `.python-version` installed.

### Installation

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/JimBoGate47/dynamic-particles-sim.git
cd dynamic-particles-sim
```

2. Install `uv` following the official instructions:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Sync dependencies:

```bash
uv sync
```

4. Activate the virtual environment:

```bash
source .venv/bin/activate
```

5. Run the test scripts:

```bash
python cli.py
python cli_plot.py
```

> ⚠️ **Note:** The simulation results may vary depending on the initial conditions of the particles.

## 📚 Citation

If you find this code useful for your research or work, please consider citing the associated publication where this
simulation framework was originally developed and applied.  
Proper citation helps support the continued development of open research tools.

> **Sirpa-Poma, J. W., Ghezzi, F., & Ramírez-Ávila, G. M. (2023).**  
> *The equilibrium of particles electrostatically confined by external forces and the competition between the
electrostatic interaction and gravitational field.*  
> *Journal of Electrostatics, 126*, 103860.  
> [https://doi.org/10.1016/j.elstat.2023.103860](https://doi.org/10.1016/j.elstat.2023.103860)
