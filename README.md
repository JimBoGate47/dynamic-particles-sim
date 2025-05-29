# Dynamic Confined Particles

This repository contains the code to simulate the dynamics of confined particles.

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

5. Run the test script:
```bash
python main_grafica.py
```

> ⚠️ **Note:** The simulation results may vary depending on the initial conditions of the particles.
