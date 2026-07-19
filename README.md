# Dynamic Confined Particles

Simulación de dinámica de partículas confinadas en un espacio acotado. Modela interacciones electrostáticas, efectos de confinamiento y comportamiento dinámico.

## Estructura del proyecto

```
├── backend/           # Motor de simulación + API FastAPI
│   ├── cli.py         # Script CLI: ejecutar simulación y guardar en DB
│   ├── cli_plot.py    # Script CLI: graficar snapshots desde DB
│   ├── cli_tests.py   # Script CLI: pruebas rápidas de simulación
│   ├── config/        # Configuración (DB, etc.)
│   ├── plotting/      # Utilidades de gráficos
│   ├── src/           # Código fuente del dominio/simulador
│   └── tests/
├── frontend/          # Interfaz de usuario con Reflex
│   └── frontend/
├── docker-compose.yml # MongoDB + mongo-express
└── pyproject.toml     # Workspace root uv
```

## Docker Setup (opcional)

Si prefieres no instalar MongoDB manualmente:

```bash
docker compose up -d
```

Inicia **MongoDB** (`mongo:8.3`) y **mongo-express** (web UI en `http://localhost:8081`).

## Requisitos

Usamos [**uv**](https://docs.astral.sh/uv/) para gestión de entornos y dependencias.

### Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/JimBoGate47/dynamic-particles-sim.git
cd dynamic-particles-sim
```

2. Instalar `uv` (si no lo tienes):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Sincronizar dependencias (instala dependencias del backend + frontend):

```bash
uv sync --all-packages
```

4. Activar el entorno virtual:

```bash
source .venv/bin/activate
```

## Uso

### Backend — Scripts de simulación (ejecutar desde `backend/`)

```bash
cd backend
python cli.py         # Ejecutar simulación y guardar en DB
python cli_plot.py    # Graficar resultados desde DB
python cli_tests.py   # Pruebas rápidas
```

### Frontend — Interfaz Reflex

```bash
cd frontend
reflex run
```

## Tests

```bash
cd backend
pytest tests/
```

## 📚 Citation

Si este código te resulta útil, por favor cita la publicación original:

> **Sirpa-Poma, J. W., Ghezzi, F., & Ramírez-Ávila, G. M. (2023).**  
> *The equilibrium of particles electrostatically confined by external forces and the competition between the electrostatic interaction and gravitational field.*  
> *Journal of Electrostatics, 126*, 103860.  
> [https://doi.org/10.1016/j.elstat.2023.103860](https://doi.org/10.1016/j.elstat.2023.103860)
