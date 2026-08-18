import json

import plotly.graph_objects as go
import reflex as rx
from loguru import logger
from pydantic import ValidationError

from frontend.domain.enums import ConfinementType
from frontend.domain.types.gravity import GravityConfig
from frontend.domain.types.snapshots import Particle, Snapshot, SnapshotsCollection
from frontend.infrastructure.simulator import SimulatorService

SIMULATION_COLUMNS = [
    {"key": "batch_id", "header": "Batch ID"},
]

CONFINEMENT_TYPES = [member.value for member in ConfinementType]


def _collect_particle_data(snapshots: list) -> dict:
    rx_list, ry_list, charge_list, step_list, p_idx_list = [], [], [], [], []

    for snapshot in snapshots:
        for idx, particle in enumerate(snapshot.particles):
            rx_list.append(particle.r[0])
            ry_list.append(particle.r[1])
            charge_list.append(particle.phys_props.get("q", 0))
            step_list.append(snapshot.step)
            p_idx_list.append(idx)

    return dict(
        rx_list=rx_list,
        ry_list=ry_list,
        charge_list=charge_list,
        step_list=step_list,
        p_idx_list=p_idx_list,
    )


def _build_figure(
    rx_list: list,
    ry_list: list,
    charge_list: list,
    step_list: list,
    p_idx_list: list,
    title: str,
) -> go.Figure:
    fig = go.Figure()

    charge_abs = [abs(c) for c in charge_list]
    max_c = max(charge_abs) if charge_abs else 1
    sizes = [max(3, c / max_c * 25) for c in charge_abs]

    fig.add_trace(
        go.Scatter(
            x=rx_list,
            y=ry_list,
            mode="markers",
            marker=dict(
                size=sizes,
                color=charge_list,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Charge"),
            ),
            text=[
                f"Step: {s} | P:{p} | q={c:.4f}"
                for s, p, c in zip(step_list, p_idx_list, charge_list)
            ],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=f"Explorador - {title}",
        xaxis_title="rx",
        yaxis_title="ry",
        template="plotly_dark",
        hovermode="closest",
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(range=[-20, 20]),
        yaxis=dict(range=[-20, 20]),
    )

    return fig


class SnapshotsState(rx.State):
    collections: list[SnapshotsCollection] = []
    snapshots: SnapshotsCollection | None = None
    show_play_modal: bool = False
    show_json_modal: bool = False
    show_new_modal: bool = False
    selected_row: SnapshotsCollection | None = None
    slider_value: int = 0
    new_snapshot_raw: str = ""
    add_gravity: bool = False
    gravity_config: GravityConfig = GravityConfig()
    confinement: str = ConfinementType.HARMONIC.value
    _current_constants_id: str = ""
    _current_constants_name: str = ""

    def set_add_gravity(self, value: bool):
        self.add_gravity = value

    def set_gravity_start(self, value: int | str):
        try:
            self.gravity_config = self.gravity_config.model_copy(
                update={"start": int(value)},
            )
        except (TypeError, ValueError):
            pass

    def set_gravity_end(self, value: int | str):
        try:
            self.gravity_config = self.gravity_config.model_copy(
                update={"end": int(value)},
            )
        except (TypeError, ValueError):
            pass

    def set_gravity_delta_g(self, value: int | str):
        self.gravity_config = self.gravity_config.model_copy(
            update={"delta_g": float(value)},
        )

    def set_confinement(self, value: str):
        self.confinement = value

    async def load_current_snapshot(self, constants_name: str, constants_id: str = ""):
        logger.info("Loading snapshots for {}", constants_name)
        self._current_constants_name = constants_name
        self._current_constants_id = constants_id
        service = SimulatorService()
        self.collections = await service.snapshot_lister(constants_name)
        self.snapshots = self.collections[-1] if self.collections else None
        return rx.redirect("/snapshots")

    def open_play_modal(self, batch_id: str = ""):
        self.slider_value = 0
        for col in self.collections:
            if col.batch_id == batch_id:
                self.snapshots = col
                break
        self.show_play_modal = True

    @rx.event
    async def run_simulation(self, batch_id: str):
        service = SimulatorService()
        try:
            col = next((c for c in self.collections if c.batch_id == batch_id), None)
            if not col or not col.snapshots:
                yield rx.toast.error("No snapshots in collection")
                return
            snapshot = col.snapshots[-1]
            wall = ConfinementType(self.confinement)
            if self.add_gravity:
                snapshots = await service.simulation_plus_gravity_runner(
                    snapshot_id=snapshot.id,
                    gravity_config=self.gravity_config,
                    wall=wall,
                )
            else:
                snapshots = await service.simulation_runner(
                    snapshot_id=snapshot.id,
                    wall=wall,
                )
            yield rx.toast.success(f"Simulación completada: {len(snapshots)} snapshots")
            yield await self.load_current_snapshot(
                self._current_constants_name,
                self._current_constants_id,
            )
        except Exception as e:
            yield rx.toast.error(f"Error en simulación: {e}")

    def close_play_modal(self):
        self.show_play_modal = False
        self.slider_value = 0

    @rx.event
    async def download_snapshot(self):
        if not self.snapshots:
            return
        current = self.snapshots.snapshots[self.slider_value]
        if not current.id:
            return
        service = SimulatorService()
        found = await service.snapshot_finder(current.id)
        return rx.download(
            data=found.model_dump_json(indent=2),
            filename=f"snapshot_{found.step}.json",
        )

    def open_modal(self, batch_id: str):
        for col in self.collections:
            if col.batch_id == batch_id:
                self.selected_row = col
                self.snapshots = col
                break
        self.show_json_modal = True

    def close_modal(self):
        self.show_json_modal = False
        self.selected_row = None

    def set_slider(self, value: int | str):
        v = int(value)
        self.slider_value = max(0, min(v, self.max_slider))

    def set_new_snapshot_raw(self, new_value: str):
        self.new_snapshot_raw = new_value

    def open_new_modal(self):
        self.new_snapshot_raw = json.dumps({
            "step": 0,
            "constants_id": self._current_constants_id,
            "particles": [],
        }, indent=2)
        self.show_new_modal = True

    def close_new_modal(self):
        self.show_new_modal = False
        self.new_snapshot_raw = ""

    async def upload_snapshot(self, files: list[rx.UploadFile]):
        for file in files:
            content = await file.read()
            try:
                data = json.loads(content)
                Snapshot.model_validate(data)

                data.pop("id", None)
                data.pop("batch_id", None)
                data.pop("constants", None)

                self.new_snapshot_raw = json.dumps(data, indent=2)

            except ValidationError as e:
                yield rx.toast.error(
                    f"JSON inválido: debe ser un Snapshot válido — {e}"
                )
            except Exception as e:
                yield rx.toast.error(f"Error al cargar JSON: {e}")

    @rx.event
    async def save_new_snapshot(self):
        try:
            data = json.loads(self.new_snapshot_raw)
            data["constants_id"] = self._current_constants_id
        except json.JSONDecodeError as e:
            yield rx.toast.error(f"JSON inválido: {e}")
            return

        for p in data.get("particles", []):
            try:
                Particle.model_validate(p)
            except ValidationError as e:
                yield rx.toast.error(f"Partícula inválida: {e}")
                return

        service = SimulatorService()
        try:
            created = await service.snapshot_creator(data)
        except Exception as e:
            yield rx.toast.error(f"Error al guardar: {e}")
            return

        self.close_new_modal()
        yield rx.toast.success(f"Snapshot (step={created.step}) creado")

        if self._current_constants_name:
            self.collections = await service.snapshot_lister(self._current_constants_name)
            self.snapshots = self.collections[-1] if self.collections else None

    @rx.var(cache=True)
    def max_slider(self) -> int:
        if not self.snapshots or not self.snapshots.snapshots:
            return 0
        return len(self.snapshots.snapshots) - 1

    @rx.var(cache=True)
    def current_idx(self) -> int:
        return self.slider_value

    @rx.var(cache=True)
    def selected_json(self) -> str:
        if self.selected_row is None:
            return ""
        return json.dumps(
            {
                "batch_id": self.selected_row.batch_id,
                "steps": self.selected_row.steps,
            },
            indent=2,
        )

    @rx.var(cache=True)
    def figure_json(self) -> go.Figure:
        if not self.snapshots or not self.snapshots.snapshots:
            return go.Figure()

        idx = min(self.slider_value, len(self.snapshots.snapshots) - 1)
        current = self.snapshots.snapshots[idx : idx + 1]
        data = _collect_particle_data(current)

        if not data["rx_list"]:
            return go.Figure()

        title = self.snapshots.snapshots[0].constants.name
        return _build_figure(**data, title=title)
