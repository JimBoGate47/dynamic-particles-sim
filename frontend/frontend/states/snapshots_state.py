import json

import plotly.graph_objects as go
import reflex as rx
from loguru import logger

from frontend.domain.types.snapshots import SnapshotsCollection
from frontend.infrastructure.simulator import SimulatorService

SIMULATION_COLUMNS = [
    {"key": "meta_id", "header": "META-ID"},
]


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
    snapshots: SnapshotsCollection | None = None
    show_play_modal: bool = False
    show_json_modal: bool = False
    selected_row: SnapshotsCollection | None = None
    slider_value: int = 0

    async def load_current_snapshot(self, constants_name: str):
        logger.info("Loading snapshots for {}", constants_name)
        service = SimulatorService()
        self.snapshots = await service.snapshot_lister(constants_name)
        return rx.redirect("/snapshots")

    def open_play_modal(self, _: str = ""):
        self.slider_value = 0
        self.show_play_modal = True

    def close_play_modal(self):
        self.show_play_modal = False
        self.slider_value = 0

    def open_modal(self, meta_id: str):
        if self.snapshots is None or self.snapshots.meta_id != meta_id:
            return
        self.selected_row = self.snapshots
        self.show_json_modal = True

    def close_modal(self):
        self.show_json_modal = False
        self.selected_row = None

    def set_slider(self, value: int | str):
        v = int(value)
        self.slider_value = max(0, min(v, self.max_slider))

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
                "meta_id": self.selected_row.meta_id,
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
