import json
import math

import plotly.graph_objects as go
import reflex as rx

from frontend.domain.types.snapshots import SnapshotsCollection
from frontend.infrastructure.simulator import SimulatorService

SIMULATION_COLUMNS = [
    {"key": "meta_id", "header": "META-ID"},
]

_DATA_X = [round(i * 0.2, 1) for i in range(60)]
_DATA_Y1 = [round(5 * math.sin(2 * math.pi * 0.5 * t) + (hash(str(t)) % 100) / 100 * 0.4 - 0.2, 2) for t in _DATA_X]
_DATA_Y2 = [round(3 * math.cos(2 * math.pi * 0.3 * t) + (hash(str(t * 2)) % 100) / 100 * 0.3 - 0.15, 2) for t in
            _DATA_X]
_MAX_IDX = len(_DATA_X) - 1


class SnapshotsState(rx.State):
    snapshots: SnapshotsCollection | None = None
    show_play_modal: bool = False
    show_json_modal: bool = False
    selected_row: SnapshotsCollection | None = None
    slider_value: int = 0

    async def load_current_snapshot(self, constants_name: str):
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
        self.slider_value = max(0, min(v, _MAX_IDX))

    @rx.var(cache=True)
    def max_slider(self) -> int:
        return _MAX_IDX

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
    def figure_json(self) -> dict:
        n = self.slider_value + 1
        x = _DATA_X[:n]
        y1 = _DATA_Y1[:n]
        y2 = _DATA_Y2[:n]
        exp_name = (
            self.snapshots.snapshots[0].constants.name
            if self.snapshots and self.snapshots.snapshots
            else "Simulacion"
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y1,
                mode="lines+markers",
                name="Senal A",
                line=dict(color="#6366f1", width=2),
                marker=dict(size=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y2,
                mode="lines+markers",
                name="Senal B",
                line=dict(color="#f43f5e", width=2),
                marker=dict(size=4),
            )
        )
        fig.update_layout(
            title=f"Explorador - {exp_name}",
            xaxis_title="Tiempo (s)",
            yaxis_title="Amplitud",
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(range=[0, _DATA_X[-1]]),
            yaxis=dict(range=[-6, 6]),
        )
        return fig.to_plotly_json()
