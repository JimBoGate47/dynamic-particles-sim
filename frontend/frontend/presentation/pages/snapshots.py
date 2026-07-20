import reflex as rx

from frontend.components.data_table import data_table
from frontend.components.dynamic_plotly import plotly_modal
from frontend.components.json_modal import json_modal
from frontend.states.snapshots_state import SnapshotsState, SIMULATION_COLUMNS

_PLAY_BUTTONS = [
    {
        "icon": "play",
        "tooltip": "Explorar datos",
        "color_scheme": "green",
        "on_click": lambda row: lambda: SnapshotsState.open_play_modal(row.meta_id),
    },
]


def _on_row_click(row) -> rx.event:
    return lambda: SnapshotsState.open_modal(row.meta_id)


def snapshots() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.hstack(
                rx.heading(
                    "Snapshot Collection",
                    size="7",
                    color="#f1f5f9",
                ),
                rx.spacer(),
                rx.link(
                    rx.button(
                        rx.icon("arrow-left"),
                        "Volver",
                        variant="soft",
                        color_scheme="gray",
                    ),
                    href="/",
                ),
                width="100%",
                margin_bottom="1rem",
            ),
            data_table(
                columns=SIMULATION_COLUMNS,
                data=[SnapshotsState.snapshots],
                on_click=_on_row_click,
                action_buttons=_PLAY_BUTTONS,
            ),
            plotly_modal(
                figure_json=SnapshotsState.figure_json,
                is_open=SnapshotsState.show_play_modal,
                slider_value=SnapshotsState.current_idx,
                max_slider=SnapshotsState.max_slider,
                on_slider_change=SnapshotsState.set_slider,
                on_close=SnapshotsState.close_play_modal,
            ),
            json_modal(
                title="Snapshots Collection",
                json_content=SnapshotsState.selected_json,
                is_open=SnapshotsState.show_json_modal,
                on_close=SnapshotsState.close_modal,
            ),
            width="100%",
            spacing="4",
            padding_y="2rem",
        ),
        max_width="1200px",
        padding_x="1rem",
    )
