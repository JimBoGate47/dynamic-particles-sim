import reflex as rx
from frontend.states.snapshots_state import SnapshotsState, SIMULATION_COLUMNS
from frontend.components.data_table import data_table
from frontend.components.dynamic_plotly import plotly_modal


_PLAY_BUTTONS = [
    {
        "icon": "play",
        "tooltip": "Explorar datos",
        "color_scheme": "green",
        "on_click": lambda row: lambda: SnapshotsState.open_play_modal(row["id"]),
    },
]


def snapshots() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.hstack(
                rx.heading(
                    "Snapshots",
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
                data=SnapshotsState.experiments,
                on_click=None,
                action_buttons=_PLAY_BUTTONS,
            ),
            plotly_modal(
                figure_json=SnapshotsState.figure_json,
                is_open=SnapshotsState.show_modal,
                slider_value=SnapshotsState.current_idx,
                max_slider=SnapshotsState.max_slider,
                on_slider_change=SnapshotsState.set_slider,
                on_close=SnapshotsState.close_play_modal,
            ),
            width="100%",
            spacing="4",
            padding_y="2rem",
        ),
        max_width="1200px",
        padding_x="1rem",
    )
