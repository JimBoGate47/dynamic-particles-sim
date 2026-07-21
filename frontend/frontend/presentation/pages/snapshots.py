import reflex as rx

from frontend.components.data_table import data_table
from frontend.components.dynamic_plotly import plotly_modal
from frontend.components.json_modal import json_modal
from frontend.states.snapshots_state import SnapshotsState, SIMULATION_COLUMNS

_PLAY_BUTTONS = [
    {
        "icon": "eye",
        "tooltip": "Explorar datos",
        "color_scheme": "green",
        "on_click": lambda row: lambda: SnapshotsState.open_play_modal(row.batch_id),
    },
    {
        "icon": "play",
        "tooltip": "Ejecutar simulación",
        "color_scheme": "blue",
        "on_click": lambda row: lambda: SnapshotsState.run_simulation(row.batch_id),
    },
]


def _on_row_click(row) -> rx.event:
    return lambda: SnapshotsState.open_modal(row.batch_id)


def _new_modal() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.vstack(
                rx.dialog.title(
                    "New Snapshot",
                    font_size="1.25rem",
                    font_weight="bold",
                ),
                rx.text_area(
                    value=SnapshotsState.new_snapshot_raw,
                    on_change=SnapshotsState.set_new_snapshot_raw,
                    min_height="300px",
                    width="100%",
                    font_family="monospace",
                    font_size="0.875rem",
                    spell_check=False,
                ),
                rx.upload(
                    rx.button(
                        rx.icon("upload"),
                        "Cargar JSON",
                        color_scheme="blue",
                        variant="soft",
                    ),
                    on_drop=SnapshotsState.handle_upload,
                    accept={".json": ["application/json"]},
                    max_files=1,
                    multiple=False,
                ),
                rx.hstack(
                    rx.dialog.close(
                        rx.button(
                            "Cancelar",
                            variant="soft",
                            color_scheme="gray",
                            on_click=SnapshotsState.close_new_modal,
                        ),
                    ),
                    rx.spacer(),
                    rx.button(
                        "Save",
                        color_scheme="green",
                        on_click=SnapshotsState.save_new_snapshot,
                    ),
                    width="100%",
                ),
                width="100%",
                spacing="3",
            ),
            style=dict(
                background="#1e293b",
                border="1px solid #334155",
                max_width="700px",
                width="90%",
            ),
        ),
        open=SnapshotsState.show_new_modal,
    )


def snapshots() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.hstack(
                rx.heading(
                    "Snapshot Collections",
                    size="7",
                    color="#f1f5f9",
                ),
                rx.spacer(),
                rx.button(
                    rx.icon("plus"),
                    "New",
                    color_scheme="green",
                    variant="solid",
                    on_click=SnapshotsState.open_new_modal,
                ),
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
                data=SnapshotsState.collections,
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
                on_download=SnapshotsState.download_snapshot,
            ),
            json_modal(
                title="Snapshots Collection",
                json_content=SnapshotsState.selected_json,
                is_open=SnapshotsState.show_json_modal,
                on_close=SnapshotsState.close_modal,
            ),
            _new_modal(),
            width="100%",
            spacing="4",
            padding_y="2rem",
        ),
        max_width="1200px",
        padding_x="1rem",
    )
