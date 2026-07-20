import reflex as rx

from frontend.components.data_table import data_table
from frontend.components.json_modal import json_modal
from frontend.states.constants_state import ConstantsState, COLUMNS
from frontend.states.snapshots_state import SnapshotsState

_ACTION_BUTTONS = [
    {
        "icon": "arrow-right",
        "tooltip": "Ver detalle",
        "color_scheme": "blue",
        "on_click": lambda row: SnapshotsState.load_current_snapshot(row.name),
    },
]


def _on_row_click(row) -> rx.event:
    return lambda: ConstantsState.open_modal(row.id)


def _new_modal() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.vstack(
                rx.dialog.title(
                    "New Constants",
                    font_size="1.25rem",
                    font_weight="bold",
                ),
                rx.text_area(
                    value=ConstantsState.new_constants_raw,
                    on_change=ConstantsState.set_new_constants_raw,
                    min_height="300px",
                    width="100%",
                    font_family="monospace",
                    font_size="0.875rem",
                    spell_check=False,
                ),
                rx.hstack(
                    rx.dialog.close(
                        rx.button(
                            "Cancelar",
                            variant="soft",
                            color_scheme="gray",
                        ),
                    ),
                    rx.spacer(),
                    rx.button(
                        "Save",
                        color_scheme="green",
                        on_click=ConstantsState.save_new_constants,
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
        open=ConstantsState.show_new_modal,
    )


def constants() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.hstack(
                rx.heading(
                    "Constants",
                    size="7",
                    color="#f1f5f9",
                ),
                rx.spacer(),
                rx.button(
                    rx.icon("plus"),
                    "New",
                    color_scheme="green",
                    variant="solid",
                    on_click=ConstantsState.open_new_modal,
                ),
                width="100%",
                margin_bottom="1rem",
            ),
            data_table(
                columns=COLUMNS,
                data=ConstantsState.rows,
                on_click=_on_row_click,
                action_buttons=_ACTION_BUTTONS,
            ),
            json_modal(
                title="Detalles",
                json_content=ConstantsState.selected_json,
                is_open=ConstantsState.show_modal,
                on_close=ConstantsState.close_modal,
            ),
            _new_modal(),
            width="100%",
            spacing="4",
            padding_y="2rem",
        ),
        max_width="1200px",
        padding_x="1rem",
    )
