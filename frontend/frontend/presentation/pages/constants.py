import reflex as rx
from frontend.states.constants_state import ConstantsState, COLUMNS
from frontend.components.data_table import data_table
from frontend.components.json_modal import json_modal


_ACTION_BUTTONS = [
    {
        "icon": "arrow-right",
        "tooltip": "Ver detalle",
        "color_scheme": "blue",
        "on_click": lambda row: rx.redirect("/snapshots"),
    },
]


def _on_row_click(row) -> rx.event:
    return lambda: ConstantsState.open_modal(row["id"])


def constants() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading(
                "Proyectos - Tabla Principal",
                size="7",
                color="#f1f5f9",
                margin_bottom="1rem",
            ),
            data_table(
                columns=COLUMNS,
                data=ConstantsState.rows,
                on_click=_on_row_click,
                action_buttons=_ACTION_BUTTONS,
            ),
            json_modal(
                title="Datos del proyecto",
                json_content=ConstantsState.selected_json,
                is_open=ConstantsState.show_modal,
                on_close=ConstantsState.close_modal,
            ),
            width="100%",
            spacing="4",
            padding_y="2rem",
        ),
        max_width="1200px",
        padding_x="1rem",
    )
