from collections.abc import Callable
import reflex as rx


def _header_cell_style() -> dict:
    return {
        "background": "#1e293b",
        "padding": "12px 16px",
        "border-bottom": "2px solid #334155",
        "text-align": "left",
        "font-weight": "bold",
        "color": "white",
    }


def _cell_style() -> dict:
    return {
        "background": "#1e293b",
        "padding": "10px 16px",
        "border-bottom": "1px solid #334155",
        "color": "#e2e8f0",
    }


def _render_header(columns: list[dict], has_actions: bool) -> rx.Component:
    cells = [
        rx.el.th(col["header"], style=_header_cell_style())
        for col in columns
    ]
    if has_actions:
        cells.append(rx.el.th("Acciones", style=_header_cell_style()))
    return rx.el.thead(rx.el.tr(*cells))


def _render_body(
    data,
    columns: list[dict],
    action_buttons: list[dict] | None,
    on_click: Callable | None,
) -> rx.Component:
    has_actions = action_buttons is not None

    def _row_template(row):
        click_handler = on_click(row) if on_click else None
        cells = [
            rx.el.td(
                rx.button(
                    row[col["key"]],
                    on_click=click_handler,
                    variant="ghost",
                    size="1",
                    padding="0",
                    height="auto",
                    color="#e2e8f0",
                    font_weight="normal",
                    cursor="pointer" if click_handler else "default",
                    _hover={"background": "transparent"},
                ),
                style=_cell_style(),
            )
            for col in columns
        ]
        if has_actions:
            buttons = [
                rx.icon_button(
                    rx.icon(btn["icon"], color="white"),
                    on_click=btn["on_click"](row),
                    tooltip=btn.get("tooltip", ""),
                    variant="ghost",
                    size="1",
                    color_scheme=btn.get("color_scheme", "blue"),
                )
                for btn in action_buttons
            ]
            cells.append(
                rx.el.td(
                    rx.hstack(*buttons, spacing="1"),
                    style=_cell_style(),
                )
            )
        return rx.el.tr(
            *cells,
            style={"background": "#1e293b"},
            _hover={"background": "#0f172a"} if has_actions else {"background": "#0f172a"},
        )

    return rx.el.tbody(
        rx.foreach(data, _row_template),
    )


def data_table(
    columns: list[dict],
    data,
    on_click: Callable | None = None,
    action_buttons: list[dict] | None = None,
) -> rx.Component:
    return rx.container(
        rx.el.table(
            _render_header(columns, action_buttons is not None),
            _render_body(data, columns, action_buttons, on_click),
            style={
                "width": "100%",
                "border-collapse": "collapse",
                "background": "#1e293b",
                "border-radius": "8px",
                "overflow": "hidden",
            },
        ),
        overflow_x="auto",
        width="100%",
        padding="0",
    )
