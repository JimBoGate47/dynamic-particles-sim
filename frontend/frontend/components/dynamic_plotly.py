from collections.abc import Callable
import reflex as rx


PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}


def plotly_modal(
    figure_json: dict,
    is_open: bool,
    slider_value: int,
    max_slider: int,
    on_slider_change: Callable,
    on_close: Callable,
) -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.vstack(
                rx.hstack(
                    rx.text("Paso:", font_size="0.875rem"),
                    rx.text(slider_value, font_weight="bold", font_size="0.875rem"),
                    rx.text("/", font_size="0.875rem", color="#94a3b8"),
                    rx.text(max_slider, font_size="0.875rem", color="#94a3b8"),
                    rx.spacer(),
                    rx.tooltip(
                        rx.icon_button(
                            rx.icon("x"),
                            on_click=on_close,
                            variant="ghost",
                            size="1",
                            color_scheme="gray",
                        ),
                        content="Cerrar",
                    ),
                    spacing="2",
                    width="100%",
                    padding_bottom="0.5rem",
                ),
                rx.plotly(
                    data=figure_json,
                    config=PLOTLY_CONFIG,
                    width="100%",
                    height="450px",
                ),
                rx.hstack(
                    rx.text(slider_value, font_size="0.75rem", color="#94a3b8"),
                    rx.input(
                        type="range",
                        min=0,
                        max=max_slider,
                        value=slider_value,
                        on_change=on_slider_change,
                        width="100%",
                    ),
                    rx.text(max_slider, font_size="0.75rem", color="#94a3b8"),
                    spacing="2",
                    width="100%",
                    align="center",
                ),
                width="100%",
                spacing="3",
            ),
            style=dict(
                background="#1e293b",
                border="1px solid #334155",
                max_width="800px",
                width="90%",
            ),
        ),
        open=is_open,
    )
