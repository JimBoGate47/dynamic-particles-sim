from collections.abc import Callable
import reflex as rx


def json_modal(
    title: str,
    json_content: str,
    is_open: bool,
    on_close: Callable,
) -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title(
                rx.hstack(
                    rx.text(title, font_size="1.25rem", font_weight="bold"),
                    rx.spacer(),
                    rx.icon_button(
                        rx.icon("x"),
                        on_click=on_close,
                        variant="ghost",
                        size="1",
                    ),
                    width="100%",
                ),
            ),
            rx.dialog.description(
                rx.code_block(
                    json_content,
                    language="json",
                    width="100%",
                    min_height="200px",
                    font_size="0.875rem",
                    border_radius="8px",
                    background="#0f172a",
                ),
                margin_top="1rem",
            ),
            style=dict(
                background="#1e293b",
                border="1px solid #334155",
                max_width="600px",
                width="90%",
            ),
        ),
        open=is_open,
    )
