import reflex as rx


def confirm_dialog(
    is_open: bool | rx.Var,
    title: str,
    description: str,
    confirm_label: str = "Borrar",
    on_confirm: rx.EventHandler | None = None,
    on_cancel: rx.EventHandler | None = None,
) -> rx.Component:
    return rx.alert_dialog.root(
        rx.alert_dialog.content(
            rx.vstack(
                rx.alert_dialog.title(
                    title,
                    font_size="1.25rem",
                    font_weight="bold",
                ),
                rx.alert_dialog.description(
                    description,
                    color="#94a3b8",
                ),
                rx.hstack(
                    rx.alert_dialog.cancel(
                        rx.button(
                            "Cancelar",
                            variant="soft",
                            color_scheme="gray",
                            on_click=on_cancel,
                        ),
                    ),
                    rx.spacer(),
                    rx.alert_dialog.action(
                        rx.button(
                            confirm_label,
                            color_scheme="red",
                            on_click=on_confirm,
                        ),
                    ),
                    width="100%",
                ),
                width="100%",
                spacing="3",
            ),
            style=dict(
                background="#1e293b",
                border="1px solid #334155",
                max_width="420px",
                width="90%",
            ),
        ),
        open=is_open,
    )