from collections.abc import Callable

import reflex as rx


def _gravity_number_field(
    label: str,
    value,
    on_change: Callable,
    disabled: bool,
    min_value: str = "0",
    step: str = "1",
) -> rx.Component:
    return rx.vstack(
        rx.text(
            label,
            color="#94a3b8",
            font_size="0.75rem",
        ),
        rx.input(
            type="number",
            step=step,
            min=min_value,
            value=value,
            on_change=on_change,
            disabled=disabled,
            width="110px",
        ),
        spacing="1",
    )


def gravity_config_input(
    disabled: bool,
    start: int,
    end: int,
    delta_g: float,
    on_start_change: Callable,
    on_end_change: Callable,
    on_delta_g_change: Callable,
) -> rx.Component:
    return rx.hstack(
        _gravity_number_field("Start", start, on_start_change, disabled),
        _gravity_number_field("End", end, on_end_change, disabled),
        _gravity_number_field(
            "Δg",
            delta_g,
            on_delta_g_change,
            disabled,
            min_value="0.01",
            step="any",
        ),
        spacing="2",
    )