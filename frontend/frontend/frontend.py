import reflex as rx
from frontend.presentation.pages.constants import constants
from frontend.presentation.pages.snapshots import snapshots


app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="blue",
        gray_color="slate",
    ),
)
app.add_page(constants, route="/")
app.add_page(snapshots, route="/snapshots")
