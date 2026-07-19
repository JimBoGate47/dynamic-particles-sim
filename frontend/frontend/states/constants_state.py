import json
import reflex as rx

ROWS = [
    {
        "id": 1,
        "name": "nombre",
        "confinement": "conf_value",
        "ruta": "false",
        "version": "vx",
    },
]

COLUMNS = [
    {"key": "id", "header": "ID"},
    {"key": "name", "header": "Nombre"},
    {"key": "confinement", "header": "Confinamiento"},
    {"key": "ruta", "header": "Guardar-Ruta"},
    {"key": "version", "header": "Version"},
]


class ConstantsState(rx.State):
    rows: list[dict] = ROWS
    selected_row: dict = {}
    show_modal: bool = False

    def open_modal(self, row_id: int):
        for row in self.rows:
            if row["id"] == row_id:
                self.selected_row = row
                break
        self.show_modal = True

    def close_modal(self):
        self.show_modal = False
        self.selected_row = {}

    @rx.var(cache=True)
    def selected_json(self) -> str:
        if not self.selected_row:
            return ""
        return json.dumps(self.selected_row, indent=2, ensure_ascii=False)
