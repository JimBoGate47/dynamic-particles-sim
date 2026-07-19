import reflex as rx

from frontend.domain.types.constants import Constants
from frontend.infrastructure.simulator import SimulatorService

COLUMNS = [
    {"key": "id", "header": "ID"},
    {"key": "name", "header": "Nombre"},
    {"key": "confinement", "header": "Confinamiento"},
    {"key": "ruta", "header": "Guardar-Ruta"},
    {"key": "version", "header": "Version"},
]


class ConstantsState(rx.State):
    rows: list[Constants] = []
    selected_row: Constants | None = None
    show_modal: bool = False

    async def load_data(self):
        service = SimulatorService()
        self.rows = await service.constants_finder()

    def open_modal(self, row_id: str):
        for row in self.rows:
            if row.id == row_id:
                self.selected_row = row
                break
        self.show_modal = True

    def close_modal(self):
        self.show_modal = False
        self.selected_row = None

    @rx.var(cache=True)
    def selected_json(self) -> str:
        if self.selected_row is None:
            return ""
        return self.selected_row.model_dump_json(indent=2)
