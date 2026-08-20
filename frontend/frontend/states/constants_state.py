import json

import reflex as rx
from pydantic import ValidationError

from frontend.domain.types.constants import Constants
from frontend.infrastructure.simulator import SimulatorService

COLUMNS = [
    {"key": "id", "header": "ID"},
    {"key": "name", "header": "Nombre"},
    {"key": "confinement", "header": "Confinamiento"},
    {"key": "ruta", "header": "Guardar-Ruta"},
    {"key": "version", "header": "Version"},
]

_NEW_DEFAULTS = {
    "name": "simulacion-test",
    "sim_props": {
        "g": 9,
        "k": 10,
        "dt": 0.05,
        "min_vel": 0,
        "k_confinement": 0.5,
        "beta": 0.6,
        "r_confinement": 12.0,
    },
    "friction": 0,
    "confinement": "radial",
    "ruta": False,
    "version": "v1",
    "barra_height": 0,
    "barra_qlamb": 0,
}


class ConstantsState(rx.State):
    rows: list[Constants] = []
    selected_row: Constants | None = None
    show_modal: bool = False
    show_new_modal: bool = False
    show_confirm_delete: bool = False
    confirm_delete_id: str = ""
    new_constants_raw: str = ""

    def set_new_constants_raw(self, new_value: str):
        self.new_constants_raw = new_value

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

    def open_new_modal(self):
        self.new_constants_raw = json.dumps(_NEW_DEFAULTS, indent=2)
        self.show_new_modal = True

    def close_new_modal(self):
        self.show_new_modal = False
        self.new_constants_raw = ""

    @rx.event
    async def save_new_constants(self):
        try:
            data = json.loads(self.new_constants_raw)
        except json.JSONDecodeError as e:
            yield rx.toast.error(f"JSON inválido: {e}")
            return

        data.pop("id", None)
        try:
            validated = Constants.model_validate(data)
        except ValidationError as e:
            yield rx.toast.error(f"Datos inválidos: {e}")
            return

        service = SimulatorService()
        try:
            created = await service.constants_creator(validated.model_dump(mode="json"))
        except Exception as e:
            yield rx.toast.error(f"Error al guardar: {e}")
            return

        self.rows.append(created)
        self.close_new_modal()
        yield rx.toast.success(f"Constants '{created.name}' creadas")

    @rx.event
    async def delete_constants(self, row_id: str):
        service = SimulatorService()
        try:
            deleted = await service.constants_deleter(row_id)
            if not deleted:
                yield rx.toast.error("No se encontraron las constants")
                return
            self.rows = await service.constants_finder()
            yield rx.toast.success("Constants eliminadas")
        except Exception as e:
            yield rx.toast.error(f"Error al eliminar: {e}")

    def open_delete_confirm(self, row_id: str):
        self.confirm_delete_id = row_id
        self.show_confirm_delete = True

    def close_delete_confirm(self):
        self.show_confirm_delete = False

    @rx.event
    async def confirm_delete_constants(self):
        row_id = self.confirm_delete_id
        self.show_confirm_delete = False
        async for event in self.delete_constants(row_id):
            yield event

    @rx.var(cache=True)
    def selected_json(self) -> str:
        if self.selected_row is None:
            return ""
        return self.selected_row.model_dump_json(indent=2)
