from beanie import Document, Indexed


class ConstantsORM(Document):
    name: Indexed(str, unique=True)
    g: float
    k: float
    dt: float
    min_vel: float
    friction: float = 0
    confinement: str = 0
    r_confinement: float = 0
    ruta: bool = False
    version: str = "v1"
    barra_height: float = 0
    barra_qlamb: float = 0

    @property
    def to_dict(self) -> dict:
        return {

        }
