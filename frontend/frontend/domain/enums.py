from enum import Enum


class ConfinementType(str, Enum):
    POTENCIAL = "potencial"
    WCA = "wca"
    HARMONIC = "harmonic"
    HARD_WALL = "hard_wall"
