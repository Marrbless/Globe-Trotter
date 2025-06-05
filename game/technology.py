from __future__ import annotations

"""Technology progression utilities."""

from enum import Enum, auto


class TechLevel(Enum):
    """Available technology eras."""

    PRIMITIVE = auto()
    MEDIEVAL = auto()
    INDUSTRIAL = auto()

