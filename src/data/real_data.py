"""Real-world stream factories used in the experiment.

Commented-out datasets are kept for local debugging only and must stay inactive
so the published suite (Bikes, Elec2, CoverType) is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from river.datasets import Bikes, Elec2

from .abalone import Abalone
from .ailerons import Ailerons
from .airquality import AirQuality
from .cal_housing import CalHousing
from .covertype import CoverType
from .wine import Wine


def get_real_datasets() -> dict[str, Callable[[], Any]]:
    """
    Retorna um dicionário de 'fábricas' de dataset.
    Cada item é uma função que, quando chamada, retorna um novo stream.
    """
    return {
        # "ailerons": lambda: Ailerons(),
        # "airquality": lambda: AirQuality(),
        # "abalone": lambda: Abalone(),
        # "wine": lambda: Wine(),
        # "california_housing": lambda: CalHousing(),
        "bikes": lambda : Bikes(),
        "elec2": lambda : Elec2(),
        "covertype": lambda : CoverType(),
    }
