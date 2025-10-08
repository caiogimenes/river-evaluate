from .airquality import AirQuality
from .yolanda import Yolanda
from .abalone import Abalone
from .wine import Wine
from .nasa import NASA
from .diamonds import Diamonds
from .energy import AppliancesEnergy
from river.datasets import Bikes
from .cal_housing import CalHousing
from .ailerons import Ailerons


def get_real_datasets():
    """
    Retorna um dicionário de 'fábricas' de dataset.
    Cada item é uma função que, quando chamada, retorna um novo stream.
    """
    return {
        # "airquality": lambda: AirQuality(numeric_only=True),
        "abalone": lambda: Abalone(numerical_only=True),
        "wine": lambda: Wine(numerical_only=True),
        "california_housing": lambda: CalHousing(),
        "ailerons": lambda: Ailerons(numeric_only=True),
        # "yolanda": lambda: Yolanda(numerical_only=True),
        # "nasa": lambda: NASA(numerical_only=True),
        # "diamonds": lambda: Diamonds(numeric_only=True),
        # "energy": lambda: AppliancesEnergy(numeric_only=True),
    }
