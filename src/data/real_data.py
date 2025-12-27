from .airquality import AirQuality
from .abalone import Abalone
from .wine import Wine
from river.datasets import Bikes, Elec2
from .cal_housing import CalHousing
from .ailerons import Ailerons
from .covertype import CoverType

def get_real_datasets():
    """
    Retorna um dicionário de 'fábricas' de dataset.
    Cada item é uma função que, quando chamada, retorna um novo stream.
    """
    return {
        "ailerons": lambda: Ailerons(),
        "airquality": lambda: AirQuality(),
        "abalone": lambda: Abalone(),
        "wine": lambda: Wine(),
        "california_housing": lambda: CalHousing(),
        "bikes": lambda : Bikes(),
        "elec2": lambda : Elec2(),
        "covertype": lambda : CoverType(),
    }
