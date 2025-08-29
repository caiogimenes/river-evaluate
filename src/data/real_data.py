from .abalone import Abalone
from .wine import Wine
from river.datasets import Bikes


def get_real_datasets():
    """
    Retorna um dicionário de 'fábricas' de dataset.
    Cada item é uma função que, quando chamada, retorna um novo stream.
    """
    return {
        # "bikes": lambda: Bikes(),
        "abalone": lambda: Abalone(numerical_only=True),
        "wine": lambda: Wine(numerical_only=True)
    }
