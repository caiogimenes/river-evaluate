import pytest
from .abalone import Abalone


def test_take():
    abalone = Abalone(numerical_only=False)
    assert next(abalone.take(1)) == ({
                                         'Sex': 'M',
                                         'Length': 0.455,
                                         'Diameter': 0.365,
                                         'Height': 0.095,
                                         'Whole_weight': 0.514,
                                         'Shucked_weight': 0.2245,
                                         'Viscera_weight': 0.101,
                                         'Shell_weight': 0.15}, 15
    )


def test_remove_categorical():
    abalone = Abalone(numerical_only=True)
    types = abalone.X.dtypes
    pass
