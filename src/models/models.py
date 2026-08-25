from __future__ import annotations

from typing import Any

from river import compose, preprocessing, tree

__all__ = ["get_models"]


def get_models() -> dict[str, Any]:
    """Retorna um dicionário de modelos para avaliação."""

    num_pipe = compose.SelectType(float, int) | preprocessing.StandardScaler()
    cat_pipe = compose.SelectType(str) | preprocessing.OneHotEncoder()

    preprocessor = num_pipe + cat_pipe
    return {
        'HATR (baseline)': preprocessor | tree.HoeffdingAdaptiveTreeRegressor(),

        'HTR-QO-0.25 (baseline)': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.QOSplitter(
                allow_multiway_splits=True,
            ),
        ),

        'HTR-QO-0.5 (baseline)': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.QOSplitter(
                radius=0.5,
                allow_multiway_splits=True,
            ),
        ),

        'HTR-AQO-Triangular': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                kernel="triangular",
            ),
        ),

        'HTR-AQO-Epanechnikov': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                kernel="epanechnikov",
            ),
        ),

        'HTR-AQO-Smooth': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                kernel="smooth",
            ),
        ),
    }
