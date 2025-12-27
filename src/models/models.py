from river import tree
from river import compose, preprocessing

def get_models():
    """Retorna um dicionário de modelos para avaliação."""

    num_pipe = compose.SelectType(float, int) | preprocessing.MinMaxScaler()
    cat_pipe = compose.SelectType(str) | preprocessing.OneHotEncoder()

    preprocessor = num_pipe + cat_pipe

    return {
        f'HATR (baseline)': preprocessor | tree.HoeffdingAdaptiveTreeRegressor(),

        f'HTR-QO-radius 0.25 (baseline)': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.QOSplitter(
                allow_multiway_splits=True,
            ),
        ),

        f'HTR-QO-radius 0.5 (baseline)': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.QOSplitter(
                radius=0.5,
                allow_multiway_splits=True,
            ),
        ),

        f'HTR-AQO-Triangular-0.5': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                radius=0.5,
                allow_multiway_splits=True,
                kernel="triangular"
            ),
        ),

        f'HTR-AQO-Epanechnikov-0.5': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                radius=0.5,
                allow_multiway_splits=True,
                kernel="epanechnikov",
            ),
        ),

        f'HTR-AQO-Smooth-0.5': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                radius=0.5,
                allow_multiway_splits=True,
                kernel="smooth",
            ),
        ),
    }