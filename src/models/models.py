from joblib.externals.loky.process_executor import MAX_DEPTH

from river import tree
from river import compose, preprocessing

def get_models():
    """Retorna um dicionário de modelos para avaliação."""

    num_pipe = compose.SelectType(float, int) | preprocessing.StandardScaler()
    cat_pipe = compose.SelectType(str) | preprocessing.OneHotEncoder()

    preprocessor = num_pipe + cat_pipe
    MAX_DEPTH = 50
    return {
        f'HATR (baseline)': preprocessor | tree.HoeffdingAdaptiveTreeRegressor(),

        f'HTR-QO-0.25 (baseline)': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.QOSplitter(
                allow_multiway_splits=True,
            ),
            max_depth=MAX_DEPTH,
        ),

        f'HTR-QO-0.5 (baseline)': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.QOSplitter(
                radius=0.5,
                allow_multiway_splits=True,
            ),
            max_depth=MAX_DEPTH
        ),

        f'HTR-AQO-Triangular': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                kernel="triangular",
            ),
            max_depth=MAX_DEPTH
        ),

        f'HTR-AQO-Epanechnikov': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                kernel="epanechnikov",
            ),
            max_depth=MAX_DEPTH
        ),

        f'HTR-AQO-Smooth': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                kernel="smooth",
            ),
            max_depth=MAX_DEPTH
        ),
    }