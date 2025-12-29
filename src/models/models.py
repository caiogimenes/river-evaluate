from river import tree, compose, preprocessing

def get_models():
    """Retorna um dicionário de modelos para avaliação."""

    num_pipe = compose.SelectType(float, int) | preprocessing.StandardScaler()
    cat_pipe = compose.SelectType(str) | preprocessing.OneHotEncoder()

    preprocessor = num_pipe + cat_pipe
    return {
        f'HATR (baseline)': preprocessor | tree.HoeffdingAdaptiveTreeRegressor(),

        f'HTR-QO-0.25 (baseline)': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.QOSplitter(
                allow_multiway_splits=True,
            ),
        ),

        f'HTR-QO-0.5 (baseline)': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.QOSplitter(
                radius=0.5,
                allow_multiway_splits=True,
            ),
        ),

        f'HTR-AQO-Triangular': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                kernel="triangular",
            ),
        ),

        f'HTR-AQO-Epanechnikov': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                kernel="epanechnikov",
            ),
        ),

        f'HTR-AQO-Smooth': preprocessor | tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                kernel="smooth",
            ),
        ),
    }