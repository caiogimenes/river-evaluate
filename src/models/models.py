from river import tree

def get_models():
    """Retorna um dicionário de modelos para avaliação."""
    return {
        f'HATR (baseline)': tree.HoeffdingAdaptiveTreeRegressor(),
        f'HTR-QO (baseline)': tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.QOSplitter(
                allow_multiway_splits=True,
            ),
        ),
        f'HTR-AQO-Triangular': tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                allow_multiway_splits=True,
                kernel="triangular"
            ),
        ),
        f'HTR-AQO-Epanechnikov': tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                allow_multiway_splits=True,
                kernel="epanechnikov"
            ),
        ),
        f'HTR-AQO-Smooth': tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.AdaptiveQOSplitter(
                allow_multiway_splits=True,
                kernel="smooth"
            ),
        ),
    }