from river import tree

def get_models():
    """Retorna um dicionário de modelos para avaliação."""
    return {
        'HTR-QO (Multi-way)': tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.QOSplitter(allow_multiway_splits=True),
        ),
        'HTR-F-QO (α=0.999)': tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.FadingQOSplitter(alpha=0.999, allow_multiway_splits=True),
        ),
        'HTR-F-QO (α=0.990)': tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.FadingQOSplitter(alpha=0.990, allow_multiway_splits=True),
        ),
        'HTR-F-QO (α=0.900)': tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.FadingQOSplitter(alpha=0.900, allow_multiway_splits=True),
        ),
        'HATR (Adaptive)': tree.HoeffdingAdaptiveTreeRegressor()
    }