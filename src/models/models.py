from river import tree

def get_models():
    """Retorna um dicionário de modelos para avaliação."""
    return {
        'HTR (baseline)': tree.HoeffdingTreeRegressor(),
        'HTR-QO (baseline)': tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.QOSplitter(),
        ),
        'HATR (baseline)': tree.HoeffdingAdaptiveTreeRegressor(),
        'HTR-F-QO (α=0.9999, 1e-2)': tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.FadingQOSplitter(
                alpha=0.9999
            ),
            delta=1e-2
        ),
        'HTR-F-QO (α=0.9999, 1e-3)': tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.FadingQOSplitter(
                alpha=0.9999
            ),
            delta=1e-3
        ),
        'HTR-F-QO (α=0.999, 1e-2)': tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.FadingQOSplitter(
                alpha=0.999
            ),
            delta=1e-2
        ),
    }