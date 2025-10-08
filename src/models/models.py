from river import tree

def get_models():
    """Retorna um dicionário de modelos para avaliação."""
    return {
        'HTR': tree.HoeffdingTreeRegressor(),
        # 'HTR-QO-binary': tree.HoeffdingTreeRegressor(
        #     splitter=tree.splitter.QOSplitter()
        # ),
        # 'HTR-QO-multiway': tree.HoeffdingTreeRegressor(
        #     splitter=tree.splitter.QOSplitter(
        #         allow_multiway_splits=True
        #     )
        # ),
        # 'HTR-F-QO (α=0.9)': tree.HoeffdingTreeRegressor(
        #     splitter=tree.splitter.FadingQOSplitter(
        #         alpha=0.9
        #     ),
        # ),
    }