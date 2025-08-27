from river.datasets import synth

def get_synth_datasets():
    """
    Retorna um dicionário de 'fábricas' de dataset.
    Cada item é uma função que, quando chamada, retorna um novo stream.
    """
    return {
        "friedman_drift_local": lambda: synth.ConceptDriftStream(
            stream=synth.FriedmanDrift(seed=42),
            drift_stream=synth.FriedmanDrift(seed=42, drift_type='lea'),
            seed=42, position=25000, width=5000
        ),
        "hyperplane_drift": lambda: synth.ConceptDriftStream(
            stream=synth.Hyperplane(seed=42, n_features=10),
            drift_stream=synth.Hyperplane(seed=42, n_features=10, n_drift_features=5),
            seed=42, position=25000, width=5000
        ),
        # "friedman_drift_abrupt": lambda: synth.ConceptDriftStream(
        #     stream=synth.FriedmanDrift(seed=42),
        #     drift_stream=synth.FriedmanDrift(seed=42, drift_type='gra', position=(10_000, 30_000)),
        #     seed=42, width=5000
        # ),
        # "friedman_drift_abrupt-2": lambda: synth.ConceptDriftStream(
        #     stream=synth.FriedmanDrift(seed=1),
        #     drift_stream=synth.FriedmanDrift(seed=1, drift_type='gra', position=(15_000, 35_000)),
        #     seed=1, width=5000
        # ),
        # "friedman_drift_slow_abrupt": lambda: synth.ConceptDriftStream(
        #     stream=synth.FriedmanDrift(seed=42),
        #     drift_stream=synth.FriedmanDrift(seed=42, drift_type='gsg', position=(10_000, 30_000)),
        #     seed=42, width=5000
        # ),
    }