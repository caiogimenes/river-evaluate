from river.datasets import synth


def get_synth_datasets():
    """
    Retorna um dicionário de 'fábricas' de dataset.
    Cada item é uma função que, quando chamada, retorna um novo stream.
    """
    return {
        # "friedman_drift_local": lambda: synth.ConceptDriftStream(
        #     stream=synth.FriedmanDrift(seed=42),
        #     drift_stream=synth.FriedmanDrift(seed=42, drift_type='lea'),
        #     seed=42, position=25000, width=5000
        # ),
        # "hyperplane_drift": lambda: synth.ConceptDriftStream(
        #     stream=synth.Hyperplane(seed=42, n_features=10),
        #     drift_stream=synth.Hyperplane(seed=42, n_features=10, n_drift_features=5),
        #     seed=42, position=25000, width=5000
        # ),
        "friedman_drift_abrupt": lambda: synth.ConceptDriftStream(
            stream=synth.FriedmanDrift(seed=42),
            drift_stream=synth.FriedmanDrift(seed=42, drift_type='gra', position=(10_000, 30_000)),
            seed=42, width=5000
        ),
        "friedman_drift_abrupt-2": lambda: synth.ConceptDriftStream(
            stream=synth.FriedmanDrift(seed=30),
            drift_stream=synth.FriedmanDrift(seed=30, drift_type='gra', position=(20_000, 90_000)),
            seed=42, width=20_000
        ),
        "friedman_drift_abrupt-3": lambda: synth.ConceptDriftStream(
            stream=synth.FriedmanDrift(seed=90),
            drift_stream=synth.FriedmanDrift(seed=90, drift_type='gra', position=(15_000, 50_000)),
            seed=42, width=2000
        ),
        "friedman_drift_abrupt-4": lambda: synth.ConceptDriftStream(
            stream=synth.FriedmanDrift(seed=1),
            drift_stream=synth.FriedmanDrift(seed=1, drift_type='gra', position=(30_000, 45_000)),
            seed=42, width=10_000
        ),
        # "friedman_drift_slow_abrupt": lambda: synth.ConceptDriftStream(
        #     stream=synth.FriedmanDrift(seed=42),
        #     drift_stream=synth.FriedmanDrift(seed=42, drift_type='gsg', position=(10_000, 30_000)),
        #     seed=42, width=5000
        # ),
        # "sine_time_series": lambda: synth.ConceptDriftStream(
        #     stream=synth.Sine(classification_function = 2, seed = 112, balance_classes = False, has_noise = True),
        #     drift_stream=synth.Sine(classification_function = 3, seed = 42, balance_classes = False, has_noise = True),
        #     position=100_000,
        #     width=5_000,
        #     seed=42
        # )
    }
