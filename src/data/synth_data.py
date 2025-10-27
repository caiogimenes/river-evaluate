from river.datasets import synth
import numpy as np

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
    }

def get_synth_abrupt_datasets(n_instances):
    """
    Retorna um dicionário de 'fábricas' de dataset com drift abrupto.
    Cada item é uma função que, quando chamada, retorna um novo stream.
    """
    np.seed = 42
    drift_positions = np.random.randint(low=0.2*n_instances, high=0.8*n_instances, size=(3,3))
    return {
        "friedman_drift_abrupt-GRA": lambda: synth.ConceptDriftStream(
            stream=synth.FriedmanDrift(seed=42),
            drift_stream=synth.FriedmanDrift(
                seed=42,
                drift_type='gra',
                position=sorted((drift_positions[0][0], drift_positions[0][1]))
            ),
            seed=42,
        ),
        "friedman_drift_abrupt-2-LEA": lambda: synth.ConceptDriftStream(
            stream=synth.FriedmanDrift(seed=90),
            drift_stream=synth.FriedmanDrift(
                seed=90,
                drift_type='lea',
                position=sorted(drift_positions[1,:])
            ),
            seed=42, width=2000
        ),
        "friedman_drift_abrupt-GSG": lambda: synth.ConceptDriftStream(
            stream=synth.FriedmanDrift(seed=90),
            drift_stream=synth.FriedmanDrift(
                seed=90,
                drift_type='gsg',
                position=sorted((drift_positions[2][0], drift_positions[2][1]))
            ),
            seed=42, width=2000
        ),
    }
