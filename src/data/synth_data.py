from river.datasets import synth
import numpy as np

def get_friedman_datasets(drift: str, n_datasets=15):
    seeds = np.random.choice(100, size=n_datasets, replace=False)
    np.random.shuffle(seeds)
    window = 1e5
    datasets = {}
    for i in range(n_datasets):
        if drift == "lea":
            position = (2e5, 5e5, 7e5)
        else:
            position = (3e5, 7e5)
        seed = int(seeds[i])

        datasets[f"""
        Friedman
        Drift = {drift.upper()}
        Seed = {seed}"""] = lambda s=seed, d=drift, p=position, w=window: synth.FriedmanDrift(
            seed=s,
            drift_type=d,
            position=p,
            transition_window=w
        ),

    return datasets

def get_hyperplane_datasets(n_datasets=15):
    seeds = np.random.choice(100, size=n_datasets, replace=False)
    np.random.shuffle(seeds)
    datasets = {}
    for i in range(n_datasets):
        drift_feat = np.random.choice(np.arange(1,10))
        mag_change = np.random.random()
        noise = np.random.random()
        seed = int(seeds[i])

        datasets[f"""
        Hyperplane
        Seed = {seed}
        Drift Feat: {drift_feat}
        Magnitude: {mag_change}
        Noise: {noise}"""] = lambda s=seed, d=drift_feat, m=mag_change, n=noise: synth.Hyperplane(
            seed=s,
            n_drift_features=d,
            mag_change=m,
            noise_percentage=n
        ),

    return datasets
