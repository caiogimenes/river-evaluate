from river.datasets import synth
import numpy as np

def get_friedman_datasets(n_datasets=10):
    drifts = ["lea", "gsg", "gra"]
    seeds = np.random.choice(100, size=n_datasets, replace=False)
    np.random.shuffle(seeds)
    window = 10_000
    datasets = {}
    for i in range(n_datasets):
        drift = np.random.choice(drifts)
        if drift == "lea":
            position = (200_000, 500_000, 700_000)
        else:
            position = (300_000, 700_000)
        seed = int(seeds[i])
        datasets[f"""
        Friedman
        Drift = {drift.upper()}
        Seed = {seed}"""] = lambda s=seed, d=drift, p=position, w=window: synth.FriedmanDrift(
            seed=s,
            drift_type=d,
            position=p,
            transition_window=window
        ),

    return datasets

def get_hyperplane_datasets(n_datasets=10):
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
