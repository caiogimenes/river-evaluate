from river.datasets import synth
import numpy as np

def get_friedman_datasets(drift_type: str | list, n_datasets=15, n_instances=1000):
    seeds_pool = np.random.choice(100, size=n_datasets, replace=False)
    np.random.shuffle(seeds_pool)
    windows_pool = np.arange(100_000, 200_000, step=20_000)
    position_pool = np.arange(n_instances/10, n_instances * 4/5, step=n_instances/4)
    datasets = {}
    for i in range(n_datasets):
        if isinstance(drift_type, list):
            drift = np.random.choice(drift_type)
        else:
            drift = drift_type

        if drift == "lea":
            position = sorted(np.random.choice(position_pool,size=3, replace=False))
        else:
            position = sorted(np.random.choice(position_pool,size=2, replace=False))

        seed = int(seeds_pool[i])
        window = np.random.choice(windows_pool)

        datasets[f"""
        Friedman
        Drift = {drift.upper()}
        Seed = {seed}
        Transition Window = {window}
        Drift positions: {position}"""] = lambda s=seed, d=drift, p=position, w=window: synth.FriedmanDrift(
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

def get_sea_datasets(n_datasets=15):
    seeds = np.random.choice(100, size=n_datasets, replace=False)
    np.random.shuffle(seeds)
    datasets = {}
    for i in range(n_datasets):
        variant = np.random.choice(np.arange(0,2))
        noise = np.random.random()
        seed = int(seeds[i])
        datasets[f"""
        SEA
        Seed = {seed}
        Variant: {variant}
        Noise: {noise}"""] = lambda s=seed, v=variant, n=noise: synth.SEA(
            seed=s,
            variant=v,
            noise=n
        ),

    return datasets