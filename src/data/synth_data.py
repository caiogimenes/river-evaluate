from river.datasets import synth
import numpy as np


def get_friedman_datasets(drift_type: str | list, n_datasets=15, n_instances=1_000_000):
    seeds_pool = np.random.choice(1000, size=n_datasets, replace=False)
    min_window = int(n_instances * 0.05)
    max_window = int(n_instances * 0.15)
    datasets = {}
    for i in range(n_datasets):
        drift = np.random.choice(drift_type) if isinstance(drift_type, list) else drift_type
        seed = int(seeds_pool[i])

        if drift == "lea":
            position = (int(n_instances * 0.25), int(n_instances * 0.5), int(n_instances * 0.75))
        else:
            position = (int(n_instances * 0.3), int(n_instances * 0.7))

        window = np.random.randint(min_window, max_window)

        dset_name = f"""
        Friedman
        Drift = {drift.upper()}
        Seed = {seed}
        Transition Window = {window}
        Drift positions: {position}"""

        datasets[dset_name] = lambda s=seed, d=drift, p=position, w=window: synth.FriedmanDrift(
            seed=s,
            drift_type=d,
            position=p,
            transition_window=w
        )

    return datasets


def get_hyperplane_datasets(n_datasets=15):
    seeds = np.random.choice(1000, size=n_datasets, replace=False)
    np.random.shuffle(seeds)
    datasets = {}
    for i in range(n_datasets):
        drift_feat = np.random.randint(3,6)
        mag_change = np.random.randint(2, 5) / 10
        noise = np.random.randint(2, 6) / 10
        seed = int(seeds[i])

        d_set_name = f"""
        Hyperplane
        Seed = {seed}
        Drift Feat: {drift_feat}
        Magnitude: {mag_change}
        Noise: {noise}"""

        datasets[d_set_name] = lambda s=seed, d=drift_feat, m=mag_change, n=noise: synth.Hyperplane(
            seed=s,
            n_drift_features=d,
            mag_change=m,
            noise_percentage=n
        )

    return datasets


def get_rbf_datasets(n_datasets=15):
    seeds = np.random.choice(1000, size=n_datasets*2, replace=False)
    np.random.shuffle(seeds)
    datasets = {}
    for i in range(n_datasets):
        n_classes = np.random.randint(2,3)
        n_features = 20
        n_centroids = 2 * n_features
        n_drift = n_centroids
        change_speed = np.random.randint(1, 4) / 10
        seed_model = int(seeds[i])
        seed_sample = int(seeds[i + n_datasets])

        dset_name = f"""
        RandomRBF
        Seed Model = {seed_model}
        Seed Sample = {seed_sample}
        N of classes: {n_classes}
        Change Speed: {change_speed}
        N of features: {n_features}
        N of centroids: {n_centroids}
        N of drift centroids: {n_drift}
        """

        datasets[dset_name] = \
            (lambda sm=seed_model, sp=seed_sample, nc=n_classes, nf=n_features, cs=change_speed, nct=n_centroids, nd=n_drift:
                   synth.RandomRBFDrift(
                       seed_model=sm,
                       seed_sample=sp,
                       n_classes=nc,
                       n_features=nf,
                       n_centroids=nct,
                       change_speed=cs,
                       n_drift_centroids=nd,
                   )
            )

    return datasets