from river.datasets import synth
import numpy as np


def _choice(rng, *args, **kwargs):
    source = np.random if rng is None else rng
    return source.choice(*args, **kwargs)


def _randint(rng, low, high):
    if rng is None:
        return int(np.random.randint(low, high))
    return int(rng.integers(low, high))


def _shuffle(rng, values):
    if rng is None:
        np.random.shuffle(values)
    else:
        rng.shuffle(values)


def get_friedman_datasets(drift_type: str | list, n_datasets=15, n_instances=1_000_000, rng=None):
    seeds_pool = _choice(rng, 1000, size=n_datasets, replace=False)
    min_window = int(n_instances * 0.05)
    max_window = int(n_instances * 0.15)
    datasets = {}
    for i in range(n_datasets):
        drift = _choice(rng, drift_type) if isinstance(drift_type, list) else drift_type
        seed = int(seeds_pool[i])

        if drift == "lea":
            position = (int(n_instances * 0.25), int(n_instances * 0.5), int(n_instances * 0.75))
        else:
            position = (int(n_instances * 0.3), int(n_instances * 0.7))

        window = _randint(rng, min_window, max_window)

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


def get_hyperplane_datasets(n_datasets=15, rng=None):
    seeds = _choice(rng, 1000, size=n_datasets, replace=False)
    _shuffle(rng, seeds)
    datasets = {}
    for i in range(n_datasets):
        drift_feat = _randint(rng, 3, 6)
        mag_change = _randint(rng, 2, 5) / 10
        noise = _randint(rng, 2, 6) / 10
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


def get_rbf_datasets(n_datasets=15, rng=None):
    seeds = _choice(rng, 1000, size=n_datasets * 2, replace=False)
    _shuffle(rng, seeds)
    datasets = {}
    for i in range(n_datasets):
        n_classes = _randint(rng, 2, 3)
        n_features = 20
        n_centroids = 2 * n_features
        n_drift = n_centroids
        change_speed = _randint(rng, 1, 4) / 10
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
