import scipy.stats
import numpy as np

def eval_significance(value, n_models, n_datasets, alpha):
    dfn = n_models - 1
    dfd = dfn * (n_datasets - 1)
    critical_value = scipy.stats.f.ppf(q=1-alpha, dfn=dfn, dfd=dfd)
    if value > critical_value:
        return True

    return False

def friedman_statistics(avg_rank, N: int):
    """
    Calculates the Friedman statistic.
    :param avg_rank: list or array of average ranks for each algorithm
    :param N: number of datasets (rows)
    """
    avg_rank = np.array(avg_rank)
    k = len(avg_rank)

    # Calculate Friedman statistic
    chi_sq = (12 * N / (k * (k + 1))) * (np.sum(avg_rank ** 2) - (k * (k + 1) ** 2) / 4)
    return chi_sq


def iman_davenport(chi_sq, N: int, k: int):
    """
    Calculates the Iman-Davenport statistic.
    :param chi_sq: The result from the Friedman statistic
    :param N: number of datasets
    :param k: number of algorithms (columns)
    """
    # Prevent division by zero if predictions are identical (denominator becomes 0)
    denominator = (N * (k - 1) - chi_sq)
    if denominator == 0:
        return np.inf

    return (N - 1) * chi_sq / denominator


def critical_difference_threshold(n_models: int, n_datasets: int, significance: float = 0.05):
    """
    Based on Table 5(a) for Nemenyi post-hoc test
    DEMSAR
    """
    critical_values_005 = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    }
    critical_values_010 = {
        2: 1.645,
        3: 2.052,
        4: 2.291,
        5: 2.459,
        6: 2.589,
        7: 2.693,
        8: 2.780,
        9: 2.855,
        10: 2.920,
    }
    cd = 0
    if significance == 0.05:
        cd = critical_values_005[n_models] * np.sqrt(n_models * (n_models+1) / (6 * n_datasets))
    elif significance == 0.1:
        cd = critical_values_010[n_models] * np.sqrt(n_models * (n_models+1) / (6 * n_datasets))

    return round(cd, 2)