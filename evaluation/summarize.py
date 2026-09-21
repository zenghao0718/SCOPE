import numpy as np


def summarize_seed_values(per_seed):
    if set(map(int, per_seed)) != {17, 42, 2026}:
        raise ValueError("exactly seeds 17, 42, 2026 required")
    keys = set.intersection(*(set(values) for values in per_seed.values()))
    result = {}
    for key in sorted(keys):
        values = [per_seed[seed][key] for seed in (17, 42, 2026)]
        if all(isinstance(v, (int, float)) and v is not None for v in values):
            result[key] = dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)))
    return result
