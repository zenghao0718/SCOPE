import numpy as np


def threshold(real_scores, quantile=0.95):
    x = np.asarray(real_scores, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("calibration requires finite real image scores")
    return float(np.quantile(x, quantile, method="higher"))


def predict(score, tau):
    return int(score > tau)
