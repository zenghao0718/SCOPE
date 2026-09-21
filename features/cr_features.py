"""Exact six condition and eight residual statistics."""
import numpy as np
from features.lowpass import lowpass

DELTA = 1e-4
CLIP = 0.999


def regularized_fisher_corr(u, v, delta=DELTA, clip=CLIP) -> float:
    u = np.asarray(u, dtype=np.float64).ravel(order="C")
    v = np.asarray(v, dtype=np.float64).ravel(order="C")
    if u.shape != v.shape or u.size == 0:
        raise ValueError("correlation vectors must have equal positive length")
    a, b = u - u.mean(), v - v.mean()
    rho = np.mean(a * b) / np.sqrt((np.mean(a * a) + delta**2) * (np.mean(b * b) + delta**2))
    return float(np.arctanh(np.clip(rho, -clip, clip)))


def extract_cr(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(patch, dtype=np.float64)
    if p.shape != (64, 64, 3):
        raise ValueError("expected a 64x64x3 patch")
    l60 = lowpass(p)
    inner = p[3:61, 3:61, :]
    lin = l60[1:59, 1:59, :]
    e = inner - lin
    y60 = np.einsum("ijc,c->ij", l60, [0.299, 0.587, 0.114])
    y = y60[1:59, 1:59]
    gx = (y60[1:59, 2:60] - y60[1:59, 0:58]) / 2
    gy = (y60[2:60, 1:59] - y60[0:58, 1:59]) / 2
    clipped = (inner.min(axis=2) <= 1/255) | (inner.max(axis=2) >= 254/255)
    c = np.array([*lin.mean(axis=(0, 1)), np.sqrt(np.mean((y-y.mean())**2)),
                  np.sqrt(np.mean(gx*gx + gy*gy)), clipped.mean()], dtype=np.float64)
    variance = np.mean((e - e.mean(axis=(0, 1)))**2, axis=(0, 1))
    ey = np.einsum("ijc,c->ij", e, [0.299, 0.587, 0.114])
    r = np.array([*np.log(np.sqrt(variance + DELTA**2)),
                  regularized_fisher_corr(ey[:, :57], ey[:, 1:58]),
                  regularized_fisher_corr(ey[:57, :], ey[1:58, :]),
                  regularized_fisher_corr(e[:, :, 0], e[:, :, 1]),
                  regularized_fisher_corr(e[:, :, 0], e[:, :, 2]),
                  regularized_fisher_corr(e[:, :, 1], e[:, :, 2])], dtype=np.float64)
    if not (np.isfinite(c).all() and np.isfinite(r).all()):
        raise FloatingPointError("nonfinite C/R feature")
    return c, r
