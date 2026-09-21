"""CPU float64 valid convolution with the fixed binomial filter."""
import numpy as np

KERNEL_1D = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
KERNEL = np.outer(KERNEL_1D, KERNEL_1D)


def lowpass(patch: np.ndarray) -> np.ndarray:
    p = np.asarray(patch, dtype=np.float64)
    if p.shape != (64, 64, 3):
        raise ValueError("expected a 64x64x3 patch")
    windows = np.lib.stride_tricks.sliding_window_view(p, (5, 5), axis=(0, 1))
    return np.einsum("ijckl,kl->ijc", windows, KERNEL, optimize=True)
