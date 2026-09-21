import numpy as np
from features.lowpass import lowpass
from features.cr_features import extract_cr, regularized_fisher_corr


def test_constant_patch():
    p = np.full((64, 64, 3), 128/255, np.float64)
    assert np.allclose(lowpass(p), p[2:62, 2:62], atol=1e-15)
    c, r = extract_cr(p)
    assert c.shape == (6,) and r.shape == (8,)
    assert np.allclose(c[3:], 0, atol=1e-12)
    assert np.allclose(r[:3], np.log(1e-4), atol=1e-12)
    assert np.allclose(r[3:], 0, atol=1e-12)


def test_regularized_corr():
    u, v = np.array([0., 1., 2.]), np.array([2., 1., 0.])
    expected = np.arctanh(-np.mean((u-u.mean())**2) / (np.mean((u-u.mean())**2) + 1e-8))
    expected = np.arctanh(np.clip(np.tanh(expected), -0.999, 0.999))
    assert np.isclose(regularized_fisher_corr(u, v), expected)
    assert regularized_fisher_corr(u, u) < np.inf
