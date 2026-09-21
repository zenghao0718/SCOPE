import numpy as np
from features.standardization import fit, transform


def test_population_and_constant():
    c = np.zeros((2, 4, 6)); r = np.zeros((2, 4, 8))
    c[1, :, 0] = 2
    artifact = fit(c, r, protocol_id="p", feature_protocol_hash="h", train_manifest_sha256="m", train_cache_sha256="c")
    assert artifact["c_mean"][0] == 1 and artifact["c_std"][0] == 1
    assert artifact["r_near_constant"][0] and artifact["r_scale"][0] == 1
    z, _ = transform(c + 10, r, artifact)
    assert z[0, 0, 0] == 9
