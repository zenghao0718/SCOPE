"""Training-only population statistics."""
import hashlib
import json
import numpy as np


def fit(c_raw, r_raw, *, protocol_id, feature_protocol_hash, train_manifest_sha256, train_cache_sha256):
    c, r = np.asarray(c_raw, np.float64), np.asarray(r_raw, np.float64)
    if c.ndim != 3 or c.shape[1:] != (4, 6) or r.shape != (c.shape[0], 4, 8):
        raise ValueError("expected aligned [N,4,6] and [N,4,8]")
    if c.shape[0] == 0 or not (np.isfinite(c).all() and np.isfinite(r).all()):
        raise ValueError("empty or nonfinite training features")
    result = dict(protocol_id=protocol_id, feature_protocol_hash=feature_protocol_hash,
                  train_manifest_sha256=train_manifest_sha256, train_cache_sha256=train_cache_sha256)
    for name, x in (("c", c), ("r", r)):
        flat = x.reshape(-1, x.shape[-1])
        mean, std = flat.mean(axis=0), flat.std(axis=0, ddof=0)
        near = std < 1e-6
        result.update({f"{name}_mean": mean.tolist(), f"{name}_std": std.tolist(),
                       f"{name}_scale": np.where(near, 1.0, std).tolist(),
                       f"{name}_near_constant": near.tolist()})
    result["artifact_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def transform(c_raw, r_raw, artifact):
    c = (np.asarray(c_raw, np.float64) - artifact["c_mean"]) / artifact["c_scale"]
    r = (np.asarray(r_raw, np.float64) - artifact["r_mean"]) / artifact["r_scale"]
    if not (np.isfinite(c).all() and np.isfinite(r).all()):
        raise FloatingPointError("nonfinite standardized features")
    return c.astype(np.float32), r.astype(np.float32)
