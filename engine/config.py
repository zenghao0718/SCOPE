"""Reject silent formal-protocol drift."""
from pathlib import Path
import yaml


def load_config(path, *, formal=True):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    checks = {
        ("protocol", "id"): "SCOPE_CR68_MDN3_v1.0",
        ("image", "patch_size"): 64, ("image", "patches_per_image"): 4,
        ("image", "min_short_side"): 64, ("image", "resize_mode"): "bilinear",
        ("feature", "c_dim"): 6, ("feature", "r_dim"): 8,
        ("feature", "lowpass_1d"): [1, 4, 6, 4, 1],
        ("feature", "lowpass_divisor"): 16.0,
        ("feature", "delta"): 1e-4, ("feature", "correlation_clip"): 0.999,
        ("model", "sigma_floor"): 0.05, ("model", "hidden_dims"): [64, 64],
        ("train", "batch_size_images"): 256, ("train", "max_epochs"): 50,
        ("train", "amp"): False, ("train", "tf32"): False,
        ("calibration", "method"): "higher", ("seeds",): [17, 42, 2026],
        ("calibration", "quantile"): 0.95, ("data", "split_salt"): "20260917",
        ("data", "real_train", "imagenet"): 5000, ("data", "real_train", "lsun"): 5000,
        ("data", "real_val", "imagenet"): 1000, ("data", "real_val", "lsun"): 1000,
        ("data", "real_calibration", "imagenet"): 1000, ("data", "real_calibration", "lsun"): 1000,
        ("data", "real_external_eval", "coco"): 2000,
    }
    for keys, expected in checks.items():
        current = config
        for key in keys:
            current = current[key]
        if current != expected and formal:
            raise ValueError(f"formal configuration changed at {'.'.join(keys)}: {current!r}")
    if config["model"]["num_components"] < 1 or config["model"]["sigma_floor"] <= 0:
        raise ValueError("invalid MDN configuration")
    if formal and config["model"]["num_components"] != 3:
        raise ValueError("formal K must be 3")
    return config
