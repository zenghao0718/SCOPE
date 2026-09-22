import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from engine.config import load_config
from data.manifests import enrich, write_manifests
from data.sources import file_records
from data.feature_cache import build_cache, load_cache


def _config():
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "scope_cr68_mdn3.yaml")


def _make_rows(tmp_path, split, count):
    root = tmp_path / split
    root.mkdir()
    for i in range(count):
        rng = np.random.default_rng(i + hash(split) % 997)
        Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(root / f"{i:03d}.png")
    rows, errors = enrich(file_records(root, "imagenet" if split != "genimage_eval" else "genimage",
                                       label=int(split == "genimage_eval"),
                                       generator="synthetic" if split == "genimage_eval" else ""))
    assert not errors
    return [{**row, "split": split} for row in rows]


def _compare_caches(a_dir, b_dir):
    a_c, a_r, a_meta = load_cache(a_dir)
    b_c, b_r, b_meta = load_cache(b_dir)
    a_coords = np.load(a_dir / "patch_yx.npy")
    b_coords = np.load(b_dir / "patch_yx.npy")
    with (a_dir / "metadata.csv").open(encoding="utf-8") as stream:
        a_rows = list(csv.DictReader(stream))
    with (b_dir / "metadata.csv").open(encoding="utf-8") as stream:
        b_rows = list(csv.DictReader(stream))
    assert np.array_equal(a_c, b_c)
    assert np.array_equal(a_r, b_r)
    assert np.array_equal(a_coords, b_coords)
    assert a_rows == b_rows
    assert a_meta["feature_protocol_hash"] == b_meta["feature_protocol_hash"]
    return a_meta


def test_workers_one_builds_cache(tmp_path):
    config = _config()
    rows = _make_rows(tmp_path, "real_train", 3)
    manifest = tmp_path / "real_train.csv"
    write_manifests({"real_train": rows}, tmp_path, config["protocol"]["id"], config["data"]["candidate_sampling_seed"])
    out = tmp_path / "w1"
    meta = build_cache(rows, manifest, out, config, workers=1, split_name="real_train")
    assert meta["sample_count"] == 3
    assert load_cache(out)[0].shape == (3, 4, 6)


def test_serial_matches_parallel(tmp_path):
    config = _config()
    rows = _make_rows(tmp_path, "real_val", 8)
    manifest = tmp_path / "real_val.csv"
    write_manifests({"real_val": rows}, tmp_path, config["protocol"]["id"], config["data"]["candidate_sampling_seed"])
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    build_cache(rows, manifest, serial, config, overwrite=True, workers=1, split_name="real_val")
    build_cache(rows, manifest, parallel, config, overwrite=True, workers=4, split_name="real_val")
    _compare_caches(serial, parallel)


def test_output_order_matches_manifest(tmp_path):
    config = _config()
    rows = _make_rows(tmp_path, "real_calibration", 6)
    manifest = tmp_path / "real_calibration.csv"
    write_manifests({"real_calibration": rows}, tmp_path, config["protocol"]["id"], config["data"]["candidate_sampling_seed"])
    out = tmp_path / "ordered"
    build_cache(rows, manifest, out, config, overwrite=True, workers=4, split_name="real_calibration")
    with (out / "metadata.csv").open(encoding="utf-8") as stream:
        meta_rows = list(csv.DictReader(stream))
    assert [row["image_id"] for row in meta_rows] == [row["image_id"] for row in rows]


def test_formal_split_error_raises(tmp_path):
    config = _config()
    rows = _make_rows(tmp_path, "real_train", 2)
    rows[1] = {**rows[1], "path": str(tmp_path / "missing.png")}
    manifest = tmp_path / "real_train.csv"
    write_manifests({"real_train": rows}, tmp_path, config["protocol"]["id"], config["data"]["candidate_sampling_seed"])
    with pytest.raises(Exception):
        build_cache(rows, manifest, tmp_path / "fail", config, overwrite=True, workers=4, split_name="real_train")


def test_eval_split_records_errors(tmp_path):
    config = _config()
    rows = _make_rows(tmp_path, "genimage_eval", 3)
    rows[1] = {**rows[1], "path": str(tmp_path / "missing.png")}
    manifest = tmp_path / "genimage_eval.csv"
    write_manifests({"genimage_eval": rows}, tmp_path, config["protocol"]["id"], config["data"]["candidate_sampling_seed"])
    out = tmp_path / "genimage"
    meta = build_cache(rows, manifest, out, config, overwrite=True, workers=2, split_name="genimage_eval")
    assert meta["error_count"] == 1
    with (out / "metadata.csv").open(encoding="utf-8") as stream:
        statuses = [row["status"] for row in csv.DictReader(stream)]
    assert statuses.count("feature_error") == 1


def test_workers_must_be_positive(tmp_path):
    config = _config()
    rows = _make_rows(tmp_path, "real_val", 1)
    manifest = tmp_path / "real_val.csv"
    write_manifests({"real_val": rows}, tmp_path, config["protocol"]["id"], config["data"]["candidate_sampling_seed"])
    with pytest.raises(ValueError, match="workers must be >= 1"):
        build_cache(rows, manifest, tmp_path / "bad", config, workers=0, split_name="real_val")
