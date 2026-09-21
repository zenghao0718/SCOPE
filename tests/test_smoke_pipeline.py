"""Synthetic end-to-end check; no formal data or formal training."""
from pathlib import Path
import numpy as np
from PIL import Image

from engine.config import load_config
from engine.trainer import train
from engine.checkpoint import load_checkpoint
from scope_data.sources import file_records
from scope_data.manifests import enrich, write_manifests
from scope_data.feature_cache import build_cache, load_cache
from features.standardization import fit
from evaluation.inference import image_scores
from evaluation.calibration import threshold
from evaluation.metrics import binary_metrics
from models.mdn import MDN


def test_pipeline(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "scope_cr68_mdn3.yaml")
    config["train"]["batch_size_images"] = 2
    config["train"]["max_epochs"] = 1
    roots = {}
    for split, count in (("real_train", 3), ("real_val", 2), ("real_calibration", 2), ("genimage_eval", 2)):
        root = tmp_path / split
        root.mkdir()
        for i in range(count):
            rng = np.random.default_rng(i + len(split))
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(root / f"{i}.png")
        rows, errors = enrich(file_records(root, split, label=int(split == "genimage_eval")))
        assert not errors
        roots[split] = [{**row, "split": split} for row in rows]
    manifest_dir = tmp_path / "manifests"
    write_manifests(roots, manifest_dir, config["protocol"]["id"], config["data"]["split_salt"])
    feature_dir = tmp_path / "features"
    arrays = {}
    for split in roots:
        manifest = manifest_dir / f"{split}.csv"
        meta = build_cache(roots[split], manifest, feature_dir / split, config)
        arrays[split] = load_cache(feature_dir / split)
        assert arrays[split][0].shape == (len(roots[split]), 4, 6)
    standardizer = fit(*arrays["real_train"][:2], protocol_id=config["protocol"]["id"],
                       feature_protocol_hash=arrays["real_train"][2]["feature_protocol_hash"],
                       train_manifest_sha256=arrays["real_train"][2]["manifest_sha256"], train_cache_sha256="synthetic")
    run_dir = tmp_path / "run"
    train(*arrays["real_train"][:2], *arrays["real_val"][:2], standardizer, config, 17, run_dir)
    best = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    model = MDN()
    model.load_state_dict(best["model_state"])
    calibration = image_scores(model, *arrays["real_calibration"][:2], standardizer)
    tau = threshold(calibration)
    scores = image_scores(model, *arrays["genimage_eval"][:2], standardizer)
    assert np.isfinite(scores).all() and np.isfinite(tau)
    result = binary_metrics([0, 1], [calibration[0], scores[0]], tau)
    assert result["auroc"] is not None
