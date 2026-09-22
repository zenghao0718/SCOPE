"""Synthetic end-to-end check; no formal data or formal training."""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from engine.config import load_config
from engine.trainer import train
from engine.checkpoint import load_checkpoint
from data.sources import file_records
from data.manifests import enrich, build_splits, write_manifests
from data.feature_cache import build_cache, load_cache
from features.standardization import fit
from evaluation.inference import image_scores
from evaluation.calibration import threshold
from evaluation.metrics import binary_metrics
from models.mdn import MDN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate as calibration_script
import _evaluate as evaluation_script


def test_pipeline(tmp_path, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "scope_cr68_mdn3.yaml")
    config["train"]["batch_size_images"] = 2
    config["train"]["max_epochs"] = 1
    roots = {}
    sources = {"real_train": "imagenet", "real_val": "imagenet", "real_calibration": "imagenet",
               "real_external_coco": "coco", "genimage_eval": "genimage"}
    for split, count in (("real_train", 3), ("real_val", 2), ("real_calibration", 2),
                         ("real_external_coco", 2), ("genimage_eval", 2)):
        root = tmp_path / split
        root.mkdir()
        for i in range(count):
            rng = np.random.default_rng(i + len(split))
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(root / f"{i}.png")
        rows, errors = enrich(file_records(root, sources[split], label=int(split == "genimage_eval"),
                                           generator="synthetic" if split == "genimage_eval" else ""))
        assert not errors
        if split == "genimage_eval":
            rows[0]["label"] = 0
        roots[split] = [{**row, "split": split} for row in rows]
    manifest_dir = tmp_path / "manifests"
    write_manifests(roots, manifest_dir, config["protocol"]["id"], config["data"]["candidate_sampling_seed"])
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
    train(*arrays["real_train"][:2], *arrays["real_val"][:2], standardizer, config, 17, run_dir, tensorboard=True)
    events = list((run_dir / "tensorboard").glob("events.out.tfevents.*"))
    assert events
    accumulator = EventAccumulator(str(run_dir / "tensorboard"))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert {"Loss/Train_NLL", "Loss/Val_NLL", "Optimization/Learning_Rate",
            "Training/Best_Val_NLL", "MDN/Sigma_Median", "MDN/Mixture_Weight_3"} <= tags
    train_log = next((run_dir / "logs").glob("*_train.log"))
    assert "Start training" in train_log.read_text(encoding="utf-8")
    assert "Training completed" in train_log.read_text(encoding="utf-8")
    best = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    model = MDN()
    model.load_state_dict(best["model_state"])
    calibration = image_scores(model, *arrays["real_calibration"][:2], standardizer)
    tau = threshold(calibration)
    scores = image_scores(model, *arrays["genimage_eval"][:2], standardizer)
    assert np.isfinite(scores).all() and np.isfinite(tau)
    result = binary_metrics([0, 1], [calibration[0], scores[0]], tau)
    assert result["auroc"] is not None
    standardizer_path = tmp_path / "standardizer.json"
    standardizer_path.write_text(json.dumps(standardizer), encoding="utf-8")
    config["data"]["real_calibration"] = {"imagenet": 2}
    config["data"]["real_external_eval"] = {"coco": 2}
    from types import SimpleNamespace
    args = SimpleNamespace(config="synthetic", feature_dir=str(feature_dir), manifest_dir=str(manifest_dir),
                           standardizer=str(standardizer_path), run_dir=str(tmp_path), seed=17, device="cpu")
    seed_dir = tmp_path / "seed_17"
    (seed_dir / "checkpoints").mkdir(parents=True)
    (seed_dir / "checkpoints" / "best.pt").write_bytes((run_dir / "checkpoints" / "best.pt").read_bytes())
    monkeypatch.setattr(calibration_script, "load_config", lambda _: config)
    monkeypatch.setattr(evaluation_script, "load_config", lambda _: config)
    calibration_script.calibrate(args)
    evaluation_script.evaluate(args, "real_external_coco")
    evaluation_script.evaluate(args, "genimage_eval")
    for phase, phrase in (("calibration", "Calibration finished"), ("coco_eval", "COCO |"),
                          ("genimage_eval", "[GenImage] synthetic")):
        logfile = next((seed_dir / "logs").glob(f"*_{phase}.log"))
        assert phrase in logfile.read_text(encoding="utf-8")


def test_sampling_manifest_smoke(tmp_path):
    def make_source(name, values):
        root = tmp_path / name
        root.mkdir()
        for index, value in enumerate(values):
            Image.fromarray(np.full((64, 64, 3), value, np.uint8)).save(root / f"{index}.png")
        return root

    gen_root = make_source("genimage", [5, 6])
    coco_root = make_source("coco", [5, 20, 21, 22, 23])
    im_root = make_source("imagenet", [5, 30, 31, 32, 33, 34, 35, 36])
    ls_root = make_source("lsun", [30, 40, 41, 42, 43, 44, 45, 46])
    gen, errors = enrich(file_records(gen_root, "genimage", label=1, generator="synthetic"))
    assert not errors and len(gen) == 2 and all(r["content_id"] for r in gen)
    splits, stats = build_splits(file_records(im_root, "imagenet"), file_records(ls_root, "lsun"),
                                 file_records(coco_root, "coco"), gen,
                                 counts={"imagenet": (2, 1, 1), "lsun": (2, 1, 1), "coco": 2})
    assert [len(splits[name]) for name in ("real_train", "real_val", "real_calibration", "real_external_coco", "genimage_eval")] == [4, 2, 2, 2, 2]
    real = sum((splits[name] for name in ("real_train", "real_val", "real_calibration", "real_external_coco")), [])
    assert len({r["content_id"] for r in real}) == len(real)
    assert not {r["content_id"] for r in real} & {r["content_id"] for r in gen}
    assert stats["coco"]["processed_candidates"] < stats["coco"]["total_candidates"]
    output = tmp_path / "manifests"
    meta = write_manifests(splits, output, "SCOPE_CR68_MDN3_v1.0", 20260917, stats)
    assert meta["candidate_sampling_seed"] == 20260917
    assert meta["stats"]["imagenet"]["selected_candidates"] == 4
    assert {"total_candidates", "processed_candidates", "selected_candidates", "decode_errors",
            "duplicate_rejections", "overlap_rejections"} <= set(meta["stats"]["imagenet"])
