import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
from _common import load_config, read_json, write_json, write_csv, model_from_best
from data.feature_cache import load_cache, feature_protocol_hash
from data.manifests import read_manifest, sha256_file
from evaluation.inference import image_scores
from evaluation.calibration import threshold
from engine.logging import setup_logger


def calibrate(args):
    start = time.monotonic()
    run_dir = Path(args.run_dir) / f"seed_{args.seed}"
    logger, _ = setup_logger(run_dir, "calibration", seed=args.seed)
    logger.info("Start calibration | seed=%s | device=%s", args.seed, args.device)
    config, standardizer = load_config(args.config), read_json(args.standardizer)
    best = run_dir / "checkpoints" / "best.pt"
    manifest = Path(args.manifest_dir) / "real_calibration.csv"
    logger.info("best_checkpoint=%s | checkpoint_sha256=%s", best, sha256_file(best))
    logger.info("standardizer=%s | standardizer_sha256=%s", args.standardizer, sha256_file(args.standardizer))
    logger.info("calibration_manifest=%s | manifest_sha256=%s", manifest, sha256_file(manifest))
    model = model_from_best(best, config, standardizer, args.seed, args.device)
    c, r, meta = load_cache(Path(args.feature_dir) / "real_calibration",
                             expected_protocol_hash=feature_protocol_hash(config), expected_manifest_hash=sha256_file(manifest))
    rows = read_manifest(manifest)
    if Counter(row["source"] for row in rows) != config["data"]["real_calibration"] or any(row["label"] != "0" for row in rows):
        raise ValueError("formal real_calibration counts or labels mismatch")
    scores = image_scores(model, c, r, standardizer, args.device)
    if len(scores) != len(rows):
        raise ValueError("calibration cache/manifest length mismatch")
    tau = threshold(scores, config["calibration"]["quantile"])
    logger.info("calibration_images=%d | score_mean=%.9g | score_std=%.9g | score_median=%.9g",
                len(scores), float(np.mean(scores)), float(np.std(scores, ddof=0)), float(np.median(scores)))
    logger.info("score_95_quantile_higher=%.9g | final_tau=%.9g", tau, tau)
    out = run_dir / "calibration"
    write_csv(out / "scores.csv", [dict(image_id=row["image_id"], score=score) for row, score in zip(rows, scores)])
    write_json(out / "calibration.json", dict(threshold=tau, quantile=config["calibration"]["quantile"],
               method="higher", num_images=len(scores), checkpoint_sha256=sha256_file(best),
               standardizer_sha256=sha256_file(args.standardizer), calibration_manifest_sha256=sha256_file(manifest),
               protocol_id=config["protocol"]["id"], seed=args.seed))
    logger.info("Calibration finished | seed=%s | tau=%.9g | scores=%s | metadata=%s | elapsed=%.1fs",
                args.seed, tau, out / "scores.csv", out / "calibration.json", time.monotonic()-start)
    return tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scope_cr68_mdn3.yaml")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--standardizer", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, choices=[17, 42, 2026], required=True)
    parser.add_argument("--device", default="cpu")
    calibrate(parser.parse_args())


if __name__ == "__main__":
    main()
