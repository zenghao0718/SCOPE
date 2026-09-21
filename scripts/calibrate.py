import argparse
from collections import Counter
from pathlib import Path
from _common import load_config, read_json, write_json, write_csv, model_from_best
from scope_data.feature_cache import load_cache, feature_protocol_hash
from scope_data.manifests import read_manifest, sha256_file
from evaluation.inference import image_scores
from evaluation.calibration import threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scope_cr68_mdn3.yaml")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--standardizer", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, choices=[17, 42, 2026], required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config, standardizer = load_config(args.config), read_json(args.standardizer)
    run_dir = Path(args.run_dir) / f"seed_{args.seed}"
    best = run_dir / "checkpoints" / "best.pt"
    model = model_from_best(best, config, standardizer, args.seed, args.device)
    manifest = Path(args.manifest_dir) / "real_calibration.csv"
    c, r, meta = load_cache(Path(args.feature_dir) / "real_calibration",
                             expected_protocol_hash=feature_protocol_hash(config), expected_manifest_hash=sha256_file(manifest))
    rows = read_manifest(manifest)
    if Counter(row["source"] for row in rows) != config["data"]["real_calibration"] or any(row["label"] != "0" for row in rows):
        raise ValueError("formal real_calibration counts or labels mismatch")
    scores = image_scores(model, c, r, standardizer, args.device)
    if len(scores) != len(rows):
        raise ValueError("calibration cache/manifest length mismatch")
    tau = threshold(scores, config["calibration"]["quantile"])
    out = run_dir / "calibration"
    write_csv(out / "scores.csv", [dict(image_id=row["image_id"], score=score) for row, score in zip(rows, scores)])
    write_json(out / "calibration.json", dict(threshold=tau, quantile=config["calibration"]["quantile"],
               method="higher", num_images=len(scores), checkpoint_sha256=sha256_file(best),
               standardizer_sha256=sha256_file(args.standardizer), calibration_manifest_sha256=sha256_file(manifest),
               protocol_id=config["protocol"]["id"], seed=args.seed))
    print(tau)


if __name__ == "__main__":
    main()
