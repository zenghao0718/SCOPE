import csv
from pathlib import Path
from _common import load_config, read_json, write_json, write_csv, model_from_best
from scope_data.feature_cache import load_cache, feature_protocol_hash
from scope_data.manifests import read_manifest, sha256_file
from evaluation.inference import image_scores
from evaluation.metrics import genimage_metrics, coco_metrics


def evaluate(args, split):
    config, standardizer = load_config(args.config), read_json(args.standardizer)
    run_dir = Path(args.run_dir) / f"seed_{args.seed}"
    best = run_dir / "checkpoints" / "best.pt"
    calibration = read_json(run_dir / "calibration" / "calibration.json")
    if calibration["checkpoint_sha256"] != sha256_file(best) or calibration["standardizer_sha256"] != sha256_file(args.standardizer):
        raise ValueError("calibration provenance mismatch")
    model = model_from_best(best, config, standardizer, args.seed, args.device)
    manifest = Path(args.manifest_dir) / f"{split}.csv"
    rows = read_manifest(manifest)
    if split == "real_external_coco" and (len(rows) != config["data"]["real_external_eval"]["coco"]
                                           or any(row["source"] != "coco" or row["label"] != "0" for row in rows)):
        raise ValueError("formal COCO external manifest mismatch")
    c, r, meta = load_cache(Path(args.feature_dir) / split, expected_protocol_hash=feature_protocol_hash(config),
                             expected_manifest_hash=sha256_file(manifest))
    with (Path(args.feature_dir) / split / "metadata.csv").open(newline="", encoding="utf-8") as stream:
        metadata = list(csv.DictReader(stream))
    if not (len(rows) == len(c) == len(metadata)):
        raise ValueError("evaluation length mismatch")
    good = [i for i, item in enumerate(metadata) if item["status"] == "ok"]
    scores = image_scores(model, c[good], r[good], standardizer, args.device) if good else []
    score_by_index = dict(zip(good, scores))
    output = []
    for index, (row, meta_row) in enumerate(zip(rows, metadata)):
        score = score_by_index.get(index)
        output.append(dict(image_id=row["image_id"], label=row["label"], generator=row["generator"],
                           score="" if score is None else float(score),
                           prediction="" if score is None else int(score > calibration["threshold"]),
                           upscaled=meta_row["upscaled"], status=meta_row["status"],
                           error_type=meta_row["error_type"], error_message=meta_row["error_message"]))
    result = genimage_metrics(output, calibration["threshold"]) if split == "genimage_eval" else coco_metrics(output, calibration["threshold"])
    dest = run_dir / "evaluation" / split
    write_csv(dest / "scores.csv", output)
    write_json(dest / "metrics.json", result)
    print(result)
