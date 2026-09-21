import csv
from pathlib import Path
import time
from _common import load_config, read_json, write_json, write_csv, model_from_best
from data.feature_cache import load_cache, feature_protocol_hash
from data.manifests import read_manifest, sha256_file
from evaluation.inference import image_scores
from evaluation.metrics import genimage_metrics, coco_metrics
from engine.logging import setup_logger


def _metric(value):
    return "N/A" if value is None else f"{value:.6g}"


def evaluate(args, split):
    start = time.monotonic()
    run_dir = Path(args.run_dir) / f"seed_{args.seed}"
    phase = "genimage_eval" if split == "genimage_eval" else "coco_eval"
    logger, _ = setup_logger(run_dir, phase, seed=args.seed)
    logger.info("Start %s | seed=%s | device=%s", phase, args.seed, args.device)
    config, standardizer = load_config(args.config), read_json(args.standardizer)
    best = run_dir / "checkpoints" / "best.pt"
    calibration = read_json(run_dir / "calibration" / "calibration.json")
    if calibration["checkpoint_sha256"] != sha256_file(best) or calibration["standardizer_sha256"] != sha256_file(args.standardizer):
        raise ValueError("calibration provenance mismatch")
    model = model_from_best(best, config, standardizer, args.seed, args.device)
    manifest = Path(args.manifest_dir) / f"{split}.csv"
    rows = read_manifest(manifest)
    logger.info("checkpoint=%s | threshold=%.9g | benchmark_images=%d",
                best, calibration["threshold"], len(rows))
    if split == "genimage_eval":
        logger.info("generator_count=%d", len({row["generator"] for row in rows}))
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
    decode_errors = sum(row["status"] in ("decode_error", "unsupported_format") for row in output)
    if split == "genimage_eval":
        for generator, metrics in result["generators"].items():
            logger.info("[GenImage] %s | n_real=%d | n_ai=%d | AUROC=%s | AP=%s | FPR=%s | TPR=%s | BACC=%s | coverage=%s",
                        generator, metrics["n_real"], metrics["n_ai"], _metric(metrics["auroc"]),
                        _metric(metrics["ap"]), _metric(metrics["fpr"]), _metric(metrics["tpr"]),
                        _metric(metrics["balanced_accuracy"]), _metric(metrics["coverage"]))
        overall = result["overall"]
        logger.info("Overall | AUROC=%s | AP=%s | FPR=%s | TPR=%s | BACC=%s | Macro_AUROC=%s | coverage=%s | decode_errors=%d | total_errors=%d",
                    _metric(overall["auroc"]), _metric(overall["ap"]), _metric(overall["fpr"]),
                    _metric(overall["tpr"]), _metric(overall["balanced_accuracy"]),
                    _metric(overall["macro_auroc"]), _metric(overall["coverage"]), decode_errors,
                    overall["num_errors"])
    else:
        quantiles = result["score_quantiles"]
        logger.info("COCO | N=%d | FPR=%s | coverage=%s | decode_errors=%d | total_errors=%d | score_mean=%s | score_std=%s | score_median=%s | q05=%s | q50=%s | q95=%s",
                    result["n"], _metric(result["fpr"]), _metric(result["coverage"]), decode_errors,
                    result["num_errors"], _metric(result["score_mean"]), _metric(result["score_std"]),
                    _metric(result["score_median"]), _metric(quantiles.get("0.05")),
                    _metric(quantiles.get("0.5")), _metric(quantiles.get("0.95")))
    logger.info("%s finished | elapsed=%.1fs | metrics=%s | scores=%s",
                phase, time.monotonic()-start, dest / "metrics.json", dest / "scores.csv")
    return result
