import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import torch
import yaml
from _common import load_config, read_json, write_json
from data.feature_cache import load_cache, feature_protocol_hash, cache_sha256
from engine.trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scope_cr68_mdn3.yaml")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--standardizer", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, choices=[17, 42, 2026], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config, standardizer = load_config(args.config), read_json(args.standardizer)
    protocol_hash = feature_protocol_hash(config)
    if standardizer["feature_protocol_hash"] != protocol_hash:
        raise ValueError("standardizer protocol mismatch")
    arrays, metas = [], []
    for split in ("real_train", "real_val"):
        split_path = Path(args.feature_dir) / split
        c, r, meta = load_cache(split_path, expected_protocol_hash=protocol_hash)
        with (split_path / "metadata.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(c) or any(row["label"] != "0" or row["source"] not in ("imagenet", "lsun") or row["status"] != "ok" for row in rows):
            raise ValueError(f"{split} must contain complete real ImageNet/LSUN cache")
        if Counter(row["source"] for row in rows) != config["data"][split]:
            raise ValueError(f"formal {split} source counts mismatch")
        arrays.extend((c, r))
        metas.append(meta)
    if metas[0]["manifest_sha256"] != standardizer["train_manifest_sha256"]:
        raise ValueError("train manifest and standardizer mismatch")
    train_cache = Path(args.feature_dir) / "real_train"
    cache_hash = cache_sha256(train_cache)
    if cache_hash != standardizer["train_cache_sha256"]:
        raise ValueError("train cache and standardizer mismatch")
    run_dir = Path(args.run_dir) / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(run_dir / "environment.json", dict(torch=torch.__version__, device=args.device))
    _, best = train(*arrays, standardizer, config, args.seed, run_dir, device=args.device, resume=args.resume,
                    val_manifest_hash=metas[1]["manifest_sha256"], tensorboard=True)
    print(best)


if __name__ == "__main__":
    main()
