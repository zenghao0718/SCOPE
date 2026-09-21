import argparse
import csv
from collections import Counter
from pathlib import Path
from _common import load_config, write_json
from scope_data.feature_cache import load_cache, feature_protocol_hash, cache_sha256
from features.standardization import fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scope_cr68_mdn3.yaml")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    cache = Path(args.feature_dir) / "real_train"
    c, r, meta = load_cache(cache, expected_protocol_hash=feature_protocol_hash(config))
    with (cache / "metadata.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(c) or any(row["label"] != "0" or row["source"] not in ("imagenet", "lsun") or row["status"] != "ok" for row in rows):
        raise ValueError("standardizer requires complete real_train ImageNet/LSUN cache")
    if Counter(row["source"] for row in rows) != config["data"]["real_train"]:
        raise ValueError("formal real_train source counts mismatch")
    artifact = fit(c, r, protocol_id=config["protocol"]["id"], feature_protocol_hash=meta["feature_protocol_hash"],
                   train_manifest_sha256=meta["manifest_sha256"], train_cache_sha256=cache_sha256(cache))
    write_json(args.output, artifact)
    print(artifact["artifact_sha256"])


if __name__ == "__main__":
    main()
