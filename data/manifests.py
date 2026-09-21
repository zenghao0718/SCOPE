"""Frozen content-ID based split construction."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import PIL
from data.decoding import decode_image, canonical_content_id
from data.lsun_lmdb import read_record

FIELDS = ["image_id", "content_id", "source", "source_category", "split", "storage_type",
          "path", "container_path", "sample_key", "label", "generator", "original_width",
          "original_height", "decode_status", "group_id"]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_record(record):
    source = record["path"] if record["storage_type"] == "file" else read_record(record["container_path"], record["sample_key"])
    return decode_image(source)


def enrich(records):
    valid, errors = [], []
    for record in records:
        result = decode_record(record)
        row = {**record, "original_width": result.original_width, "original_height": result.original_height,
               "decode_status": result.status, "content_id": canonical_content_id(result.rgb) if result.rgb is not None else ""}
        (valid if result.status == "ok" else errors).append(row)
    return valid, errors


def stable_sort(rows, salt, source):
    return sorted(rows, key=lambda r: (hashlib.sha256(f"{salt}|{source}|{r['content_id']}".encode()).hexdigest(), r["content_id"]))


def deduplicate(rows):
    kept = {}
    for row in sorted(rows, key=lambda r: r["image_id"]):
        kept.setdefault(row["content_id"], row)
    return list(kept.values())


def build_splits(imagenet, lsun, coco, genimage, *, salt="20260917", counts=None):
    counts = counts or {"imagenet": (5000, 1000, 1000), "lsun": (5000, 1000, 1000), "coco": 2000}
    # Benchmark rows, including duplicates, remain unchanged.
    genimage = [{**r, "split": "genimage_eval"} for r in genimage]
    coco_unique = stable_sort(deduplicate(coco), salt, "coco")
    if len(coco_unique) < counts["coco"]:
        raise ValueError("insufficient COCO candidates")
    external = [{**r, "split": "real_external_eval"} for r in coco_unique[:counts["coco"]]]
    excluded = {r["content_id"] for r in genimage + external if r["content_id"]}
    im = deduplicate(r for r in imagenet if r["content_id"] not in excluded)
    im_ids = {r["content_id"] for r in im}
    ls = deduplicate(r for r in lsun if r["content_id"] not in excluded and r["content_id"] not in im_ids)
    splits = {name: [] for name in ("real_train", "real_val", "real_calibration")}
    for source, rows in (("imagenet", im), ("lsun", ls)):
        rows = stable_sort(rows, salt, source)
        needed = sum(counts[source])
        if len(rows) < needed:
            raise ValueError(f"insufficient {source} candidates: {len(rows)} < {needed}")
        offset = 0
        for name, count in zip(splits, counts[source]):
            splits[name].extend({**r, "split": name} for r in rows[offset:offset+count])
            offset += count
    groups = {}
    for name, rows in splits.items():
        for row in rows:
            group = row.get("group_id")
            if group:
                key = (row["source"], group)
                if key in groups and groups[key] != name:
                    raise ValueError(f"group {key} spans real splits")
                groups[key] = name
    return {**splits, "real_external_coco": external, "genimage_eval": genimage}


def write_manifests(splits, output, protocol_id, salt, stats=None, roots=None):
    output = Path(output)
    if (output / "manifest_meta.json").exists() or any((output / f"{name}.csv").exists() for name in splits):
        raise FileExistsError("manifest already frozen; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rows in splits.items():
        path = output / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        hashes[name] = sha256_file(path)
    meta = dict(protocol_id=protocol_id, salt=salt, manifest_sha256=hashes, stats=stats or {},
                source_roots=roots or {}, pillow_version=PIL.__version__,
                created_at=datetime.now(timezone.utc).isoformat())
    (output / "manifest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))
