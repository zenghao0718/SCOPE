"""Frozen content-ID based split construction from sampled real candidates."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import PIL
import numpy as np
from data.decoding import decode_image, canonical_content_id
from data.lsun_lmdb import read_record
from data.sampling import deterministic_candidate_order, collect_valid_unique

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


def enrich(records, *, progress=None, progress_every=500):
    valid, errors = [], []
    for index, record in enumerate(records, start=1):
        result = decode_record(record)
        row = {**record, "original_width": result.original_width, "original_height": result.original_height,
               "decode_status": result.status, "content_id": canonical_content_id(result.rgb) if result.rgb is not None else ""}
        (valid if result.status == "ok" else errors).append(row)
        if progress and (index % progress_every == 0 or index == len(records)):
            progress(index, len(valid), len(errors))
    return valid, errors


def build_splits(imagenet, lsun, coco, genimage, *, candidate_sampling_seed=20260917,
                 counts=None, decode_fn=decode_record, progress=None):
    """Consume real candidates in COCO, ImageNet, LSUN order; GenImage is pre-frozen."""
    counts = counts or {"imagenet": (5000, 1000, 1000), "lsun": (5000, 1000, 1000), "coco": 2000}
    if not genimage or any("decode_status" not in r or (r["decode_status"] == "ok" and not r.get("content_id"))
                           for r in genimage):
        raise ValueError("GenImage benchmark must be fully decoded before real candidate sampling")
    # Benchmark rows, including duplicates, remain unchanged.
    genimage = [{**r, "split": "genimage_eval"} for r in genimage]
    excluded = {r["content_id"] for r in genimage if r.get("content_id")}
    stats = {}
    external, stats["coco"] = collect_valid_unique(
        deterministic_candidate_order(coco, candidate_sampling_seed), counts["coco"],
        excluded, decode_fn, source="coco", progress=progress)
    external = [{**r, "split": "real_external_eval"} for r in external]
    excluded.update(r["content_id"] for r in external)
    im, stats["imagenet"] = collect_valid_unique(
        deterministic_candidate_order(imagenet, candidate_sampling_seed), sum(counts["imagenet"]),
        excluded, decode_fn, source="imagenet", progress=progress)
    excluded.update(r["content_id"] for r in im)
    ls, stats["lsun"] = collect_valid_unique(
        deterministic_candidate_order(lsun, candidate_sampling_seed), sum(counts["lsun"]),
        excluded, decode_fn, source="lsun", progress=progress)
    splits = {name: [] for name in ("real_train", "real_val", "real_calibration")}
    for source, rows in (("imagenet", im), ("lsun", ls)):
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
    return {**splits, "real_external_coco": external, "genimage_eval": genimage}, stats


def write_manifests(splits, output, protocol_id, candidate_sampling_seed, stats=None, roots=None):
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
    meta = dict(protocol_id=protocol_id, candidate_sampling_seed=candidate_sampling_seed,
                manifest_sha256=hashes, stats=stats or {},
                source_roots=roots or {}, pillow_version=PIL.__version__, numpy_version=np.__version__,
                created_at=datetime.now(timezone.utc).isoformat())
    (output / "manifest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))
