"""Feature extraction and cache provenance."""
import csv
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from features.cr_features import extract_cr
from data.decoding import resize_if_needed
from data.manifests import decode_record, sha256_file
from data.patches import extract_patches

ERROR_SPLITS = frozenset({"genimage_eval", "real_external_eval", "real_external_coco"})


def feature_protocol_hash(config):
    root = Path(__file__).resolve().parents[1]
    implementation = hashlib.sha256()
    for name in ("data/decoding.py", "data/patches.py", "features/lowpass.py", "features/cr_features.py"):
        implementation.update(name.encode())
        implementation.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
    definition = dict(protocol_id=config["protocol"]["id"], decode_protocol="pillow-full-exif-rgb-alpha-white-v1",
                      resize_rule="short-side-64-ceil-bilinear-v1", patch_rule="four-quarter-centers-v1",
                      lowpass_kernel=config["feature"]["lowpass_1d"], lowpass_divisor=config["feature"]["lowpass_divisor"],
                      c_names=config["feature"]["c_names"], r_names=config["feature"]["r_names"],
                      delta=config["feature"]["delta"], correlation_clip=config["feature"]["correlation_clip"],
                      precision="CPU float64", implementation_sha256=implementation.hexdigest())
    return hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def extract_one(row):
    decoded = decode_record(row)
    if decoded.status != "ok":
        raise ValueError(f"{row['image_id']}: {decoded.status}")
    rgb, resize = resize_if_needed(decoded.rgb)
    patches, coords, unique = extract_patches(rgb.astype(np.float64) / 255.0)
    values = [extract_cr(p) for p in patches]
    metadata = {key: row.get(key, "") for key in ("image_id", "content_id", "source", "label", "generator")}
    metadata.update(upscaled=resize["upscaled"], original_width=decoded.original_width,
                    original_height=decoded.original_height, processed_width=resize["processed_width"],
                    processed_height=resize["processed_height"], scale_x=resize["scale_x"],
                    scale_y=resize["scale_y"], num_unique_patches=unique, status="ok",
                    error_type="", error_message="")
    return np.stack([v[0] for v in values]), np.stack([v[1] for v in values]), coords, metadata


def _error_record(row, exc):
    metadata = {key: row.get(key, "") for key in ("image_id", "content_id", "source", "label", "generator")}
    metadata.update(upscaled="", original_width=row.get("original_width", ""),
                    original_height=row.get("original_height", ""), processed_width="", processed_height="",
                    scale_x="", scale_y="", num_unique_patches="",
                    status=row.get("decode_status") if row.get("decode_status") != "ok" else "feature_error",
                    error_type=type(exc).__name__, error_message=str(exc))
    return (np.zeros((4, 6), np.float64), np.zeros((4, 8), np.float64),
            np.zeros((4, 2), np.int32), metadata)


def _extract_row(args):
    row, allow_errors = args
    try:
        return extract_one(row)
    except Exception as exc:
        if not allow_errors:
            raise
        return _error_record(row, exc)


def _extract_serial(rows, allow_errors):
    extracted, error_count = [], 0
    for row in rows:
        try:
            extracted.append(extract_one(row))
        except Exception as exc:
            if not allow_errors:
                raise
            error_count += 1
            extracted.append(_error_record(row, exc))
    return extracted, error_count


def _extract_parallel(rows, allow_errors, workers, chunksize, split_name, progress_interval):
    total = len(rows)
    tasks = [(row, allow_errors) for row in rows]
    extracted, error_count = [], 0
    started = time.perf_counter()
    label = split_name or "cache"
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, item in enumerate(executor.map(_extract_row, tasks, chunksize=chunksize), start=1):
            extracted.append(item)
            if item[3].get("status") != "ok":
                error_count += 1
            if index % progress_interval == 0 or index == total:
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed > 0 else 0.0
                remaining = (total - index) / rate if rate > 0 else 0.0
                print(f"[{label}] processed={index}/{total} | workers={workers} | elapsed={elapsed:.1f}s | "
                      f"images/s={rate:.1f} | errors={error_count} | ETA={remaining:.1f}s", flush=True)
    return extracted, error_count


def build_cache(rows, manifest_path, output, config, overwrite=False, *, workers=1, split_name="",
                progress_interval=500):
    if workers < 1:
        raise ValueError("workers must be >= 1")
    output = Path(output)
    protocol_hash = feature_protocol_hash(config)
    manifest_hash = sha256_file(manifest_path)
    meta_path = output / "cache_meta.json"
    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["feature_protocol_hash"] != protocol_hash or meta["manifest_sha256"] != manifest_hash:
            raise ValueError("cache protocol or manifest mismatch; use --overwrite")
        return meta
    output.mkdir(parents=True, exist_ok=True)
    allow_errors = any(row.get("split") in ERROR_SPLITS for row in rows) or split_name in ERROR_SPLITS
    chunksize = max(1, min(32, len(rows) // max(workers * 4, 1) or 1))
    if workers == 1:
        extracted, error_count = _extract_serial(rows, allow_errors)
    else:
        extracted, error_count = _extract_parallel(rows, allow_errors, workers, chunksize, split_name, progress_interval)
    if not extracted:
        raise ValueError("empty manifest")
    c, r, coords, metadata = zip(*extracted)
    np.save(output / "c_raw.npy", np.stack(c))
    np.save(output / "r_raw.npy", np.stack(r))
    np.save(output / "patch_yx.npy", np.stack(coords).astype(np.int32))
    with (output / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    meta = dict(protocol_id=config["protocol"]["id"], feature_protocol_hash=protocol_hash,
                manifest_sha256=manifest_hash, sample_count=len(c), c_shape=[len(c), 4, 6],
                r_shape=[len(c), 4, 8], error_count=error_count, numpy_version=np.__version__, pillow_version=PIL.__version__,
                python_version=platform.python_version(), created_at=datetime.now(timezone.utc).isoformat())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_cache(path, *, expected_protocol_hash=None, expected_manifest_hash=None):
    path = Path(path)
    meta = json.loads((path / "cache_meta.json").read_text(encoding="utf-8"))
    if expected_protocol_hash and meta["feature_protocol_hash"] != expected_protocol_hash:
        raise ValueError("feature protocol hash mismatch")
    if expected_manifest_hash and meta["manifest_sha256"] != expected_manifest_hash:
        raise ValueError("manifest hash mismatch")
    c, r = np.load(path / "c_raw.npy"), np.load(path / "r_raw.npy")
    if c.shape != tuple(meta["c_shape"]) or r.shape != tuple(meta["r_shape"]):
        raise ValueError("cache shape mismatch")
    return c, r, meta


def cache_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    for name in ("c_raw.npy", "r_raw.npy"):
        digest.update(bytes.fromhex(sha256_file(path / name)))
    return digest.hexdigest()
