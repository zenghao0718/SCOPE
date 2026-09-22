"""Deterministic candidate ordering and incremental content-ID selection."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import lmdb
import numpy as np

from data.decoding import canonical_content_id


def stable_identity(record: dict) -> str:
    if record["storage_type"] == "file":
        return f"file|{Path(record['path']).as_posix()}"
    if record["storage_type"] == "lmdb":
        return f"lmdb|{Path(record['container_path']).as_posix()}|{record['sample_key']}"
    raise ValueError(f"unsupported storage type: {record['storage_type']}")


def deterministic_candidate_order(candidates: Iterable[dict], seed: int, identity_fn=stable_identity) -> list[dict]:
    candidates = sorted(candidates, key=identity_fn)
    identities = [identity_fn(record) for record in candidates]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate candidate identity")
    order = np.random.default_rng(seed).permutation(len(candidates))
    return [candidates[int(index)] for index in order]


def collect_valid_unique(
    ordered_candidates: Iterable[dict],
    target_count: int,
    excluded_content_ids: set[str],
    decode_fn: Callable[[dict], object],
    *,
    source: str,
    progress: Callable[[str, dict], None] | None = None,
    progress_every: int = 500,
) -> tuple[list[dict], dict]:
    if target_count < 1 or progress_every < 1:
        raise ValueError("target_count and progress_every must be positive")
    candidates = list(ordered_candidates)
    stats = dict(total_candidates=len(candidates), processed_candidates=0, selected_candidates=0,
                 decode_errors=0, duplicate_rejections=0, overlap_rejections=0)
    selected = []
    selected_ids = set()
    for record in candidates:
        stats["processed_candidates"] += 1
        try:
            result = decode_fn(record)
        except (OSError, ValueError, KeyError, RuntimeError, lmdb.Error):
            stats["decode_errors"] += 1
        else:
            if result.status != "ok" or result.rgb is None:
                stats["decode_errors"] += 1
            else:
                content_id = canonical_content_id(result.rgb)
                if content_id in excluded_content_ids:
                    stats["overlap_rejections"] += 1
                elif content_id in selected_ids:
                    stats["duplicate_rejections"] += 1
                else:
                    selected_ids.add(content_id)
                    selected.append({**record, "content_id": content_id,
                                     "original_width": result.original_width,
                                     "original_height": result.original_height,
                                     "decode_status": result.status})
                    stats["selected_candidates"] += 1
        if progress and (stats["processed_candidates"] % progress_every == 0 or len(selected) == target_count):
            progress(source, dict(stats))
        if len(selected) == target_count:
            return selected, stats
    if progress:
        progress(source, dict(stats))
    raise ValueError(f"insufficient {source} candidates: selected={len(selected)} target={target_count} "
                     f"processed={stats['processed_candidates']} total={stats['total_candidates']}")
