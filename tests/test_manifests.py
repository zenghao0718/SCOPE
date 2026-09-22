from types import SimpleNamespace
from pathlib import Path
from io import BytesIO

import lmdb
import numpy as np
import pytest
import yaml
from PIL import Image

from data.decoding import canonical_content_id
from data.manifests import build_splits, decode_record
from data.lsun_lmdb import lmdb_records, read_record
from data.sampling import deterministic_candidate_order, collect_valid_unique
from engine.config import load_config


def candidate(source, identity, pixel):
    return dict(image_id=f"{source}:{identity}", source=source, source_category="synthetic",
                storage_type="file", path=f"/synthetic/{source}/{identity}.png", container_path="",
                sample_key="", label=0, generator="", group_id="", pixel=pixel)


def content_id(pixel):
    return canonical_content_id(np.full((2, 2, 3), pixel, np.uint8))


def decode(record):
    if record["pixel"] is None:
        return SimpleNamespace(status="decode_error", rgb=None, original_width=0, original_height=0)
    return SimpleNamespace(status="ok", rgb=np.full((2, 2, 3), record["pixel"], np.uint8),
                           original_width=2, original_height=2)


def genimage(pixel):
    return dict(candidate("genimage", "eval", pixel), content_id=content_id(pixel), decode_status="ok")


def test_same_seed_order_independence_and_formal_isolation():
    im = [candidate("imagenet", str(i), i) for i in range(2, 25)]
    ls = [candidate("lsun", str(i), i) for i in range(2, 10)] + [candidate("lsun", str(i), i+40) for i in range(10, 25)]
    co = [candidate("coco", str(i), i) for i in range(30, 45)]
    counts = {"imagenet": (2, 1, 1), "lsun": (2, 1, 1), "coco": 1}
    first, stats = build_splits(im, ls, co, [genimage(0)], counts=counts, decode_fn=decode)
    second, _ = build_splits(list(reversed(im)), list(reversed(ls)), list(reversed(co)),
                             [genimage(0)], counts=counts, decode_fn=decode)
    assert {name: [r["image_id"] for r in rows] for name, rows in first.items()} == {
        name: [r["image_id"] for r in rows] for name, rows in second.items()}
    assert stats["imagenet"]["selected_candidates"] == 4
    assert stats["lsun"]["selected_candidates"] == 4
    assert stats["coco"]["selected_candidates"] == 1
    real = sum((first[s] for s in ("real_train", "real_val", "real_calibration", "real_external_coco")), [])
    assert len(real) == 9 and len({r["content_id"] for r in real}) == len(real)
    assert all(r["content_id"] != content_id(0) for r in real)


def test_incremental_refill_stays_on_one_permutation():
    candidates = [candidate("imagenet", str(i), value) for i, value in enumerate(
        [None, 1, 1, 2, 3, 4, 5, None, 6, 7, 8, 9])]
    ordered = deterministic_candidate_order(candidates, 20260917)
    calls = []
    def counted(record):
        calls.append(record["image_id"])
        return decode(record)
    selected, stats = collect_valid_unique(ordered, 4, {content_id(3)}, counted, source="imagenet")
    assert len(selected) == 4
    assert stats["processed_candidates"] == len(calls)
    assert stats["processed_candidates"] > 4
    assert calls == [r["image_id"] for r in ordered[:len(calls)]]
    assert stats["decode_errors"] + stats["duplicate_rejections"] + stats["overlap_rejections"] > 0


def test_no_full_decode_and_shortage():
    candidates = [candidate("coco", str(i), i) for i in range(100)]
    calls = []
    def counted(record):
        calls.append(record["image_id"])
        return decode(record)
    selected, stats = collect_valid_unique(deterministic_candidate_order(candidates, 20260917),
                                           5, set(), counted, source="coco")
    assert len(selected) == 5 and stats["processed_candidates"] == len(calls) == 5
    with pytest.raises(ValueError, match="insufficient coco candidates"):
        collect_valid_unique(candidates[:2], 3, set(), counted, source="coco")


def test_build_splits_decodes_only_consumed_real_candidates():
    im = [candidate("imagenet", str(i), i + 40) for i in range(100)]
    ls = [candidate("lsun", str(i), i + 145) for i in range(100)]
    co = [candidate("coco", str(i), i + 1) for i in range(100)]
    calls = {"imagenet": 0, "lsun": 0, "coco": 0}
    def counted(record):
        calls[record["source"]] += 1
        return decode(record)
    _, stats = build_splits(im, ls, co, [genimage(0)], decode_fn=counted,
                            counts={"imagenet": (5, 0, 0), "lsun": (5, 0, 0), "coco": 5})
    assert calls == {"imagenet": 5, "lsun": 5, "coco": 5}
    assert all(stats[source]["processed_candidates"] == 5 for source in calls)


def test_unused_imagenet_candidate_does_not_exclude_lsun():
    im = [candidate("imagenet", str(i), i) for i in range(10, 16)]
    unselected_pixel = deterministic_candidate_order(im, 20260917)[1]["pixel"]
    ls = [candidate("lsun", "same-as-unused-imagenet", unselected_pixel)]
    co = [candidate("coco", "external", 50)]
    splits, _ = build_splits(im, ls, co, [genimage(0)], decode_fn=decode,
                             counts={"imagenet": (1, 0, 0), "lsun": (1, 0, 0), "coco": 1})
    assert splits["real_train"][1]["content_id"] == content_id(unselected_pixel)


def test_formal_sampling_seed_cannot_change(tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs" / "scope_cr68_mdn3.yaml"
    config = load_config(path)
    config["data"]["candidate_sampling_seed"] = 1
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="formal configuration changed"):
        load_config(changed)


def test_genimage_must_be_fully_decoded_before_sampling():
    with pytest.raises(ValueError, match="GenImage benchmark must be fully decoded"):
        build_splits([], [], [], [candidate("genimage", "raw", 1)], decode_fn=decode,
                     counts={"imagenet": (1, 0, 0), "lsun": (1, 0, 0), "coco": 1})


def test_lmdb_enumerates_keys_and_decodes_only_selected(tmp_path):
    root = tmp_path / "tiny.lmdb"
    env = lmdb.open(str(root), map_size=1024 * 1024)
    with env.begin(write=True) as tx:
        for index in range(3):
            stream = BytesIO()
            Image.fromarray(np.full((64, 64, 3), index + 1, np.uint8)).save(stream, format="PNG")
            tx.put(f"key-{index}".encode(), stream.getvalue())
    env.close()
    records = lmdb_records(root)
    assert len(records) == 3
    assert read_record(records[0]["container_path"], records[0]["sample_key"])
    selected, stats = collect_valid_unique(deterministic_candidate_order(records, 20260917),
                                           1, set(), decode_record, source="lsun")
    assert len(selected) == 1 and stats["processed_candidates"] == 1
