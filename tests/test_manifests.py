import pytest
from data.manifests import build_splits


def row(source, ident):
    return dict(image_id=f"{source}:{ident}", content_id=ident, source=source, decode_status="ok", group_id="")


def test_eval_exclusion_cross_source_priority_and_shortage():
    im = [row("imagenet", str(i)) for i in range(8)]
    ls = [row("lsun", str(i)) for i in range(2)] + [row("lsun", f"L{i}") for i in range(8)]
    co = [row("coco", "C")]
    gen = [row("genimage", "0")]
    result = build_splits(im, ls, co, gen, counts={"imagenet": (2, 1, 1), "lsun": (2, 1, 1), "coco": 1})
    real = sum((result[s] for s in ("real_train", "real_val", "real_calibration")), [])
    assert len(real) == 8 and len({r["content_id"] for r in real}) == 8
    assert all(r["content_id"] not in {"C", "0"} for r in real)
    with pytest.raises(ValueError):
        build_splits([], [], co, gen)
