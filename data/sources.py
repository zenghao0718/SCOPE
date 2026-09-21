"""Explicit data discovery; never infer GenImage labels from directory names."""
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_files(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: str(p))


def file_records(root, source, *, label=0, generator="", group_id=""):
    prefix = f"genimage:{generator}:{label}" if source == "genimage" else source
    return [dict(image_id=f"{prefix}:{p.relative_to(root).as_posix()}", source=source,
                 source_category=p.parent.name, storage_type="file", path=str(p.resolve()),
                 container_path="", sample_key="", label=label, generator=generator,
                 group_id=group_id) for p in image_files(root)]


def genimage_records(mappings):
    if not mappings:
        raise ValueError("explicit GenImage generator/label/path mappings required")
    out = []
    seen = set()
    for index, item in enumerate(mappings):
        if not {"generator", "label", "path"} <= set(item) or item["label"] not in (0, 1):
            raise ValueError("each GenImage mapping needs generator, label, path")
        key = (item["generator"], item["label"], str(Path(item["path"]).resolve()))
        if key in seen:
            raise ValueError("ambiguous duplicate GenImage mapping")
        seen.add(key)
        records = file_records(item["path"], "genimage", label=item["label"], generator=item["generator"])
        out.extend({**record, "image_id": f"map{index}:{record['image_id']}"} for record in records)
    if len({r["path"] for r in out}) != len(out):
        raise ValueError("GenImage mappings overlap")
    return out
