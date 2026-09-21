"""LSUN raw LMDB records, decoded in memory without re-encoding."""
from pathlib import Path


def lmdb_records(path):
    import lmdb
    path = Path(path).resolve()
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, subdir=path.is_dir())
    try:
        with env.begin() as tx:
            keys = [key for key, _ in tx.cursor() if key != b"__keys__" and key != b"__len__"]
    finally:
        env.close()
    return [dict(image_id=f"lsun:{path.name}:{key.hex()}", source="lsun", source_category=path.name,
                 storage_type="lmdb", path="", container_path=str(path), sample_key=key.hex(),
                 label=0, generator="", group_id="") for key in sorted(keys)]


def read_record(container_path, sample_key):
    import lmdb
    path = Path(container_path)
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, subdir=path.is_dir())
    try:
        with env.begin() as tx:
            value = tx.get(bytes.fromhex(sample_key))
            if value is None:
                raise KeyError(sample_key)
            return value
    finally:
        env.close()
