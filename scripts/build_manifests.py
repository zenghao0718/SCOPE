import argparse
from collections import Counter
from pathlib import Path
from _common import load_config, paths_config
from data.sources import file_records, genimage_records
from data.lsun_lmdb import lmdb_records
from data.manifests import enrich, build_splits, write_manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scope_cr68_mdn3.yaml")
    parser.add_argument("--paths-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config, paths = load_config(args.config), paths_config(args.paths_config)
    sources = paths["sources"]
    candidates = dict(genimage=genimage_records(sources["genimage_eval"]),
                      coco=file_records(sources["coco_root"], "coco"),
                      imagenet=file_records(sources["imagenet_root"], "imagenet"))
    lsun = []
    for root in sources["lsun_roots"]:
        path = Path(root)
        is_lmdb = path.suffix.lower() == ".lmdb" or (path.is_dir() and (path / "data.mdb").is_file())
        lsun.extend(lmdb_records(path) if is_lmdb else file_records(path, "lsun"))
    candidates["lsun"] = lsun
    valid, stats = {}, {}
    for name, rows in candidates.items():
        valid[name], errors = enrich(rows)
        stats[name] = dict(candidates=len(rows), rejected=len(errors), valid=len(valid[name]),
                           duplicate_content=len(valid[name]) - len({r["content_id"] for r in valid[name]}),
                           errors=dict(Counter(r["decode_status"] for r in errors)))
        if name == "genimage":
            # Keep failed benchmark entries for coverage accounting.
            valid[name].extend(errors)
    splits = build_splits(**valid, salt=config["data"]["split_salt"])
    for name, rows in splits.items():
        stats[name] = dict(final=len(rows))
    meta = write_manifests(splits, args.output_dir, config["protocol"]["id"], config["data"]["split_salt"], stats, sources)
    print(meta["manifest_sha256"])


if __name__ == "__main__":
    main()
