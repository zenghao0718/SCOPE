import argparse
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
    sampling_seed = config["data"]["candidate_sampling_seed"]
    gen_candidates = genimage_records(sources["genimage_eval"])
    if not gen_candidates:
        raise ValueError("empty GenImage evaluation candidate list")
    print(f"[GenImage] total candidates={len(gen_candidates)}", flush=True)
    gen_valid, gen_errors = enrich(gen_candidates, progress=lambda processed, valid, errors:
                                   print(f"[GenImage] decoded={processed} valid={valid} errors={errors}", flush=True))
    gen_by_id = {r["image_id"]: r for r in gen_valid + gen_errors}
    genimage = [gen_by_id[r["image_id"]] for r in gen_candidates]
    candidates = dict(coco=file_records(sources["coco_root"], "coco"),
                      imagenet=file_records(sources["imagenet_root"], "imagenet"))
    lsun = []
    for root in sources["lsun_roots"]:
        path = Path(root)
        is_lmdb = path.suffix.lower() == ".lmdb" or (path.is_dir() and (path / "data.mdb").is_file())
        lsun.extend(lmdb_records(path) if is_lmdb else file_records(path, "lsun"))
    candidates["lsun"] = lsun
    display = {"coco": "COCO", "imagenet": "ImageNet", "lsun": "LSUN"}
    for name, rows in candidates.items():
        print(f"[{display[name]}] total candidates={len(rows)}", flush=True)
    def progress(source, current):
        target = (config["data"]["real_external_eval"]["coco"] if source == "coco" else
                  sum(config["data"][split][source] for split in ("real_train", "real_val", "real_calibration")))
        print(f"[{display[source]}] processed={current['processed_candidates']} selected={current['selected_candidates']} "
              f"target={target} "
              f"duplicates={current['duplicate_rejections']} excluded={current['overlap_rejections']} "
              f"errors={current['decode_errors']}", flush=True)
    splits, stats = build_splits(**candidates, genimage=genimage, candidate_sampling_seed=sampling_seed,
                                 progress=progress)
    stats["genimage"] = dict(total_candidates=len(gen_candidates), processed_candidates=len(gen_candidates),
                             selected_candidates=len(gen_candidates), decode_errors=len(gen_errors),
                             duplicate_rejections=0, overlap_rejections=0, valid_candidates=len(gen_valid))
    stats["final_splits"] = {name: len(rows) for name, rows in splits.items()}
    meta = write_manifests(splits, args.output_dir, config["protocol"]["id"], sampling_seed, stats, sources)
    print(meta["manifest_sha256"])


if __name__ == "__main__":
    main()
