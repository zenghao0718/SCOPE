import argparse
from pathlib import Path
from _common import load_config
from data.manifests import read_manifest
from data.feature_cache import build_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scope_cr68_mdn3.yaml")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--split", choices=["real_train", "real_val", "real_calibration", "real_external_coco", "genimage_eval"], required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="CPU worker processes for image-level extraction")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    manifest = Path(args.manifest_dir) / f"{args.split}.csv"
    rows = read_manifest(manifest)
    print(build_cache(rows, manifest, Path(args.feature_dir) / args.split, load_config(args.config), args.overwrite,
                      workers=args.workers, split_name=args.split))


if __name__ == "__main__":
    main()
