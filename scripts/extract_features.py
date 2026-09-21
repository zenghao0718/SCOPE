import argparse
from pathlib import Path
from _common import load_config
from scope_data.manifests import read_manifest
from scope_data.feature_cache import build_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scope_cr68_mdn3.yaml")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--split", choices=["real_train", "real_val", "real_calibration", "real_external_coco", "genimage_eval"], required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = Path(args.manifest_dir) / f"{args.split}.csv"
    rows = read_manifest(manifest)
    print(build_cache(rows, manifest, Path(args.feature_dir) / args.split, load_config(args.config), args.overwrite))


if __name__ == "__main__":
    main()
