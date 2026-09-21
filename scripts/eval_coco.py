import argparse
from _evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scope_cr68_mdn3.yaml")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--standardizer", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, choices=[17, 42, 2026], required=True)
    parser.add_argument("--device", default="cpu")
    evaluate(parser.parse_args(), "real_external_coco")


if __name__ == "__main__":
    main()
