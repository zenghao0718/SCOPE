import argparse
from pathlib import Path
from _common import read_json, write_json, write_csv
from evaluation.summarize import summarize_seed_values


def flatten(prefix, value):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(flatten(f"{prefix}_{key}" if prefix else key, child))
        return result
    return {prefix: value} if isinstance(value, (int, float)) and value is not None else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    values = {}
    for seed in (17, 42, 2026):
        base = root / f"seed_{seed}" / "evaluation"
        gen = read_json(base / "genimage_eval" / "metrics.json")
        coco = read_json(base / "real_external_coco" / "metrics.json")
        values[seed] = {**flatten("genimage", gen), **flatten("coco", coco)}
    summary = summarize_seed_values(values)
    out = root / "summary"
    columns = sorted(set.union(*(set(metrics) for metrics in values.values())))
    write_csv(out / "per_seed_metrics.csv", [dict(seed=seed, **{key: metrics.get(key, "") for key in columns})
                                               for seed, metrics in values.items()])
    write_csv(out / "summary_mean_std.csv", [dict(metric=k, **v) for k, v in summary.items()])
    write_json(out / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
