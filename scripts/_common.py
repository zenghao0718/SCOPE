import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from engine.config import load_config
from engine.checkpoint import load_checkpoint
from models.mdn import MDN


def paths_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_from_best(path, config, standardizer, seed, device="cpu"):
    checkpoint = load_checkpoint(path, device)
    if checkpoint["seed"] != seed or checkpoint["standardizer_hash"] != standardizer["artifact_sha256"]:
        raise ValueError("best checkpoint provenance mismatch")
    mc, fc = config["model"], config["feature"]
    model = MDN(fc["c_dim"], fc["r_dim"], mc["num_components"], mc["sigma_floor"], mc["hidden_dims"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()
