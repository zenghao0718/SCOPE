# SCOPE CR68-MDN3

SCOPE is a real-only conditional residual statistics anomaly detector. Four fixed 64×64 patches per image yield a 6-dimensional image state (C) and 8-dimensional residual statistic (R). A three-component diagonal Gaussian MDN learns (P(R\mid C)) from real images only. The mean patch NLL is an anomaly score, **not an AI probability**. [METHOD_SPEC.md](docs/METHOD_SPEC.md) is the authoritative method definition.

## Layout

- `data/`: unified decoding, content IDs, manifests, patches, and feature caches
- `features/`: fixed C/R formulas and train-only standardization
- `models/`, `losses/`, `engine/`: MDN, full mixture NLL, deterministic training
- `evaluation/`, `scripts/`: calibration, evaluation, summary, and command line entry points
- `docs/`: method and implementation specifications

Use sibling directories `SCOPE_DATA/` for manifests and caches and `SCOPE_RUNS/` for model and evaluation outputs. Copy `configs/local_paths.example.yaml` to ignored `configs/local_paths.yaml`, then set explicit ImageNet, LSUN, COCO, and GenImage evaluation paths. GenImage entries require explicit generator and label mappings. They are never guessed from directory names.

## Environment

Install a PyTorch build appropriate for the machine first. On AutoDL, keep its working CUDA PyTorch installation. Then run `python -m pip install -e '.[dev]'`. The dependency declaration does not select or force a CUDA wheel index.

## Formal workflow

The formal data split and feature cache are shared by all three seeds. Freeze GenImage first, COCO second, and exclude their content IDs from ImageNet/LSUN before deterministic sorting and splitting. Only `real_train` fits the standardizer or trains the MDN. `real_val` selects the best checkpoint and controls stopping. `real_calibration` only sets the threshold. COCO and GenImage are final evaluation only; their results must never change the model or protocol.

```bash
python scripts/build_manifests.py --paths-config configs/local_paths.yaml --output-dir ../SCOPE_DATA/manifests/SCOPE_CR68_MDN3_v1.0
python scripts/extract_features.py --manifest-dir ../SCOPE_DATA/manifests/SCOPE_CR68_MDN3_v1.0 --feature-dir ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0 --split real_train
python scripts/extract_features.py --manifest-dir ../SCOPE_DATA/manifests/SCOPE_CR68_MDN3_v1.0 --feature-dir ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0 --split real_val
python scripts/extract_features.py --manifest-dir ../SCOPE_DATA/manifests/SCOPE_CR68_MDN3_v1.0 --feature-dir ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0 --split real_calibration
python scripts/extract_features.py --manifest-dir ../SCOPE_DATA/manifests/SCOPE_CR68_MDN3_v1.0 --feature-dir ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0 --split real_external_coco
python scripts/extract_features.py --manifest-dir ../SCOPE_DATA/manifests/SCOPE_CR68_MDN3_v1.0 --feature-dir ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0 --split genimage_eval
python scripts/fit_standardizer.py --feature-dir ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0 --output ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/standardizer.json
python scripts/train.py --feature-dir ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0 --standardizer ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/standardizer.json --run-dir ../SCOPE_RUNS/SCOPE_CR68_MDN3_v1.0 --seed 17
python scripts/calibrate.py --feature-dir ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0 --manifest-dir ../SCOPE_DATA/manifests/SCOPE_CR68_MDN3_v1.0 --standardizer ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/standardizer.json --run-dir ../SCOPE_RUNS/SCOPE_CR68_MDN3_v1.0 --seed 17
python scripts/eval_coco.py --feature-dir ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0 --manifest-dir ../SCOPE_DATA/manifests/SCOPE_CR68_MDN3_v1.0 --standardizer ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/standardizer.json --run-dir ../SCOPE_RUNS/SCOPE_CR68_MDN3_v1.0 --seed 17
python scripts/eval_genimage.py --feature-dir ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0 --manifest-dir ../SCOPE_DATA/manifests/SCOPE_CR68_MDN3_v1.0 --standardizer ../SCOPE_DATA/features/SCOPE_CR68_MDN3_v1.0/standardizer.json --run-dir ../SCOPE_RUNS/SCOPE_CR68_MDN3_v1.0 --seed 17
```

Repeat train through evaluation for seeds `42` and `2026`, then run `python scripts/summarize_seeds.py --run-dir ../SCOPE_RUNS/SCOPE_CR68_MDN3_v1.0`. `python -m pytest -q` runs synthetic tests without formal data.

## Monitoring and results

Each seed writes its own timestamped training, calibration, COCO, and GenImage logs under `seed_<seed>/logs/`. Every log streams to the terminal and a UTF-8 `.log` file. Training also keeps `logs/metrics.csv` and grouped TensorBoard events in `seed_<seed>/tensorboard/`. View them with:

```bash
tensorboard --logdir ../SCOPE_RUNS/SCOPE_CR68_MDN3_v1.0 --bind_all --port 6006
```

Calibration scores and threshold metadata are in `seed_<seed>/calibration/`. COCO and GenImage score rows and metrics are in `seed_<seed>/evaluation/real_external_coco/` and `seed_<seed>/evaluation/genimage_eval/` respectively. These reports are for audit and final evaluation; they do not feed back into training.
