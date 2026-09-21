"""Per-run console, UTF-8 file, and CSV logging."""
from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(run_dir, phase, *, seed=None, level=logging.INFO):
    """Return one logger and timestamped path without stacking handlers."""
    if phase not in {"train", "calibration", "coco_eval", "genimage_eval"}:
        raise ValueError(f"unknown log phase: {phase}")
    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    name = f"scope.{phase}.{seed}.{log_dir.resolve()}"
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return logger, Path(handler.baseFilename)
    path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{phase}.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    for handler in (console, file_handler):
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def append_metrics(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
