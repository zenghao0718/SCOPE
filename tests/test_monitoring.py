from pathlib import Path

import numpy as np
import torch

from engine.config import load_config
from engine.checkpoint import load_checkpoint
from engine.logging import setup_logger
from engine.trainer import train
from features.standardization import fit


def test_logger_has_console_and_file_without_duplicate_handlers(tmp_path):
    logger, path = setup_logger(tmp_path, "calibration", seed=17)
    logger_again, same_path = setup_logger(tmp_path, "calibration", seed=17)
    assert logger is logger_again and path == same_path
    assert len(logger.handlers) == 2
    logger.info("Calibration finished | seed=17 | tau=1.25")
    content = path.read_text(encoding="utf-8")
    assert content.count("Calibration finished") == 1
    assert " | INFO | " in content


def test_tensorboard_switch_preserves_training_result(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "scope_cr68_mdn3.yaml")
    config["train"]["batch_size_images"] = 2
    config["train"]["max_epochs"] = 1
    rng = np.random.default_rng(123)
    c_train = rng.normal(size=(3, 4, 6))
    r_train = rng.normal(size=(3, 4, 8))
    c_val = rng.normal(size=(2, 4, 6))
    r_val = rng.normal(size=(2, 4, 8))
    standardizer = fit(c_train, r_train, protocol_id=config["protocol"]["id"],
                       feature_protocol_hash="synthetic", train_manifest_sha256="synthetic",
                       train_cache_sha256="synthetic")
    for enabled in (False, True):
        train(c_train, r_train, c_val, r_val, standardizer, config, 17,
              tmp_path / str(enabled), tensorboard=enabled)
    off = load_checkpoint(tmp_path / "False" / "checkpoints" / "best.pt")
    on = load_checkpoint(tmp_path / "True" / "checkpoints" / "best.pt")
    assert off["val_nll"] == on["val_nll"]
    assert all(torch.equal(off["model_state"][key], on["model_state"][key])
               for key in off["model_state"])
