"""Image-weighted deterministic training and validation."""
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from features.standardization import transform
from losses.mdn_nll import patch_nll
from engine.seed import set_seed, epoch_order
from engine.checkpoint import save_checkpoint, load_checkpoint
from engine.logging import append_metrics, setup_logger
from models.mdn import MDN


def _score_batch(model, c, r):
    b = len(c)
    nll = patch_nll(r.reshape(-1, r.shape[-1]), *model(c.reshape(-1, c.shape[-1])))
    return nll.reshape(b, 4).mean(dim=1)


@torch.no_grad()
def validate(model, c, r, batch_size, device):
    model.eval()
    total = 0.0
    for start in range(0, len(c), batch_size):
        cb = torch.from_numpy(c[start:start+batch_size]).to(device)
        rb = torch.from_numpy(r[start:start+batch_size]).to(device)
        scores = _score_batch(model, cb, rb)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("nonfinite validation score")
        total += float(scores.double().sum().item())
    return total / len(c)


@torch.no_grad()
def mdn_diagnostics(model, c, batch_size, device):
    """Deterministic validation-only readout; preserve the incoming mode."""
    was_training = model.training
    model.eval()
    sigmas = []
    weights = []
    try:
        for start in range(0, len(c), batch_size):
            cb = torch.from_numpy(c[start:start+batch_size].reshape(-1, c.shape[-1])).to(device)
            log_pi, _, sigma = model(cb)
            sigmas.append(sigma.detach().cpu().reshape(-1))
            weights.append(log_pi.exp().detach().cpu())
    finally:
        model.train(was_training)
    sigma = torch.cat(sigmas)
    pi = torch.cat(weights).mean(dim=0)
    return dict(sigma_min=float(sigma.min()), sigma_median=float(sigma.median()),
                sigma_max=float(sigma.max()), mixture_weights=pi.tolist())


def _git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def train(c_train, r_train, c_val, r_val, standardizer, config, seed, run_dir, *, device="cpu", resume=False, val_manifest_hash="", tensorboard=False):
    run_dir = Path(run_dir)
    logger, log_path = setup_logger(run_dir, "train", seed=seed)
    writer = None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), flush_secs=10)
        writer.add_custom_scalars({"Overview": {
            "Train NLL vs Val NLL": ["Multiline", ["Loss/Train_NLL", "Loss/Val_NLL"]],
            "Best Val NLL": ["Multiline", ["Training/Best_Val_NLL"]],
            "Learning Rate": ["Multiline", ["Optimization/Learning_Rate"]],
        }})
    set_seed(seed)
    ct, rt = transform(c_train, r_train, standardizer)
    cv, rv = transform(c_val, r_val, standardizer)
    if not len(ct) or not len(cv):
        raise ValueError("training and validation must be nonempty")
    mc = config["model"]
    model = MDN(config["feature"]["c_dim"], config["feature"]["r_dim"],
                mc["num_components"], mc["sigma_floor"], mc["hidden_dims"]).to(device)
    tc = config["train"]
    weights = [p for name, p in model.named_parameters() if name.endswith("weight")]
    biases = [p for name, p in model.named_parameters() if name.endswith("bias")]
    optimizer = torch.optim.AdamW([{"params": weights, "weight_decay": tc["weight_decay_weight"]},
                                   {"params": biases, "weight_decay": 0.0}], lr=tc["learning_rate"],
                                  betas=tuple(tc["betas"]), eps=tc["adam_eps"], amsgrad=False,
                                  foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config["scheduler"])
    start, best, reference, bad = 1, float("inf"), float("inf"), 0
    last_path = run_dir / "checkpoints" / "last.pt"
    best_path = run_dir / "checkpoints" / "best.pt"
    logger.info("Start training | seed=%s | device=%s | resume=%s", seed, device, resume)
    logger.info("Protocol=%s | git_commit=%s | torch=%s | cuda_available=%s | gpu=%s",
                config["protocol"]["id"], _git_commit(), torch.__version__, torch.cuda.is_available(),
                torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "none")
    logger.info("train_images=%s | val_images=%s | batch_size=%s | max_epochs=%s | lr=%s | K=%s",
                len(ct), len(cv), tc["batch_size_images"], tc["max_epochs"], tc["learning_rate"], mc["num_components"])
    logger.info("standardizer_hash=%s | train_manifest_hash=%s | val_manifest_hash=%s",
                standardizer["artifact_sha256"], standardizer["train_manifest_sha256"], val_manifest_hash)
    overall_start = time.monotonic()
    best_epoch = None
    final_epoch = start - 1
    if resume:
        checkpoint = load_checkpoint(last_path, device)
        if (checkpoint["model_seed"] != seed or checkpoint["standardizer_hash"] != standardizer["artifact_sha256"]
                or checkpoint["protocol_id"] != config["protocol"]["id"]
                or checkpoint["val_manifest_hash"] != val_manifest_hash):
            raise ValueError("resume provenance mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if str(device).startswith("cuda") and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        start, best, reference, bad = checkpoint["epoch"] + 1, checkpoint["best_value"], checkpoint["stop_reference"], checkpoint["bad_epochs"]
        if best_path.exists():
            best_epoch = load_checkpoint(best_path, "cpu")["epoch"]
        logger.info("Resume from epoch %03d | next_epoch=%03d", checkpoint["epoch"], start)
    for epoch in range(start, tc["max_epochs"] + 1):
        tick = time.monotonic()
        model.train()
        total = 0.0
        grad_norm_total = 0.0
        batch_count = 0
        lr_used = optimizer.param_groups[0]["lr"]
        order = epoch_order(len(ct), seed, epoch)
        batch_size = tc["batch_size_images"]
        for start_index in range(0, len(order), batch_size):
            indexes = order[start_index:start_index+batch_size]
            c = torch.from_numpy(ct[indexes]).to(device)
            r = torch.from_numpy(rt[indexes]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _score_batch(model, c, r).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip_norm"])
            grad_norm_total += float(grad_norm)
            batch_count += 1
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite parameter")
            total += float(loss.detach()) * len(indexes)
        train_nll = total / len(ct)
        val_nll = validate(model, cv, rv, batch_size, device)
        if val_nll < best:
            best = val_nll
            best_epoch = epoch
            save_checkpoint(best_path, dict(model_state=model.state_dict(), epoch=epoch,
                            val_nll=val_nll, seed=seed, protocol_id=config["protocol"]["id"],
                            feature_protocol_hash=standardizer["feature_protocol_hash"],
                            standardizer_hash=standardizer["artifact_sha256"], config=config))
            logger.info("New best checkpoint saved | epoch=%03d | val_nll=%.9g | path=%s", epoch, val_nll, best_path)
        if val_nll < reference - config["early_stopping"]["min_delta"]:
            reference, bad = val_nll, 0
        else:
            bad += 1
        scheduler.step(val_nll)
        lr_next = optimizer.param_groups[0]["lr"]
        if lr_next < lr_used:
            logger.info("Learning rate reduced | epoch=%03d | from=%.9g | to=%.9g", epoch, lr_used, lr_next)
        save_checkpoint(last_path, dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                        scheduler_state=scheduler.state_dict(), epoch=epoch, best_value=best,
                        stop_reference=reference, bad_epochs=bad, model_seed=seed,
                        python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state(),
                        torch_rng_state=torch.get_rng_state(),
                        cuda_rng_state=torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None,
                        protocol_id=config["protocol"]["id"], feature_protocol_hash=standardizer["feature_protocol_hash"],
                        standardizer_hash=standardizer["artifact_sha256"],
                        train_manifest_hash=standardizer["train_manifest_sha256"], val_manifest_hash=val_manifest_hash,
                        config=config))
        append_metrics(run_dir / "logs" / "metrics.csv", dict(epoch=epoch, train_nll=train_nll, val_nll=val_nll,
                       learning_rate=lr_used, best_value=best, stop_reference=reference,
                       bad_epochs=bad, elapsed_seconds=time.monotonic()-tick,
                       grad_norm=grad_norm_total / batch_count))
        diagnostics = mdn_diagnostics(model, cv, batch_size, device)
        elapsed = time.monotonic() - tick
        eta = elapsed * (tc["max_epochs"] - epoch)
        logger.info("Epoch %03d/%03d | train_nll=%.9g | val_nll=%.9g | best=%.9g | lr=%.9g | bad_epochs=%d | grad_norm=%.9g | time=%.1fs | ETA=%.1fs",
                    epoch, tc["max_epochs"], train_nll, val_nll, best, lr_used, bad,
                    grad_norm_total / batch_count, elapsed, eta)
        if writer is not None:
            scalars = {
                "Loss/Train_NLL": train_nll, "Loss/Val_NLL": val_nll,
                "Optimization/Learning_Rate": lr_used,
                "Optimization/Grad_Norm": grad_norm_total / batch_count,
                "Training/Best_Val_NLL": best, "Training/Stop_Reference": reference,
                "Training/Bad_Epochs": bad, "Training/Epoch_Time_Seconds": elapsed,
                "MDN/Sigma_Min": diagnostics["sigma_min"],
                "MDN/Sigma_Median": diagnostics["sigma_median"],
                "MDN/Sigma_Max": diagnostics["sigma_max"],
            }
            scalars.update({f"MDN/Mixture_Weight_{k+1}": weight for k, weight in enumerate(diagnostics["mixture_weights"])})
            for tag, value in scalars.items():
                writer.add_scalar(tag, value, epoch)
            writer.flush()
        final_epoch = epoch
        if bad >= config["early_stopping"]["patience"]:
            logger.info("Early stopping triggered | epoch=%03d | bad_epochs=%d", epoch, bad)
            break
    if writer is not None:
        writer.close()
    logger.info("Training completed | seed=%s | best_epoch=%s | best_val_nll=%.9g | final_epoch=%s | total_time=%.1fs",
                seed, best_epoch, best, final_epoch, time.monotonic()-overall_start)
    logger.info("Artifacts | best_checkpoint=%s | last_checkpoint=%s | tensorboard=%s | log=%s",
                best_path, last_path, run_dir / "tensorboard" if tensorboard else "disabled", log_path)
    return model, best
