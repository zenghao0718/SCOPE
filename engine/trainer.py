"""Image-weighted deterministic training and validation."""
import random
import time
import numpy as np
import torch
from features.standardization import transform
from losses.mdn_nll import patch_nll
from engine.seed import set_seed, epoch_order
from engine.checkpoint import save_checkpoint, load_checkpoint
from engine.logging import append_metrics
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


def train(c_train, r_train, c_val, r_val, standardizer, config, seed, run_dir, *, device="cpu", resume=False, val_manifest_hash="", tensorboard=False):
    from pathlib import Path
    run_dir = Path(run_dir)
    writer = None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir / "tensorboard"))
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
    for epoch in range(start, tc["max_epochs"] + 1):
        tick = time.monotonic()
        model.train()
        total = 0.0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip_norm"])
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite parameter")
            total += float(loss.detach()) * len(indexes)
        train_nll = total / len(ct)
        val_nll = validate(model, cv, rv, batch_size, device)
        if val_nll < best:
            best = val_nll
            save_checkpoint(run_dir / "checkpoints" / "best.pt", dict(model_state=model.state_dict(), epoch=epoch,
                            val_nll=val_nll, seed=seed, protocol_id=config["protocol"]["id"],
                            feature_protocol_hash=standardizer["feature_protocol_hash"],
                            standardizer_hash=standardizer["artifact_sha256"], config=config))
        if val_nll < reference - config["early_stopping"]["min_delta"]:
            reference, bad = val_nll, 0
        else:
            bad += 1
        scheduler.step(val_nll)
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
                       learning_rate=optimizer.param_groups[0]["lr"], best_value=best, stop_reference=reference,
                       bad_epochs=bad, elapsed_seconds=time.monotonic()-tick))
        with (run_dir / "logs" / "train.log").open("a", encoding="utf-8") as stream:
            stream.write(f"epoch={epoch} train_nll={train_nll:.9g} val_nll={val_nll:.9g} "
                         f"lr={optimizer.param_groups[0]['lr']:.9g} best={best:.9g} bad_epochs={bad}\n")
        if writer is not None:
            writer.add_scalar("nll/train", train_nll, epoch)
            writer.add_scalar("nll/validation", val_nll, epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        if bad >= config["early_stopping"]["patience"]:
            break
    if writer is not None:
        writer.close()
    return model, best
