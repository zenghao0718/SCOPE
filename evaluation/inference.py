import numpy as np
import torch
from features.standardization import transform
from losses.mdn_nll import patch_nll
from data.decoding import decode_image, resize_if_needed
from data.patches import extract_patches
from features.cr_features import extract_cr


@torch.no_grad()
def image_scores(model, c_raw, r_raw, standardizer, device="cpu", batch_size=256):
    c, r = transform(c_raw, r_raw, standardizer)
    if c.shape[1:] != (4, 6) or r.shape[1:] != (4, 8):
        raise ValueError("one image must have four C/R pairs")
    model.eval()
    scores = []
    for start in range(0, len(c), batch_size):
        cb = torch.from_numpy(c[start:start+batch_size].reshape(-1, 6)).to(device)
        rb = torch.from_numpy(r[start:start+batch_size].reshape(-1, 8)).to(device)
        nll = patch_nll(rb, *model(cb)).reshape(-1, 4).mean(dim=1)
        if not torch.isfinite(nll).all():
            raise FloatingPointError("nonfinite image score")
        scores.extend(nll.cpu().double().tolist())
    return np.asarray(scores, dtype=np.float64)


@torch.no_grad()
def score_cached_features(model, c_raw, r_raw, standardizer, tau, device="cpu"):
    c, r = transform(np.asarray(c_raw)[None], np.asarray(r_raw)[None], standardizer)
    cb = torch.from_numpy(c.reshape(4, 6)).to(device)
    rb = torch.from_numpy(r.reshape(4, 8)).to(device)
    model.eval()
    patches = patch_nll(rb, *model(cb)).cpu().double().numpy()
    if not np.isfinite(patches).all():
        raise FloatingPointError("nonfinite patch score")
    score = float(patches.mean())
    return dict(patch_nll=patches.tolist(), image_score=score, prediction=int(score > tau), threshold=tau)


def score_image(model, source, standardizer, tau, device="cpu"):
    decoded = decode_image(source)
    if decoded.status != "ok":
        return dict(status=decoded.status, error_type=decoded.status)
    rgb, info = resize_if_needed(decoded.rgb)
    patches, coords, unique = extract_patches(rgb.astype(np.float64) / 255)
    values = [extract_cr(p) for p in patches]
    result = score_cached_features(model, np.stack([v[0] for v in values]),
                                   np.stack([v[1] for v in values]), standardizer, tau, device)
    return dict(result, status="ok", upscaled=info["upscaled"], patch_yx=coords.tolist(),
                num_unique_patches=unique)
