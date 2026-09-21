"""Evaluation of frozen predictions, including coverage."""
import numpy as np


def _auroc(y, s):
    """Mann-Whitney AUC with average ranks for exact ties."""
    order = np.argsort(s, kind="stable")
    ranked = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and s[order[end]] == s[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + 1 + end) / 2
        start = end
    positives = y == 1
    p, n = int(positives.sum()), int((~positives).sum())
    return float((ranked[positives].sum() - p * (p+1)/2) / (p*n))


def _average_precision(y, s):
    order = np.argsort(-s, kind="stable")
    labels = y[order] == 1
    sorted_scores = s[order]
    total_positive = int(labels.sum())
    seen_positive, area, start = 0, 0.0, 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        newly_positive = int(labels[start:end].sum())
        seen_positive += newly_positive
        area += (newly_positive / total_positive) * (seen_positive / end)
        start = end
    return float(area)


def binary_metrics(labels, scores, tau):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or not np.isfinite(s).all() or not set(y).issubset({0, 1}):
        raise ValueError("invalid labels or scores")
    pred = s > tau
    real, ai = y == 0, y == 1
    fpr = float(pred[real].mean()) if real.any() else None
    tpr = float(pred[ai].mean()) if ai.any() else None
    both = real.any() and ai.any()
    return dict(n_real=int(real.sum()), n_ai=int(ai.sum()), auroc=_auroc(y, s) if both else None,
                ap=_average_precision(y, s) if both else None, fpr=fpr, tpr=tpr,
                balanced_accuracy=(tpr + 1 - fpr) / 2 if both else None)


def genimage_metrics(rows, tau):
    valid = [r for r in rows if r.get("status") == "ok"]
    def calculate(items, all_items):
        result = binary_metrics([int(r["label"]) for r in items], [float(r["score"]) for r in items], tau)
        result.update(coverage=len(items) / len(all_items) if all_items else 0, num_errors=len(all_items)-len(items))
        return result
    generators = sorted({r["generator"] for r in rows})
    by_generator = {g: calculate([r for r in valid if r["generator"] == g],
                                 [r for r in rows if r["generator"] == g]) for g in generators}
    overall = calculate(valid, rows)
    aucs = [m["auroc"] for m in by_generator.values() if m["auroc"] is not None]
    overall["macro_auroc"] = float(np.mean(aucs)) if aucs else None
    groups = {}
    for value in (True, False):
        group = [r for r in valid if str(r.get("upscaled", "")).lower() == str(value).lower()]
        groups[str(value).lower()] = binary_metrics([int(r["label"]) for r in group],
                                                   [float(r["score"]) for r in group], tau) if group else None
    return dict(overall=overall, generators=by_generator, upscaled=groups)


def coco_metrics(rows, tau):
    valid = [r for r in rows if r.get("status") == "ok"]
    if not valid:
        return dict(n=0, fpr=None, score_mean=None, score_std=None, score_median=None,
                    score_quantiles={}, coverage=0.0, num_errors=len(rows), upscaled_fpr={})
    scores = np.array([float(r["score"]) for r in valid])
    return dict(n=len(valid), fpr=float(np.mean(scores > tau)), score_mean=float(scores.mean()),
                score_std=float(scores.std(ddof=0)), score_median=float(np.median(scores)),
                score_quantiles={str(q): float(np.quantile(scores, q)) for q in (0.05, 0.5, 0.95)},
                coverage=len(valid)/len(rows), num_errors=len(rows)-len(valid),
                upscaled_fpr={str(v).lower(): float(np.mean([float(r["score"]) > tau for r in valid if str(r.get("upscaled", "")).lower() == str(v).lower()]))
                              for v in (True, False) if any(str(r.get("upscaled", "")).lower() == str(v).lower() for r in valid)})
