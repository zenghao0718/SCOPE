# SCOPE project rules

- `docs/METHOD_SPEC.md` is the sole method authority. If it conflicts with `docs/IMPLEMENTATION_SPEC.md`, stop the affected implementation and report the conflict.
- Training, normalization, and checkpoint selection use real images only. `real_calibration` is used only for the threshold. GenImage is used only for final evaluation after freezing the model.
- Never select a seed, checkpoint, threshold, K, feature, or hyperparameter with test results.
- Never silently rewrite the formal configuration. Record all seeds, manifests, and feature caches for traceability. A feature protocol change invalidates old caches.
- Do not launch formal training locally. Before committing, run compile, pytest, and synthetic smoke.
