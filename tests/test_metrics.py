from evaluation.metrics import binary_metrics, genimage_metrics


def test_exact_metrics():
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.3)
    assert result["auroc"] == 0.75
    assert result["fpr"] == 0.5 and result["tpr"] == 1
    rows = [dict(label=y, score=s, generator="g", status="ok", upscaled=False)
            for y, s in zip([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])]
    assert genimage_metrics(rows, 0.3)["overall"]["macro_auroc"] == 0.75


def test_ties_and_error_coverage():
    tied = binary_metrics([0, 1], [0.5, 0.5], 0.5)
    assert tied["auroc"] == 0.5 and tied["ap"] == 0.5
    rows = [dict(label=0, score="", generator="g", status="decode_error")]
    result = genimage_metrics(rows, 0.5)
    assert result["overall"]["coverage"] == 0
    assert result["overall"]["num_errors"] == 1
