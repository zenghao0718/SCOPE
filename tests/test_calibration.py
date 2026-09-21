from evaluation.calibration import threshold, predict


def test_higher_and_strict_threshold():
    tau = threshold([0, 1, 2, 3], 0.5)
    assert tau == 2 and predict(2, tau) == 0 and predict(3, tau) == 1
