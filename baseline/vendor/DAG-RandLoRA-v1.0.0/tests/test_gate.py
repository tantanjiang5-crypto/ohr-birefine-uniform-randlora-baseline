from randlora_damage import ExperimentMetrics, evaluate_experiment_gate


def metric(**kwargs):
    base = dict(
        d_ap75=0.20,
        low_occupancy_mean_iou=0.40,
        classification_accuracy=0.80,
        box_background_fpr=0.10,
        oversegmentation_ratio=1.0,
        classification_metric=0.75,
    )
    base.update(kwargs)
    return ExperimentMetrics(**base)


def test_gate_pass_and_stop_rules():
    baseline = metric()
    passed = evaluate_experiment_gate(
        baseline,
        metric(d_ap75=0.211, classification_accuracy=0.796, box_background_fpr=0.09),
        epoch=20,
    )
    assert passed.pass_gate
    assert not passed.stop_early

    stopped = evaluate_experiment_gate(
        baseline,
        metric(
            d_ap75=0.202,
            low_occupancy_mean_iou=0.39,
            oversegmentation_ratio=1.06,
            classification_metric=0.77,
        ),
        epoch=15,
    )
    assert stopped.stop_early
