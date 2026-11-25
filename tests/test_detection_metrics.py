import math

import pytest

from src.evaluation import evaluate_detection

CLASSES = ["aphid", "whitefly"]


def _gt():
    return [
        {"boxes": [[10, 10, 50, 50], [100, 100, 140, 140]], "labels": [0, 1]},
        {"boxes": [[20, 20, 60, 60]], "labels": [0]},
    ]


def test_perfect_predictions_give_map_one():
    gts = _gt()
    preds = [
        {"boxes": g["boxes"], "scores": [0.99] * len(g["boxes"]), "labels": g["labels"]}
        for g in gts
    ]
    m = evaluate_detection(gts, preds, CLASSES)
    assert m["map_50"] == pytest.approx(1.0)
    assert m["map_50_95"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["ap50_aphid"] == pytest.approx(1.0)
    assert m["ap50_whitefly"] == pytest.approx(1.0)


def test_empty_predictions_give_zero():
    gts = _gt()
    preds = [{"boxes": [], "scores": [], "labels": []} for _ in gts]
    m = evaluate_detection(gts, preds, CLASSES)
    assert m["map_50"] == 0.0
    assert m["map_50_95"] == 0.0
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0


def test_wrong_class_is_not_a_match():
    gts = [{"boxes": [[10, 10, 50, 50]], "labels": [0]}]
    preds = [{"boxes": [[10, 10, 50, 50]], "scores": [0.9], "labels": [1]}]
    m = evaluate_detection(gts, preds, CLASSES)
    assert m["map_50"] == pytest.approx(0.0)
    assert m["precision"] == 0.0 and m["recall"] == 0.0


def test_half_of_boxes_found():
    gts = _gt()  # 3 GT boxes
    preds = [
        {"boxes": [[10, 10, 50, 50]], "scores": [0.9], "labels": [0]},  # finds 1
        {"boxes": [], "scores": [], "labels": []},
    ]
    m = evaluate_detection(gts, preds, CLASSES)
    assert m["recall"] == pytest.approx(1 / 3)
    assert m["precision"] == pytest.approx(1.0)


def test_low_confidence_boxes_ignored_for_precision_recall():
    gts = [{"boxes": [[10, 10, 50, 50]], "labels": [0]}]
    preds = [{"boxes": [[10, 10, 50, 50]], "scores": [0.10], "labels": [0]}]
    m = evaluate_detection(gts, preds, CLASSES, conf_threshold=0.25)
    assert m["recall"] == 0.0
    assert m["map_50"] == pytest.approx(1.0)  # mAP does not depend on the threshold


def test_false_positive_lowers_precision():
    gts = [{"boxes": [[10, 10, 50, 50]], "labels": [0]}]
    preds = [{
        "boxes": [[10, 10, 50, 50], [200, 200, 240, 240]],
        "scores": [0.9, 0.8], "labels": [0, 0],
    }]
    m = evaluate_detection(gts, preds, CLASSES)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)


def test_small_medium_large_buckets():
    gts = [{"boxes": [[0, 0, 10, 10]], "labels": [0]}]  # area 100 -> small
    preds = [{"boxes": [[0, 0, 10, 10]], "scores": [0.9], "labels": [0]}]
    m = evaluate_detection(gts, preds, CLASSES)
    assert m["map_small"] == pytest.approx(1.0)
    assert math.isnan(m["map_large"])  # no large objects exist


def test_no_ground_truth_raises():
    gts = [{"boxes": [], "labels": []}]
    preds = [{"boxes": [], "scores": [], "labels": []}]
    with pytest.raises(ValueError):
        evaluate_detection(gts, preds, CLASSES)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_detection(_gt(), [], CLASSES)
