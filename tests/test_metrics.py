import numpy as np
import pytest

from hardhat_detector.metrics import alarm_metrics, box_iou, match_ground_truth, recall_by_size, size_bucket


def test_iou_of_identical_disjoint_and_half_overlapping_boxes():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]], dtype=float)
    assert box_iou(a, b)[0] == pytest.approx([1.0, 0.0, 1 / 3])


def test_matching_needs_overlap_and_the_same_class():
    gt_boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    gt_classes = np.array([0, 1])
    preds = np.array([[1, 1, 10, 10], [50, 50, 60, 60]], dtype=float)
    found = match_ground_truth(preds, np.array([0.9, 0.8]), np.array([0, 0]), gt_boxes, gt_classes)
    # The second prediction is in the right place but says "helmet" for a bare head.
    assert found.tolist() == [True, False]


def test_each_ground_truth_box_is_claimed_at_most_once():
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=float)
    preds = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    found = match_ground_truth(preds, np.array([0.9, 0.8]), np.array([0, 0]), gt_boxes, np.array([0]))
    assert found.tolist() == [True]


def test_no_predictions_finds_nothing():
    found = match_ground_truth(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.array([[0, 0, 5, 5.0]]), np.array([1]))
    assert found.tolist() == [False]


def test_size_buckets_follow_coco_area_limits():
    assert size_bucket(np.array([0, 0, 31, 31])) == "small"
    assert size_bucket(np.array([0, 0, 32, 32])) == "medium"
    assert size_bucket(np.array([0, 0, 96, 96])) == "large"


def test_recall_by_size_counts_hits_per_bucket():
    boxes = [np.array([[0, 0, 10, 10], [0, 0, 200, 200]]), np.array([[0, 0, 12, 12]])]
    found = [np.array([True, False]), np.array([False])]
    result = recall_by_size(boxes, found)
    assert result["small"] == {"boxes": 2, "recall": 0.5}
    assert result["large"] == {"boxes": 1, "recall": 0.0}
    assert result["medium"] == {"boxes": 0, "recall": None}


def test_alarm_metrics():
    result = alarm_metrics(truth=[True, True, False, False], raised=[True, False, True, False])
    assert (result["true_alarms"], result["false_alarms"], result["missed"]) == (1, 1, 1)
    assert result["precision"] == 0.5 and result["recall"] == 0.5
