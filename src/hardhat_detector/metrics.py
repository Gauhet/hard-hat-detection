"""Evaluation helpers that Ultralytics doesn't provide: recall by object size, and how well the
detector works as a "someone here has no hard hat" alarm for a whole photo."""

import numpy as np

# COCO's size buckets, by box area in pixels of the original photo.
SIZE_BUCKETS = (("small", 0, 32**2), ("medium", 32**2, 96**2), ("large", 96**2, float("inf")))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection-over-union of two sets of (x1, y1, x2, y2) boxes: shape (len(a), len(b))."""
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection)


def match_ground_truth(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Which ground-truth boxes were found. Predictions are taken from most to least confident,
    and each one claims the unclaimed same-class box it overlaps most, if that overlap is at
    least `iou_threshold`. This is the matching rule mAP uses, at a single threshold."""
    found = np.zeros(len(gt_boxes), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return found
    iou = box_iou(pred_boxes, gt_boxes)
    iou[pred_classes[:, None] != gt_classes[None, :]] = 0
    for i in np.argsort(-pred_scores):
        candidates = np.where(found, 0, iou[i])
        best = candidates.argmax()
        if candidates[best] >= iou_threshold:
            found[best] = True
    return found


def size_bucket(box: np.ndarray) -> str:
    area = (box[2] - box[0]) * (box[3] - box[1])
    return next(name for name, low, high in SIZE_BUCKETS if low <= area < high)


def recall_by_size(gt_boxes: list[np.ndarray], found: list[np.ndarray]) -> dict[str, dict]:
    """Share of ground-truth boxes found, per size bucket, over a whole set of photos."""
    totals = {name: [0, 0] for name, _, _ in SIZE_BUCKETS}
    for boxes, hits in zip(gt_boxes, found):
        for box, hit in zip(boxes, hits):
            totals[size_bucket(box)][0] += 1
            totals[size_bucket(box)][1] += int(hit)
    return {name: {"boxes": n, "recall": hits / n if n else None} for name, (n, hits) in totals.items()}


def alarm_metrics(truth: list[bool], raised: list[bool]) -> dict:
    """Treat the detector as a per-photo alarm: `truth` says whether a photo really shows a
    bare head, `raised` whether the detector found one."""
    truth, raised = np.asarray(truth, dtype=bool), np.asarray(raised, dtype=bool)
    tp = int((truth & raised).sum())
    fp = int((~truth & raised).sum())
    fn = int((truth & ~raised).sum())
    return {
        "photos": len(truth),
        "photos_with_bare_head": int(truth.sum()),
        "true_alarms": tp,
        "false_alarms": fp,
        "missed": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }
