"""Class-aware detection metrics for the side-by-side vision experiment.

Both inputs use numeric ``[x1, y1, x2, y2, confidence, class_id]`` rows.
Coordinates describe continuous image rectangles (no inclusive-pixel ``+1``).
Callers must apply the same visible-object/ROI policy to ground truth before
scoring either detector. Ground truth is used for evaluation only.
"""

from __future__ import annotations

import numpy as np


def _rows(values, name):
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric N x 6 array") from error
    if result.ndim == 1 and result.size == 0:
        result = result.reshape(0, 6)
    if result.ndim != 2 or result.shape[1] != 6:
        raise ValueError(f"{name} must be a numeric N x 6 array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(result[:, 2:4] <= result[:, :2]):
        raise ValueError(f"{name} boxes must have positive width and height")
    if np.any((result[:, 4] < 0) | (result[:, 4] > 1)):
        raise ValueError(f"{name} confidence must be in [0, 1]")
    classes = result[:, 5]
    if np.any((classes < 0) | (classes != np.floor(classes))):
        raise ValueError(f"{name} class_id must be a nonnegative integer")
    return result


def _ious(box, candidates):
    intersection_wh = np.maximum(
        0.0, np.minimum(box[2:4], candidates[:, 2:4])
        - np.maximum(box[:2], candidates[:, :2]),
    )
    intersection = np.prod(intersection_wh, axis=1)
    area = np.prod(box[2:4] - box[:2])
    candidate_areas = np.prod(candidates[:, 2:4] - candidates[:, :2], axis=1)
    return intersection / (area + candidate_areas - intersection)


def score_detections(predictions, truths, iou_threshold=0.5):
    """Return ``tp, fp, fn, precision, recall`` for one image.

    Predictions are matched in descending confidence order to the best
    unmatched ground-truth box of the same class. A match includes equality
    at ``iou_threshold``; one truth can never satisfy multiple predictions.
    ``iou_threshold`` must be in ``(0, 1]``. An empty denominator produces 0.0,
    so an empty scene does not report artificial perfect detection accuracy.

    These are image-level detection counts, not tracking IDs or mAP. Summed
    counts, rather than the mean of frame percentages, give stream precision
    and recall without giving empty frames extra weight.
    """
    try:
        iou_threshold = float(iou_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("iou_threshold must be finite and in (0, 1]") from error
    if not np.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be finite and in (0, 1]")
    predictions = _rows(predictions, "predictions")
    truths = _rows(truths, "truths")
    unmatched = np.ones(len(truths), dtype=bool)
    true_positives = 0
    for index in np.argsort(-predictions[:, 4], kind="stable"):
        prediction = predictions[index]
        eligible = np.flatnonzero(unmatched & (truths[:, 5] == prediction[5]))
        if not len(eligible):
            continue
        overlaps = _ious(prediction, truths[eligible])
        best = int(np.argmax(overlaps))
        if overlaps[best] >= iou_threshold:
            unmatched[eligible[best]] = False
            true_positives += 1
    false_positives = len(predictions) - true_positives
    false_negatives = len(truths) - true_positives
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": true_positives / len(predictions) if len(predictions) else 0.0,
        "recall": true_positives / len(truths) if len(truths) else 0.0,
    }
