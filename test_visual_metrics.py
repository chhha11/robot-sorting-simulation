"""Detection benchmark accounting, independent of simulator identities."""

import unittest

import numpy as np

from visual_test_metrics import score_detections


class VisualMetricsTests(unittest.TestCase):
    def test_duplicates_wrong_class_and_missed_target_are_not_successes(self):
        truths = [[0, 0, 10, 10, 1, 0], [20, 0, 30, 10, 1, 1]]
        predictions = [
            [0, 0, 10, 10, .9, 0],
            [0, 0, 10, 10, .8, 0],
            [20, 0, 30, 10, .7, 0],
        ]
        score = score_detections(predictions, truths)
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 2, 1))
        self.assertAlmostEqual(score["precision"], 1 / 3)
        self.assertEqual(score["recall"], .5)

    def test_empty_frames_and_no_predictions(self):
        self.assertEqual(score_detections([], []), {
            "tp": 0, "fp": 0, "fn": 0, "precision": 0., "recall": 0.,
        })
        box = [[0, 0, 10, 10, 1, 0]]
        missing = score_detections(np.empty((0, 6)), box)
        false_alarm = score_detections(box, [])
        self.assertEqual((missing["tp"], missing["fp"], missing["fn"]), (0, 0, 1))
        self.assertEqual((false_alarm["tp"], false_alarm["fp"], false_alarm["fn"]), (0, 1, 0))

    def test_exact_iou_boundary_uses_continuous_rectangle_area(self):
        truth = [[0, 0, 3, 2, 1, 0]]
        prediction = [[1, 0, 4, 2, .8, 0]]  # intersection 4 / union 8
        self.assertEqual(score_detections(prediction, truth, .5)["tp"], 1)
        self.assertEqual(score_detections(prediction, truth, .50001)["tp"], 0)
        self.assertEqual(score_detections(truth, truth, 1)["tp"], 1)
        edge_touch = [[3, 0, 6, 2, .8, 0]]
        self.assertEqual(score_detections(edge_touch, truth, .00001)["tp"], 0)

    def test_highest_confidence_prediction_matches_first(self):
        truths = [[0, 0, 10, 10, 1, 0], [5, 0, 15, 10, 1, 0]]
        # The high-confidence box consumes the left truth. The lower box can
        # no longer reuse it even though that box's overlap would also pass.
        predictions = [[0, 0, 7, 10, .1, 0], [0, 0, 10, 10, .9, 0]]
        self.assertEqual(score_detections(predictions, truths)["tp"], 1)
        self.assertEqual(score_detections(predictions[::-1], truths)["tp"], 1)

    def test_best_unmatched_truth_and_class_matching(self):
        truths = [[0, 0, 10, 10, 1, 0], [5, 0, 15, 10, 1, 0],
                  [0, 0, 10, 10, 1, 1]]
        predictions = [[4, 0, 14, 10, .9, 0], [0, 0, 10, 10, .8, 0],
                       [0, 0, 10, 10, .7, 1]]
        self.assertEqual(score_detections(predictions, truths), {
            "tp": 3, "fp": 0, "fn": 0, "precision": 1., "recall": 1.,
        })

    def test_invalid_rows_and_nonfinite_inputs_fail_explicitly(self):
        bad_arrays = [
            [[0, 0, 1, 1, .5]], np.empty((0, 5)), [0, 0, 1, 1, .5, 0],
            [[0, 0, np.nan, 1, .5, 0]], [[0, 0, 1, np.inf, .5, 0]],
            [[1, 0, 0, 1, .5, 0]], [[0, 0, 0, 1, .5, 0]],
            [[0, 0, 1, 1, 1.1, 0]], [[0, 0, 1, 1, -.1, 0]],
            [[0, 0, 1, 1, .5, -1]], [[0, 0, 1, 1, .5, .5]],
        ]
        for values in bad_arrays:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    score_detections(values, [])
                with self.assertRaises(ValueError):
                    score_detections([], values)
        for threshold in (0, -1, 1.1, np.nan, np.inf, None):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                score_detections([], [], threshold)


if __name__ == "__main__":
    unittest.main(verbosity=2)
