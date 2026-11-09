"""Fast tests for honest same-frame benchmark/demo boundaries; no model downloads."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from vision import SimColorVision
from visual_test import (
    ColorThresholdDetector, RealYoloDetector, compose_panel, perturb_rgb,
    run_visual_test,
)


class _Tensor:
    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.rows


class VisualDemoTests(unittest.TestCase):
    def test_color_adapter_calls_original_contour_detector(self):
        rgb = np.zeros((160, 240, 3), dtype=np.uint8)
        rgb[20:60, 20:70] = (240, 20, 10)
        rgb[80:130, 150:210] = (10, 40, 240)
        original_pixels = rgb.copy()
        backend = SimColorVision.__new__(SimColorVision)
        adapter = ColorThresholdDetector.__new__(ColorThresholdDetector)
        adapter.backend = backend
        original_method = SimColorVision.detect_colors
        with patch.object(SimColorVision, "detect_colors", autospec=True,
                          side_effect=original_method) as detection:
            rows = adapter.predict(rgb)
        self.assertEqual(detection.call_count, 1)
        self.assertIs(detection.call_args.args[0], backend)
        self.assertIs(detection.call_args.args[1], rgb)
        np.testing.assert_array_equal(rows, [[20, 20, 70, 60, 1, 0],
                                             [150, 80, 210, 130, 1, 1]])
        np.testing.assert_array_equal(rgb, original_pixels)

    def test_common_exposure_is_deterministic_and_preserves_source(self):
        rgb = np.full((40, 60, 3), (180, 30, 20), dtype=np.uint8)
        original = rgb.copy()
        for scenario in ("clean", "crowded", "occluded", "dim", "glare"):
            with self.subTest(scenario=scenario):
                first, settings = perturb_rgb(rgb, scenario, 1.25, seed=7, frame_index=2)
                again, repeated = perturb_rgb(rgb, scenario, 1.25, seed=7, frame_index=2)
                np.testing.assert_array_equal(first, again)
                np.testing.assert_array_equal(rgb, original)
                self.assertEqual(settings, repeated)
                self.assertFalse(np.shares_memory(first, rgb))
        dim, _ = perturb_rgb(rgb, "dim", 1.25, seed=7, frame_index=2)
        other_noise, _ = perturb_rgb(rgb, "dim", 1.25, seed=7, frame_index=3)
        self.assertFalse(np.array_equal(dim, other_noise))
        clean, _ = perturb_rgb(rgb, "clean", 1.25)
        np.testing.assert_array_equal(clean, rgb)

    def test_mock_run_never_reports_model_precision_or_recall(self):
        source = np.full((320, 480, 3), (100, 40, 20), dtype=np.uint8)
        originals = source.copy()
        received = []
        scenes = []

        class FakeScene:
            def __init__(self, **kwargs):
                self.env = object()
                self.closed = False
                scenes.append(self)

            def capture(self, timestamp, scenario):
                return SimpleNamespace(rgb=source)

            def truth(self, frame):
                return [dict(bbox=(10, 20, 40, 60), class_id=0, eligible=True),
                        dict(bbox=None, class_id=1, eligible=False)]

            def close(self):
                self.closed = True

        def color_predict(rgb):
            received.append(rgb.copy())
            return np.empty((0, 6))

        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(
                visual_duration=5., visual_period=1., visual_fps=1,
                visual_width=480, visual_height=320, visual_iou=.5,
                visual_confidence=.25, visual_record=None, visual_yolo="mock",
                visual_scenario="cycle", model="not-used.pt", save_dir=folder,
                headless=True, seed=7,
            )
            with patch("visual_test_scene.VisualStressScene", FakeScene), \
                 patch("visual_test.ColorThresholdDetector") as color_factory, \
                 patch("visual_test.RealYoloDetector", side_effect=AssertionError("mock loaded neural model")), \
                 patch("visual_test.cv2.imwrite", return_value=True), \
                 contextlib.redirect_stdout(io.StringIO()):
                color_factory.return_value.predict.side_effect = color_predict
                report = run_visual_test(args)
            saved = json.loads((Path(folder)/"metrics.json").read_text(encoding="utf-8"))
        self.assertFalse(report["right_is_neural_inference"])
        self.assertTrue(report["mock_uses_simulation_annotations"])
        self.assertIsNone(report["model"])
        self.assertEqual(len(saved["frames"]), 5)
        for frame in saved["frames"]:
            self.assertIsNone(frame["yolo_metrics"])
            self.assertEqual(frame["right_count"], 1)
            self.assertEqual(frame["visible_truth"], 1)
            self.assertEqual(frame["color_metrics"]["fn"], 1)
        for totals in saved["per_scenario"].values():
            self.assertIsNone(totals["yolo"])
        self.assertEqual(len(received), 5)
        np.testing.assert_array_equal(received[0], source)
        self.assertFalse(np.array_equal(received[3], source))
        np.testing.assert_array_equal(source, originals)
        self.assertTrue(scenes[0].closed)

    def test_mock_screen_disclaims_inference_and_model_accuracy(self):
        messages = []
        rgb = np.zeros((320, 800, 3), dtype=np.uint8)
        score = dict(precision=0., recall=0., fp=0, fn=1)
        with patch("visual_test._text", side_effect=lambda image, text, *args, **kwargs: messages.append(text)):
            panel = compose_panel(rgb, [], [[10, 10, 20, 20, 1, 0]], 1, score, None,
                                  "dim", 1., "mock", {"exposure": .16}, 0.)
        self.assertEqual(panel.shape, (572, 1606, 3))
        self.assertTrue(any("NOT NEURAL INFERENCE" in text for text in messages))
        self.assertTrue(any("Model accuracy: N/A" in text for text in messages))
        self.assertTrue(any("NOT a YOLO accuracy benchmark" in text for text in messages))

    def test_real_rejects_coco_names_and_requires_both_color_classes(self):
        for names in ({0: "person", 1: "bicycle"}, {0: "red_box"}):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as folder:
                weights = Path(folder)/"weights.pt"
                weights.write_bytes(b"test-weights")
                fake_ultralytics = SimpleNamespace(YOLO=Mock(return_value=SimpleNamespace(names=names)))
                with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                    with self.assertRaisesRegex(ValueError, "COCO.*cannot benchmark"):
                        RealYoloDetector(weights)

    def test_real_accepts_custom_names_and_only_sends_bgr_pixels_to_model(self):
        rgb = np.full((40, 60, 3), (240, 20, 10), dtype=np.uint8)
        original = rgb.copy()
        model = SimpleNamespace(
            names={3: "Box_A", 7: "blue_box", 9: "person"},
            predict=Mock(return_value=[SimpleNamespace(boxes=SimpleNamespace(data=_Tensor([
                [1, 2, 10, 20, .9, 3], [20, 2, 30, 20, .8, 7], [32, 2, 40, 20, .7, 9],
            ])))]),
        )
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder)/"weights.pt"
            weights.write_bytes(b"test-weights")
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=model))}):
                detector = RealYoloDetector(weights, confidence=.4)
                rows = detector.predict(rgb)
        np.testing.assert_array_equal(rows, [[1, 2, 10, 20, .9, 0], [20, 2, 30, 20, .8, 1]])
        self.assertEqual(set(model.predict.call_args.kwargs), {"source", "conf", "verbose"})
        np.testing.assert_array_equal(model.predict.call_args.kwargs["source"], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.assertEqual(model.predict.call_args.kwargs["conf"], .4)
        np.testing.assert_array_equal(rgb, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
