"""Rendered RGB-D regressions for four placement platforms and belt-only IDs.

Fixture positions only create/project test proposals. Every detector receives
RGB and metric depth; neither body poses nor segmentation enter the pipeline.
The neural adapter uses injected proposals, not COCO claims about custom boxes.
"""

from types import SimpleNamespace
import unittest

import numpy as np
import pybullet as p

from env_setup import SimulationEnvironment
from vision import SimColorVision, YoloVision, create_vision


RED = (.95, .08, .03, 1)
BLUE = (.03, .18, .95, 1)


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Proposals:
    """Fixed image-space neural proposals; calibrated depth decides validity."""

    def __init__(self, camera, fixtures):
        view = np.asarray(camera.view_matrix).reshape(4, 4, order="F")
        projection = np.asarray(camera.projection_matrix).reshape(4, 4, order="F")
        boxes, classes = [], []
        for point, class_id in fixtures:
            clip = projection @ view @ np.r_[point, 1.0]
            ndc = clip[:3] / clip[3]
            u = (ndc[0] + 1) * camera.config.width / 2 - camera.pixel_offset[0]
            v = (1 - ndc[1]) * camera.config.height / 2 - camera.pixel_offset[1]
            boxes.append((u - 4, v - 4, u + 4, v + 4))
            classes.append(class_id)
        self.boxes = SimpleNamespace(xyxy=_Tensor(np.asarray(boxes).reshape(-1, 4)),
                                     conf=_Tensor(np.full(len(boxes), .9)),
                                     cls=_Tensor(classes))

    def predict(self, source, **options):
        return [SimpleNamespace(boxes=self.boxes,
                                names={0: "red_box", 1: "blue_box"})]


class DualStationVisionTests(unittest.TestCase):
    def setUp(self):
        self.env = SimulationEnvironment(gui=False)
        self.addCleanup(self.env.close)
        self.assertEqual(len(self.env.drop_zones), 4)

    def detectors(self, fixtures):
        color = SimColorVision(self.env, noise_std=8, lighting_jitter=.02, noise_seed=2026)
        mock = YoloVision(self.env, model=object(), noise_std=8,
                          lighting_jitter=.02, noise_seed=2026)
        neural = create_vision(self.env, detector="yolo-real")
        neural.model = _Proposals(self.env.camera, fixtures)
        self.assertEqual(color.pickup_bounds, mock.pickup_bounds)
        self.assertEqual(color.pickup_bounds, neural.pickup_bounds)
        return (color, mock, neural)

    def test_empty_outlined_platforms_and_bare_belt_do_not_start_tracks(self):
        self.env.step(120)
        fixtures = [(zone.surface_position, index % 2)
                    for index, zone in enumerate(self.env.drop_zones)]
        # Even a confident neural proposal on the bare belt fails the height band.
        fixtures.append(((1.65, 0, self.env.conveyor_top_z), 0))
        rgb, depth = self.env.camera.capture_rgbd()
        for detector in self.detectors(fixtures):
            with self.subTest(detector=type(detector).__name__):
                for frame in range(3):
                    self.assertEqual(detector.detect(rgb, depth, timestamp=frame / 20), [])
                self.assertEqual(detector._tracker.next_track_id, 1)
                self.assertFalse(detector._tracker.tracks)

    def test_placed_boxes_are_excluded_and_outer_belt_ids_remain_stable(self):
        fixtures = []
        for zone in self.env.drop_zones:
            class_id = int("Blue" in zone.name)
            x, y, z = zone.surface_position
            self.env.create_test_object((x, y, z + .05), shape="box",
                                        color=(RED, BLUE)[class_id])
            fixtures.append(((x, y, z + .10), class_id))

        # Both positions are beyond the old hard-coded [-1.40, 1.48] ROI.
        # They are still on the actual belt and must receive the first IDs.
        expected = ((-1.65, -.07, "Box_A"), (1.65, .07, "Box_B"))
        for class_id, (x, y, _) in enumerate(expected):
            self.env.create_test_object((x, y, self.env.conveyor_top_z + .05),
                                        shape="box", color=(RED, BLUE)[class_id])
            fixtures.append(((x, y, self.env.conveyor_top_z + .10), class_id))
        self.env.step(120)
        rgb, depth = self.env.camera.capture_rgbd()

        for detector in self.detectors(fixtures):
            with self.subTest(detector=type(detector).__name__):
                ids = {}
                for frame in range(20):
                    results = detector.detect(rgb, depth, timestamp=frame / 20)
                    self.assertEqual(len(results), 2)
                    self.assertEqual({item["track_id"] for item in results}, {1, 2})
                    for item in results:
                        x, y, label = next(p for p in expected if p[2] == item["class"])
                        self.assertLess(np.linalg.norm(np.asarray(item["pos_3d"][:2]) - (x, y)), .025)
                        ids.setdefault(label, item["track_id"])
                        self.assertEqual(ids[label], item["track_id"])
                self.assertEqual(detector._tracker.next_track_id, 3)

    def test_colored_object_above_conveyor_height_is_rejected(self):
        self.env.step(120)
        point = (1.65, 0, self.env.conveyor_top_z + .30)
        self.env.create_test_object(point, shape="box", color=RED)
        # Capture immediately: this is a held/elevated observation, not a drop.
        rgb, depth = self.env.camera.capture_rgbd()
        fixtures = [((point[0], point[1], point[2] + .05), 0)]
        for detector in self.detectors(fixtures):
            with self.subTest(detector=type(detector).__name__):
                self.assertEqual(detector.detect(rgb, depth, timestamp=0), [])
                self.assertEqual(detector._tracker.next_track_id, 1)

    def test_pickup_entry_and_home_projections_remain_visible(self):
        # A nearly vertical camera hid the entry behind the parked wrist.
        # Exercise both colors at entry edges and below both standby tools.
        layout = self.env.layout
        positions = [(layout.spawn_x, y) for y in (*layout.spawn_y_range, 0)]
        positions.extend((home[0], 0) for home in layout.home_tcp_positions)
        positions.append((0, 0))
        detector = YoloVision(self.env, model=object())
        for x, y in positions:
            for class_id, color in enumerate((RED, BLUE)):
                with self.subTest(position=(x, y), color=class_id):
                    body = self.env.create_test_object(
                        (x, y, self.env.conveyor_top_z + .05), shape="box", color=color)
                    try:
                        rgb, depth = self.env.camera.capture_rgbd()
                        results = detector.detect(rgb, depth, timestamp=0)
                        self.assertEqual(len(results), 1)
                        self.assertEqual(results[0]["class"], ("Box_A", "Box_B")[class_id])
                        error = np.linalg.norm(np.asarray(results[0]["pos_3d"][:2]) - (x, y))
                        self.assertLess(error, .035)
                    finally:
                        p.removeBody(body, physicsClientId=self.env.client_id)

    def test_adjacent_same_color_side_faces_do_not_merge_visual_ids(self):
        # At this oblique camera angle, the side faces of two 20 cm-spaced
        # blue boxes touch in RGB. Their measured top faces are separate.
        expected = ((-.304, -.0602), (-.104, -.0648))
        for x, y in expected:
            self.env.create_test_object((x, y, self.env.conveyor_top_z + .05),
                                        shape="box", color=BLUE)
        self.env.step(120)
        rgb, depth = self.env.camera.capture_rgbd()
        detectors = (SimColorVision(self.env, noise_std=10, lighting_jitter=.04, noise_seed=2026),
                     YoloVision(self.env, model=object(), noise_std=10,
                                lighting_jitter=.04, noise_seed=2026))
        for detector in detectors:
            with self.subTest(detector=type(detector).__name__):
                identities = {}
                for frame in range(20):
                    results = detector.detect(rgb, depth, timestamp=frame / 20)
                    self.assertEqual(len(results), 2)
                    self.assertEqual({item["track_id"] for item in results}, {1, 2})
                    for item in results:
                        index = min(range(2), key=lambda i: np.linalg.norm(
                            np.asarray(item["pos_3d"][:2]) - expected[i]))
                        self.assertLess(np.linalg.norm(
                            np.asarray(item["pos_3d"][:2]) - expected[index]), .025)
                        identities.setdefault(index, item["track_id"])
                        self.assertEqual(identities[index], item["track_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
