"""Phase-two regressions: actual RGB-D mock boxes and jittered associations."""
import unittest

import numpy as np

from calibration import RGBDCalibration
from env_setup import SimulationEnvironment
from vision import Detection, YoloVision, _WorldTracker


class YoloTrackingTests(unittest.TestCase):
    def test_calibrated_camera_pose(self):
        pose = np.eye(4)
        pose[:3, 3] = (1, 2, 3)
        calibration = RGBDCalibration([[100, 0, 50], [0, 100, 40], [0, 0, 1]], pose)
        np.testing.assert_allclose(calibration.pixel_to_world((60, 60), 2), (1.2, 2.4, 5))
        with self.assertRaises(ValueError):
            calibration.pixel_to_world((60, 60), 0)

    def test_multiple_rgbd_boxes_have_stable_ids_and_literal_box_centres(self):
        env = SimulationEnvironment(gui=False)
        try:
            expected = []
            for label, x, y, color in (
                ('Box_A', -.11, -.06, (.95, .08, .03, 1)),
                ('Box_A', .11, -.06, (.95, .08, .03, 1)),
                ('Box_B', -.11, .06, (.03, .18, .95, 1)),
                ('Box_B', .11, .06, (.03, .18, .95, 1)),
            ):
                env.create_test_object((x, y, env.conveyor_top_z + .05), shape='box', color=color)
                expected.append((label, np.array([x, y])))
            env.step(120)
            vision = YoloVision(env, noise_std=24, lighting_jitter=.08, noise_seed=2026)
            rgb, depth = vision.get_camera_data()
            identities = {}
            for frame in range(80):
                items = vision.detect(rgb, depth, timestamp=frame / 20)
                self.assertEqual(len(items), 4)
                self.assertEqual(vision.last_yolo_output.shape[1], 6)
                self.assertTrue(np.all(np.isfinite(vision.last_yolo_output)))
                self.assertTrue(np.all((vision.last_yolo_output[:, 4] >= 0) &
                                       (vision.last_yolo_output[:, 4] <= 1)))
                self.assertTrue(set(vision.last_yolo_output[:, 5]) <= {0, 1})
                self.assertEqual(len({item['track_id'] for item in items}), 4)
                for item in items:
                    index = min(range(4), key=lambda i: np.linalg.norm(item['pos_3d'][:2] - expected[i][1]))
                    self.assertEqual(item['class'], expected[index][0])
                    self.assertLess(np.linalg.norm(item['pos_3d'][:2] - expected[index][1]), .02)
                    identities.setdefault(index, item['track_id'])
                    self.assertEqual(identities[index], item['track_id'])
                    x1, y1, x2, y2 = item['bbox']
                    self.assertEqual(item['pixel_center'], ((x1+x2)/2, (y1+y2)/2))
                    np.testing.assert_allclose(item['pos_3d'], vision.pixel_to_world(
                        item['pixel_center'], float(np.median(depth[
                            round(item['pixel_center'][1])-2:round(item['pixel_center'][1])+3,
                            round(item['pixel_center'][0])-2:round(item['pixel_center'][0])+3])),
                        (depth.shape[1], depth.shape[0])))
            self.assertEqual(set(identities.values()), {1, 2, 3, 4})
            self.assertEqual(vision.detect(np.zeros_like(rgb), depth, timestamp=4), [])
            self.assertEqual(vision.last_yolo_output.shape, (0, 6))
            self.assertEqual({i['track_id'] for i in vision.detect(rgb, depth, timestamp=5)}, {1, 2, 3, 4})
        finally:
            env.close()

    def test_completed_ids_stay_deduplicated_after_history_eviction(self):
        import contextlib
        import io
        from scheduler import ArmRoute, ConveyorScheduler, DetectedObject

        class Arm:
            idle = True
            last_error = None
            calls = 0
            is_tracking_target = True

            def get_tcp_position(self):
                return (0, 0, .52)

            def is_idle(self):
                return self.idle

            def estimate_pick_seconds(self, target):
                return 0

            def assign_task(self, *args, **kwargs):
                self.calls += 1
                self.idle = False

            def update_tracking_observation(self, *args):
                pass

        arm = Arm()
        scheduler = ConveyorScheduler({'Box_A': ArmRoute('arm1', arm, ((0, 0, .52),))})
        with contextlib.redirect_stdout(io.StringIO()):
            for track_id in range(1, 515):
                item = DetectedObject(track_id, 'Box_A', (0, 0, .52))
                self.assertEqual(len(scheduler.update([item, item], track_id * 2)), 1)
                self.assertEqual(scheduler.update([item], track_id * 2), ())
                arm.idle = True
                scheduler.tick(track_id * 2 + 1)
                task, = scheduler.drain_finished()
                scheduler.retire(task, True)
            self.assertNotIn(1, scheduler._terminal_ids)
            self.assertEqual(scheduler.update([DetectedObject(1, 'Box_A', (0, 0, .52))], 1030), ())
            self.assertFalse(scheduler.pending_tasks)
            self.assertEqual(arm.calls, 514)

    def test_shuffled_same_class_motion_jitter_occlusion_and_new_object(self):
        tracker = _WorldTracker((.05, 0, 0))
        rng = np.random.default_rng(2026)
        detection = Detection(0, 'Box_A', .9, (0, 0, 10, 10))
        identities = {}
        for frame in range(160):
            t = frame / 10
            indices = list(range(4 if frame < 80 else 5))
            rng.shuffle(indices)
            observations, visible = [], []
            for index in indices:
                if index == 1 and 30 <= frame < 60:
                    continue
                point = np.array([index * .22 + .05*t, 0, .52]) + rng.uniform(-.008, .008, 3)
                observations.append((detection, tuple(point)))
                visible.append(index)
            items = tracker.update(observations, t)
            self.assertEqual(len({i['track_id'] for i in items}), len(items))
            for index, item in zip(visible, items):
                identities.setdefault(index, item['track_id'])
                self.assertEqual(item['track_id'], identities[index])
        self.assertEqual(set(identities.values()), {1, 2, 3, 4, 5})
        self.assertEqual(tracker.next_track_id, 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
