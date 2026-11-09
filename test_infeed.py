"""Colour-batch replay, balancing, and real RGB-D stream regressions."""
import collections
import contextlib
import io
import unittest

import numpy as np
import pybullet as p

from env_setup import SimulationEnvironment
from vision import YoloVision


class InfeedColourTests(unittest.TestCase):
    def setUp(self):
        # Colour policy does not require loading a physics world.
        self.env = SimulationEnvironment.__new__(SimulationEnvironment)
        self.env.conveyor_speed = .05
        self.env.simulation_step = 0
        self.env._infeed_colors = []

    def sequence(self, count=30, **options):
        self.env.enable_infeed(interval=(4, 4), **options)
        return [self.env._next_infeed_label() for _ in range(count)]

    def test_default_short_demo_switches_colour_for_every_seed(self):
        for seed in range(100):
            sequence = self.sequence(count=12, seed=seed)
            self.assertEqual(set(sequence[:4]), {'Box_A', 'Box_B'})
            lengths = []
            previous = None
            for label in sequence:
                if label != previous:
                    lengths.append(0)
                lengths[-1] += 1
                previous = label
            self.assertTrue(all(length in (1, 3) for length in lengths[:-1]))

    def test_probability_balance_and_random_initial_colour(self):
        self.env.enable_infeed(seed=2026, interval=(4, 4))
        sizes, labels, previous = [], [], None
        for _ in range(10000):
            label = self.env._next_infeed_label()
            count = 1 + len(self.env._infeed_colors)
            self.assertNotEqual(label, previous)
            self.assertTrue(all(self.env._next_infeed_label() == label for _ in range(count - 1)))
            sizes.append(count)
            labels.extend([label] * count)
            previous = label
        self.assertLess(abs(sizes.count(3) / len(sizes) - .60), .02)
        self.assertLess(abs(labels.count('Box_A') / len(labels) - .50), .03)
        firsts = [self.sequence(count=1, seed=seed)[0] for seed in range(1000)]
        self.assertLess(abs(firsts.count('Box_A') / len(firsts) - .50), .05)

    def test_reset_replays_and_motion_randomness_does_not_change_colours(self):
        expected = self.sequence(seed=7)
        self.env.enable_infeed(seed=7, interval=(4, 4))
        actual = []
        for index in range(30):
            for _ in range(index):
                self.env._infeed_rng.random()
            actual.append(self.env._next_infeed_label())
        self.assertEqual(expected, actual)
        self.assertEqual(expected, self.sequence(seed=7))

    def test_zero_probability_is_independent_random_not_alternating(self):
        sequence = self.sequence(count=10000, seed=7, cluster_probability=0)
        self.assertFalse(self.env._infeed_colors)
        self.assertLess(abs(sequence.count('Box_A') / len(sequence) - .50), .03)
        repeats = sum(a == b for a, b in zip(sequence, sequence[1:]))
        self.assertLess(abs(repeats / (len(sequence) - 1) - .50), .03)

    def test_forced_bursts_alternate_and_custom_lengths_work(self):
        sequence = self.sequence(count=18, seed=7, cluster_probability=1)
        for start in range(0, 18, 3):
            self.assertEqual(len(set(sequence[start:start+3])), 1)
            if start:
                self.assertNotEqual(sequence[start], sequence[start-1])
        self.env.enable_infeed(seed=7, interval=(4, 4), cluster_probability=1, cluster_size=(4, 5))
        for _ in range(30):
            self.env._next_infeed_label()
            remaining = len(self.env._infeed_colors)
            self.assertIn(remaining + 1, (4, 5))
            for _ in range(remaining):
                self.env._next_infeed_label()


class InfeedCameraTests(unittest.TestCase):
    def test_generated_red_blue_stream_has_stable_distinct_camera_ids(self):
        env = SimulationEnvironment(gui=False, conveyor_speed=.05)
        try:
            vision = YoloVision(env, conveyor_velocity=(.05, 0, 0),
                                noise_std=10, lighting_jitter=.04, noise_seed=2026)
            env.enable_infeed(seed=7, interval=(4, 4), limit=6)
            identities, observations, colours = {}, collections.Counter(), set()
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(44):
                    env.step(120)
                    detections = vision.detect(timestamp=env.simulation_step * env.time_step)
                    self.assertEqual(len({d['track_id'] for d in detections}), len(detections))
                    for detected in detections:
                        # Ground truth is used only here to validate image-derived IDs.
                        body = min(env.stream_objects, key=lambda candidate: np.linalg.norm(
                            np.asarray(p.getBasePositionAndOrientation(candidate)[0])[:2]
                            - detected['pos_3d'][:2]))
                        label, serial = env.stream_objects[body]
                        position = p.getBasePositionAndOrientation(body)[0]
                        self.assertLess(np.linalg.norm(np.asarray(position)[:2] - detected['pos_3d'][:2]), .025)
                        self.assertEqual(detected['class'], label)
                        identities.setdefault(serial, detected['track_id'])
                        self.assertEqual(identities[serial], detected['track_id'])
                        observations[serial] += 1
                        colours.add(label)
            self.assertEqual(colours, {'Box_A', 'Box_B'})
            self.assertEqual(set(identities), set(range(1, 7)))
            self.assertEqual(len(set(identities.values())), 6)
            self.assertGreaterEqual(min(observations.values()), 2)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
