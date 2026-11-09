"""Independent tick updates and measured parallel RGB-D pick/place regression."""

import contextlib
import io
import json
from types import SimpleNamespace
import unittest

import pybullet as p
import numpy as np

from env_setup import CameraConfig, SimulationEnvironment
from main import (ParallelMotionMonitor, create_controllers,
                  simulation_attachment_id, update_arms)
from scheduler import TaskStatus
from vision import YoloVision


class TickUpdateTests(unittest.TestCase):
    def test_measured_zone_statistics_can_be_saved_as_json(self):
        class Arm:
            workspace_gate = SimpleNamespace(
                in_spatial_zone=lambda point: point[2] < 1.10 and np.linalg.norm(point[:2]) < .30)

            def __init__(self, point):
                self.point = point

            def get_measured_tcp_position(self):
                return self.point

            def is_idle(self):
                return False

        monitor = ParallelMotionMonitor([Arm((0, 0, .95)), Arm((.37, 0, 1.25))], 1/240)
        monitor.sample(1)
        saved = json.loads(json.dumps(monitor.summary()))
        self.assertEqual(saved["central_zone_max_occupancy"], 1)
        self.assertEqual(saved["central_zone_violation_steps"], 0)

    def test_waiting_or_busy_peer_never_suppresses_other_update(self):
        class Arm:
            def __init__(self, name, waiting=False):
                self.state = SimpleNamespace(name=name)
                self._workspace_waiting = waiting
                self.calls = 0

            def is_idle(self):
                return self.state.name == "IDLE"

            def update(self):
                self.calls += 1
                return self.state

        arms = [Arm("MOVE_TO_TARGET_ABOVE", True), Arm("IDLE")]
        previous = [None, None]
        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(100):
                update_arms(arms, step, 1 / 240, previous)
        self.assertEqual([arm.calls for arm in arms], [100, 100])

    def test_two_assigned_states_are_not_counted_as_parallel_motion(self):
        class Arm:
            _workspace_waiting = False
            point = (0, 0, 1)

            def get_measured_tcp_position(self):
                return self.point

            def is_idle(self):
                return False

        arms = [Arm(), Arm()]
        monitor = ParallelMotionMonitor(arms, 1 / 240)
        for step in range(120):
            arms[0].point = (step / 2400, 0, 1)
            monitor.sample(step)
        self.assertEqual(monitor.summary()["actual_parallel_seconds"], 0)


class PhysicalParallelMotionTests(unittest.TestCase):
    def test_different_visual_targets_move_together_and_reach_own_stations(self):
        self.exercise(moving=False)

    def test_moving_targets_do_not_deadlock_carry_against_peer_return(self):
        self.exercise(moving=True)

    def exercise(self, moving):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            env = SimulationEnvironment(gui=False,
                                        camera_config=CameraConfig(width=480, height=360),
                                        conveyor_speed=.05 if moving else 0)
            try:
                if moving:
                    env.enable_infeed(seed=7, interval=(6, 9), limit=2)
                    fixtures = env.stream_objects
                else:
                    fixtures = {
                        env.create_test_object((-.30, -.03, env.conveyor_top_z + .05),
                                               shape="box", color=(.95, .08, .03, 1)): "Box_A",
                        env.create_test_object((.30, .03, env.conveyor_top_z + .05),
                                               shape="box", color=(.03, .18, .95, 1)): "Box_B",
                    }
                arms, scheduler = create_controllers(env, moving=moving)
                workspace_gate = arms[0].workspace_gate
                vision = YoloVision(env, model=object(),
                                    conveyor_velocity=(env.conveyor_speed if moving else 0, 0, 0))
                monitor = ParallelMotionMonitor(arms, env.time_step)
                previous, bodies = [None] * len(arms), {}
                finished = []
                for _ in range(8000):
                    env.set_transport_exclusions(arm.held_object_id for arm in arms)
                    env.step()
                    step = env.simulation_step
                    if step >= 120 and step % 60 == 0:
                        frame = env.camera.capture_frame()
                        detections = vision.detect(frame.rgb, frame.depth, step * env.time_step)
                        candidates = []
                        for item in detections:
                            body = simulation_attachment_id(frame, item["pixel_center"], fixtures)
                            if body is None:
                                continue
                            bodies[item["track_id"]] = body
                            candidates.append(dict(item, arm_options={
                                "object_id": body, "grasp_offset_z": -.015,
                                "object_orientation": (0, 0, 0, 1),
                                "object_point_local": (0, 0, .05),
                                "object_long_axis_local": (1, 0, 0),
                            }))
                        scheduler.update(candidates, step)
                    update_arms(arms, step, env.time_step, previous)
                    monitor.sample(step)
                    scheduler.tick(step)
                    finished.extend(scheduler.drain_finished())
                    if len(finished) == 2 and all(arm.is_idle() for arm in arms):
                        break
                self.assertEqual(len(finished), 2, output.getvalue())
                self.assertEqual({task.arm_name for task in finished}, {"arm1", "arm2"})
                self.assertTrue(all(task.status == TaskStatus.COMPLETED for task in finished),
                                output.getvalue())
                self.assertTrue(all(task.handoff_count == 0 for task in finished), output.getvalue())
                self.assertGreaterEqual(monitor.summary()["longest_parallel_seconds"], .25,
                                        "busy states alone are insufficient: " + output.getvalue())
                for task in finished:
                    zone = env.get_arm_drop_zone(task.arm_name, task.label)
                    self.assertTrue(p.getContactPoints(bodies[task.track_id], zone.body_id),
                                    "payload did not reach its own dedicated station")
                self.assertTrue(all(arm.last_collision is None and arm.last_error is None for arm in arms),
                                output.getvalue())
                self.assertEqual(p.getNumConstraints(), len(env.gripper_attachments))
            finally:
                env.close()
            self.assertIsNone(workspace_gate._client, "shadow collision client leaked after env.close()")


if __name__ == "__main__":
    unittest.main(verbosity=2)
