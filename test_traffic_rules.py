"""Measured RGB-D/physics traffic regressions for the shared pickup cell.

The first fixture reproduces the screenshot's order: one real empty grasp,
same-ID handoff to Arm 2, then a fresh task for the recovered Arm 1.  Tests
inspect actual TCPs and complete robot/tool/payload collision shapes; assigned
FSM states alone do not establish either safety or forward progress.
"""

import contextlib
import io
import json
import math
from pathlib import Path
import unittest

import cv2
import pybullet as p

from env_setup import CameraConfig, SimulationEnvironment
from fault_injection import FaultInjectionPanel
from main import create_controllers, simulation_attachment_id, update_arms
from robot_arm import ArmState
from scheduler import TaskStatus
from vision import YoloVision


def _candidates(env, vision, fixtures):
    frame = env.camera.capture_frame()
    result = []
    for item in vision.detect(frame.rgb, frame.depth, env.simulation_step*env.time_step):
        body = simulation_attachment_id(frame, item["pixel_center"], fixtures)
        if body is not None:
            result.append((body, dict(item, arm_options={
                "object_id": body, "grasp_offset_z": -.015,
                "object_orientation": (0, 0, 0, 1),
                "object_point_local": (0, 0, .05),
                "object_long_axis_local": (1, 0, 0),
            })))
    return result


class TrafficRulesIntegrationTests(unittest.TestCase):
    def assert_physical_clearance(self, gate, arms, log, *, check_zone=True):
        shapes = [gate._bodies(arm) for arm in arms]
        distance = gate._distance(shapes[0], shapes[1], 1)
        self.assertGreater(distance, .020, "complete arm/tool/payload clearance: " + log.getvalue())
        self.assertTrue(all(arm.collision_count == 0 for arm in arms), log.getvalue())
        low = [arm.robot_id for arm in arms
               if gate.in_spatial_zone(arm.get_measured_tcp_position())]
        if check_zone:
            self.assertLessEqual(len(low), 1,
                                 f"both measured TCPs entered the central traffic volume: {low}")

    def assert_token_ledger(self, events):
        owner = None
        entered = set()
        for event in events:
            kind, arm_id = event["event"], event.get("arm_id")
            if kind == "TOKEN_ACQUIRED":
                self.assertIn(owner, (None, arm_id), "a second token was issued while owned")
                owner = arm_id
                entered.add(arm_id)
            elif kind == "TOKEN_RELEASED":
                self.assertEqual(owner, arm_id, "a nonowner released the central token")
                owner = None
        self.assertEqual(len(entered), 2, "both arms must actually use the shared cell")
        self.assertIsNone(owner, "central token leaked after both tasks finished")

    def test_failed_pickup_handoff_and_source_new_task_finish_without_circular_wait(self):
        self._exercise_handoff()

    def test_legacy_pair_already_approaching_yields_lower_priority_arm_then_completes(self):
        self._exercise_handoff(legacy=True)

    def _exercise_handoff(self, *, legacy=False):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            env = SimulationEnvironment(gui=False,
                                        camera_config=CameraConfig(width=480, height=360))
            panel = None
            try:
                first = env.create_test_object((-.32, -.06, env.conveyor_top_z+.05),
                                              shape="box", color=(.03, .18, .95, 1))
                second = env.create_test_object((-.08, .04, env.conveyor_top_z+.05),
                                               shape="box", color=(.95, .08, .03, 1))
                self.assertGreater(math.dist(p.getBasePositionAndOrientation(first)[0],
                                             p.getBasePositionAndOrientation(second)[0]), .19)
                fixtures = {first, second}
                panel = FaultInjectionPanel(env, gui=False)
                panel.request("grasp")
                arms, scheduler = create_controllers(env, fault_panel=panel)
                gate = arms[0].workspace_gate
                acquire = gate.acquire
                vision = YoloVision(env, model=object())
                previous, assignments, finished = [None, None], [], []
                bodies, handoff_step = {}, None
                waiting_samples = 0
                handoff_priority_checks = payload_priority_checks = 0
                source_new_task = None
                legacy_admission = False
                legacy_recovered = False
                low_yield_start = None
                low_peak_z = -math.inf
                scene_saved = False
                samples = []
                yield_origins = {}
                maximum_yield_displacement = 0
                high_yield_beside_payload = False

                def legacy_acquire(arm_id):
                    # Recreate only the PREVIOUS VERSION'S admission policy,
                    # so the real collision arbiter encounters an already
                    # approaching pair. Complete geometry protection remains
                    # enabled throughout. No joint/TCP pose is teleported.
                    if arm_id not in gate.reservations:
                        gate._sequence += 1
                        gate.reservations[arm_id] = gate._sequence
                    return True

                for _ in range(round(55/env.time_step)):
                    env.step()
                    step = env.simulation_step
                    panel.tick(step)
                    if step >= 120 and step % 60 == 0:
                        candidates = []
                        for body, item in _candidates(env, vision, fixtures):
                            if body == second and handoff_step is None:
                                continue
                            bodies[item["track_id"]] = body
                            candidates.append(item)
                        for task in scheduler.update(candidates, step):
                            assignments.append((task.track_id, task.arm_name, task.handoff_count))
                            if task.handoff_count:
                                handoff_step = step
                            elif len(assignments) > 1:
                                source_new_task = task.track_id
                                if legacy:
                                    legacy_admission = True
                                    gate.acquire = legacy_acquire
                    update_arms(arms, step, env.time_step, previous)
                    scheduler.tick(step)
                    finished.extend(scheduler.drain_finished())
                    for index, yielding_arm in enumerate(arms):
                        if yielding_arm.traffic_yielding:
                            actual = yielding_arm.get_measured_tcp_position()
                            origin = yield_origins.setdefault(yielding_arm.robot_id, actual)
                            maximum_yield_displacement = max(maximum_yield_displacement,
                                                              math.dist(origin, actual))
                            if (arms[1-index].held_object_id is not None
                                    and actual[2] >= yielding_arm._initial_tcp_position[2]-.03):
                                high_yield_beside_payload = True
                    if legacy and arms[0].traffic_yielding:
                        if low_yield_start is None:
                            low_yield_start = arms[0].get_measured_tcp_position()
                            gate.acquire = acquire
                        low_peak_z = max(low_peak_z, arms[0].get_measured_tcp_position()[2])
                        if not gate.in_spatial_zone(arms[0].get_measured_tcp_position(), gate.tool_margin):
                            legacy_recovered = True
                        if (not scene_saved and low_peak_z >= arms[0]._initial_tcp_position[2]-.03
                                and arms[1].state in (ArmState.DESCEND, ArmState.GRASP, ArmState.LIFT)):
                            image = env.camera.capture_frame().rgb
                            cv2.imwrite(str(Path(__file__).parent / "outputs" / "traffic_yield_scene.png"),
                                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                            scene_saved = True
                    if step % 6 == 0:
                        self.assert_physical_clearance(gate, arms, log,
                                                       check_zone=not legacy_admission or legacy_recovered)
                        samples.append(dict(step=step,
                                            tcp=[arm.get_measured_tcp_position() for arm in arms],
                                            phase=[arm.state.name for arm in arms],
                                            wait=[arm._workspace_waiting or arm._proximity_waiting for arm in arms],
                                            yielding=[arm.traffic_yielding for arm in arms],
                                            token_owner=gate.owner))
                        if (source_new_task is not None and not arms[0].is_idle()
                                and not arms[1].is_idle() and arms[1].held_object_id is None
                                and arms[0].held_object_id is None
                                and arms[1].traffic_handoff_count == 1):
                            self.assertGreater(gate.priority(arms[1]), gate.priority(arms[0]),
                                               "a running handoff must outrank a fresh pickup")
                            handoff_priority_checks += 1
                        carrying = next((arm for arm in arms if arm.held_object_id is not None), None)
                        if carrying is not None:
                            peer = next(arm for arm in arms if arm is not carrying)
                            # Exercise the documented hierarchy with a REAL
                            # attached payload. Metadata is restored before
                            # either FSM or motor is allowed another update.
                            old_handoff = peer.traffic_handoff_count
                            peer.set_traffic_task(peer.traffic_track_id, 1)
                            self.assertGreater(gate.priority(carrying), gate.priority(peer),
                                               "an attached payload must outrank even a handoff")
                            peer.set_traffic_task(peer.traffic_track_id, old_handoff)
                            self.assertGreaterEqual(p.getNumConstraints(), len(env.gripper_attachments)+1)
                            payload_priority_checks += 1
                        if (source_new_task is not None
                                and arms[0]._workspace_waiting
                                and not arms[1].is_idle()):
                            waiting_samples += 1
                            self.assertGreaterEqual(arms[0].get_measured_tcp_position()[2], 1.10,
                                                    "waiting source must stay above the traffic volume")
                    if len(finished) == 2 and all(arm.is_idle() for arm in arms):
                        break
                self.assertEqual(len(finished), 2, log.getvalue())
                self.assertEqual(len(assignments), 3, log.getvalue())
                self.assertEqual(assignments[0][1:], ("arm1", 0))
                self.assertEqual(assignments[1], (assignments[0][0], "arm2", 1))
                self.assertEqual(assignments[2][1:], ("arm1", 0))
                if legacy:
                    self.assertIsNotNone(low_yield_start,
                                         "an already approaching lower-priority arm did not actively yield")
                    self.assertGreater(low_peak_z, low_yield_start[2]+.10,
                                       "YIELD_STARTED without an actual upward retreat")
                    self.assertTrue(legacy_recovered, "yield never cleared the shared volume")
                    self.assertGreaterEqual(arms[0].traffic_yield_count, 1)
                    self.assertTrue(any(event["event"] == "REPLAN" for event in gate.traffic_events))
                else:
                    self.assertGreater(waiting_samples, 0,
                                       "the new source task did not wait safely for the handoff token")
                    self.assertGreater(maximum_yield_displacement, .10,
                                       "priority yield log did not correspond to an actual retreat")
                    self.assertTrue(high_yield_beside_payload,
                                    "empty returning peer did not park high for the carrying arm")
                    self.assertTrue(any(event["event"] == "REPLAN" for event in gate.traffic_events))
                self.assertGreater(handoff_priority_checks, 0)
                self.assertGreater(payload_priority_checks, 0)
                self.assertTrue(all(task.status == TaskStatus.COMPLETED for task in finished),
                                log.getvalue())
                self.assertLess(env.simulation_step*env.time_step, 55)
                self.assertEqual(sum(task.handoff_count for task in finished), 1)
                for task in finished:
                    zone = env.get_arm_drop_zone(task.arm_name, task.label)
                    self.assertTrue(p.getContactPoints(bodies[task.track_id], zone.body_id),
                                    "completed state without physical delivery to the arm's own station")
                if not legacy:
                    self.assert_token_ledger(gate.traffic_events)
                self.assertGreater(gate.minimum_observed_clearance, .020)
                if legacy:
                    result = dict(elapsed=env.simulation_step*env.time_step, assignments=assignments,
                                  finished=[dict(track_id=task.track_id, arm=task.arm_name,
                                                 status=task.status.name, handoff_count=task.handoff_count)
                                            for task in finished],
                                  collisions=[arm.collision_count for arm in arms],
                                  yield_counts=[arm.traffic_yield_count for arm in arms],
                                  traffic_events=list(gate.traffic_events), samples=samples)
                    output = Path(__file__).parent / "outputs"
                    (output / "traffic_legacy_after_fix.json").write_text(
                        json.dumps(result, indent=2), encoding="utf-8")
                    (output / "traffic_legacy_after_fix.log").write_text(log.getvalue(), encoding="utf-8")
            finally:
                if panel is not None:
                    panel.close()
                env.close()

    def test_two_second_geometry_stall_retracts_real_tcp_and_resumes_original_task(self):
        self._exercise_stall()

    def test_external_target_loss_during_actual_yield_does_not_resume_cancelled_pickup(self):
        self._exercise_stall(cancel_during_yield=True)

    def _exercise_stall(self, *, cancel_during_yield=False):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            env = SimulationEnvironment(gui=False,
                                        camera_config=CameraConfig(width=480, height=360))
            try:
                body = env.create_test_object((-.30, -.04, env.conveyor_top_z+.05),
                                             shape="box", color=(.95, .08, .03, 1))
                arms, scheduler = create_controllers(env)
                arm, peer = arms
                gate = arm.workspace_gate
                real_motion_allowed = gate.motion_allowed
                blocked_step = None
                blocked_tcp = None
                maximum_retreat_height = -math.inf
                injection_finished = False
                saw_actual_yield_motion = False
                cancelled_step = None
                samples = []

                def temporary_block(controller):
                    nonlocal blocked_step, blocked_tcp, injection_finished
                    if (controller is arm and not injection_finished
                            and controller.state == ArmState.MOVE_TO_TARGET_ABOVE
                            and controller.get_measured_tcp_position()[2] < 1.10):
                        if blocked_step is None:
                            blocked_step = env.simulation_step
                            blocked_tcp = controller.get_measured_tcp_position()
                        if controller.traffic_yielding:
                            injection_finished = True
                        else:
                            # Deterministic sensor/arbiter deferral. Physics,
                            # motors, shadow retreat validation, task tracking
                            # and the other arm's updates all remain real.
                            return False, (peer.robot_id, .06, .01)
                    return real_motion_allowed(controller)

                gate.motion_allowed = temporary_block
                vision = YoloVision(env, model=object())
                previous, assignments, finished = [None, None], [], []
                for _ in range(round(35/env.time_step)):
                    env.step()
                    step = env.simulation_step
                    if step >= 120 and step % 60 == 0:
                        candidates = [item for _, item in _candidates(env, vision, {body})]
                        assignments.extend(scheduler.update(candidates, step))
                    update_arms(arms, step, env.time_step, previous)
                    scheduler.tick(step)
                    finished.extend(scheduler.drain_finished())
                    if arm.traffic_yielding:
                        injection_finished = True
                    if blocked_tcp is not None and arm.traffic_yielding:
                        actual = arm.get_measured_tcp_position()
                        maximum_retreat_height = max(maximum_retreat_height, actual[2])
                        if actual[2] >= blocked_tcp[2]+.10:
                            saw_actual_yield_motion = True
                        if (cancel_during_yield and cancelled_step is None
                                and actual[2] >= blocked_tcp[2]+.06):
                            cancelled_step = step
                            arm.cancel_task("target lost during traffic yield test")
                    if step % 6 == 0:
                        self.assert_physical_clearance(gate, arms, log)
                        samples.append(dict(step=step, tcp=arm.get_measured_tcp_position(),
                                            phase=arm.state.name, yielding=arm.traffic_yielding))
                    if len(finished) == 1 and all(controller.is_idle() for controller in arms):
                        break
                gate.motion_allowed = real_motion_allowed
                self.assertIsNotNone(blocked_step, "fixture did not reach the pregrasp motion")
                timeouts = [event for event in gate.traffic_events if event["event"] == "WAIT_TIMEOUT"]
                self.assertTrue(timeouts, "continuous 2-second stop did not trigger recovery: " + log.getvalue())
                delay = timeouts[0]["time"] - blocked_step*env.time_step
                self.assertGreaterEqual(delay, 1.95)
                self.assertLessEqual(delay, 2.15)
                self.assertTrue(injection_finished)
                self.assertEqual(arm.traffic_yield_count, 1, log.getvalue())
                self.assertEqual(len(assignments), 1, "retreat must resume the original task, not redispatch")
                self.assertEqual(len(finished), 1, log.getvalue())
                self.assertEqual(finished[0].track_id, assignments[0].track_id)
                if cancel_during_yield:
                    self.assertIsNotNone(cancelled_step, "fixture never cancelled an actual moving retreat")
                    self.assertGreater(maximum_retreat_height, blocked_tcp[2]+.05)
                    self.assertEqual(finished[0].status, TaskStatus.FAILED)
                    self.assertEqual(finished[0].handoff_count, 0)
                    self.assertFalse(arm.traffic_yielding)
                    self.assertIsNone(arm._traffic_resume_state)
                    self.assertIsNone(arm._traffic_yield_requested)
                    self.assertTrue(arm._at_initial_joint_pose())
                    self.assertTrue(p.getContactPoints(body, env.conveyor_id))
                    self.assertTrue(any(event["event"] == "YIELD_CANCELLED" for event in gate.traffic_events))
                    self.assertFalse(any(event["event"] == "REPLAN"
                                         and event["time"] >= cancelled_step*env.time_step
                                         for event in gate.traffic_events),
                                     "a cancelled pickup resumed its saved traffic state")
                else:
                    self.assertTrue(saw_actual_yield_motion,
                                    f"retreat event without real upward TCP motion; peak={maximum_retreat_height}: "
                                    + log.getvalue())
                    self.assertEqual(finished[0].status, TaskStatus.COMPLETED, log.getvalue())
                    zone = env.get_arm_drop_zone(finished[0].arm_name, finished[0].label)
                    self.assertTrue(p.getContactPoints(body, zone.body_id))
                    self.assertTrue(any(event["event"] == "REPLAN" for event in gate.traffic_events))
                output = Path(__file__).parent / "outputs"
                name = "traffic_yield_cancel" if cancel_during_yield else "traffic_watchdog"
                result = dict(elapsed=env.simulation_step*env.time_step,
                              stop_to_timeout_seconds=delay, blocked_tcp=blocked_tcp,
                              maximum_retreat_height=maximum_retreat_height,
                              actual_rise=maximum_retreat_height-blocked_tcp[2],
                              cancelled_step=cancelled_step, assignment_count=len(assignments),
                              finished_status=finished[0].status.name,
                              collisions=[controller.collision_count for controller in arms],
                              yield_count=arm.traffic_yield_count,
                              traffic_events=list(gate.traffic_events), samples=samples)
                (output / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                (output / f"{name}.log").write_text(log.getvalue(), encoding="utf-8")
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
