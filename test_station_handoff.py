"""Physical RGB-D regressions for the two-station, one-handoff lifecycle.

Run: python -m unittest -v test_station_handoff
Only fixture creation uses known coordinates. All scheduler observations and
attachment handles come from rendered RGB/depth/segmentation frames.
"""

import math
import unittest

import pybullet as p

from env_setup import CameraConfig, SimulationEnvironment
from fault_injection import FaultInjectionPanel
from main import camera_delivery_error, create_controllers, simulation_attachment_id
from robot_arm import ArmState
from scheduler import TaskStatus
from vision import YoloVision


class StationHandoffIntegrationTests(unittest.TestCase):
    def _exercise(self, failure, *, fail_recipient=False, label="Box_B"):
        env = SimulationEnvironment(gui=False, camera_config=CameraConfig(width=480, height=360))
        panel = None
        try:
            scene_bodies, scene_constraints = p.getNumBodies(), p.getNumConstraints()
            color = (.95, .08, .03, 1) if label == "Box_A" else (.03, .18, .95, 1)
            # Keep the source arm's cost clearly lower than the recipient's,
            # while using a visible upstream point at the edge of the shared
            # cell so its RGB-D track remains stable during source recovery.
            body = env.create_test_object((-.32, -.06, env.conveyor_top_z + .05),
                                          shape="box", color=color)
            removed = []

            def remove_payload(handle):
                removed.append((handle, env.simulation_step))
                p.removeBody(handle)

            panel = FaultInjectionPanel(env, remove_target=remove_payload, gui=False)
            arms, scheduler = create_controllers(env, fault_panel=panel)
            vision = YoloVision(env, model=object())
            gate = arms[0].workspace_gate
            if failure == "grasp":
                panel.request("grasp")

            assignments, terminals, aborts = [], [], {}
            observed_ids, pending_steps = set(), []
            first_arm = None
            timeout_budget = None
            recipient_fault_queued = False
            terminal_idle_since = None
            visible_after_final_failure = 0
            handoff_gate_checks = 0
            recipient = None

            # 75 simulated seconds bounds all attempts, recovery and observation.
            for _ in range(18000):
                env.step()
                step = env.simulation_step
                panel.tick(step)
                if step % 60 == 0:
                    frame = env.camera.capture_frame()
                    observations = vision.detect(frame.rgb, frame.depth, step * env.time_step)
                    candidates = []
                    for item in observations:
                        handle = simulation_attachment_id(frame, item["pixel_center"], {body})
                        if handle is None:
                            continue
                        self.assertEqual(handle, body)
                        observed_ids.add(item["track_id"])
                        candidates.append(dict(item, arm_options={
                            "object_id": handle, "grasp_offset_z": -.015,
                            "object_orientation": (0, 0, 0, 1),
                            "object_point_local": (0, 0, .05),
                            "object_long_axis_local": (1, 0, 0),
                        }))

                    if step >= 120:
                        started = scheduler.update(candidates, step)
                        for task in started:
                            adapter = scheduler.arms[task.arm_name].arm
                            self.assertEqual(adapter.object_id, body)
                            assignments.append({"track_id": task.track_id, "arm": task.arm_name,
                                                "step": step, "body": adapter.object_id,
                                                "handoff_count": task.handoff_count})
                            if len(assignments) == 1:
                                first_arm = adapter
                                self.assertEqual(task.arm_name, "arm1")
                                if failure == "timeout":
                                    timeout_budget = adapter.arm.state_timeout_steps
                                    adapter.arm.state_timeout_steps = 60
                            else:
                                self.assertEqual(len(assignments), 2, "unexpected third assignment")
                                self.assertEqual(task.track_id, assignments[0]["track_id"])
                                self.assertNotEqual(task.arm_name, assignments[0]["arm"])
                                recipient = adapter
                                if fail_recipient:
                                    # The production fault panel intentionally
                                    # serializes one GUI fault. Early handoff
                                    # can be assigned before that first fault
                                    # has finished recovering, so inject this
                                    # second real empty grasp at its assignment
                                    # boundary rather than queuing it too late.
                                    adapter.height_bias = .15
                                    adapter.fault = "grasp"
                                    adapter._saved_approach_height = adapter.arm.approach_height
                                    adapter.arm.approach_height = .02
                                    target = adapter.arm._target_pos
                                    adapter.arm._target_pos = (*target[:2], target[2] + .15)
                                    recipient_fault_queued = True
                                self.assertIsNone(first_arm.held_object_id)
                                self.assertFalse(removed, "payload was removed before handoff")

                        if (terminals and terminals[-1].status == TaskStatus.FAILED
                                and any(item["track_id"] == assignments[0]["track_id"]
                                        for item in candidates)):
                            visible_after_final_failure += 1

                for adapter in arms:
                    previous = adapter.state
                    state = adapter.update()
                    if state == ArmState.ABORT and previous != ArmState.ABORT:
                        aborts.setdefault(adapter.display_name, []).append({
                            "step": step, "phase": adapter.failure_phase,
                            "reason": adapter.last_error,
                        })
                        if failure == "timeout" and adapter is first_arm:
                            self.assertEqual(adapter.failure_phase, "MOVE_TO_TARGET_ABOVE")
                            self.assertIn("Cartesian approach", adapter.last_error)
                            self.assertGreater(math.dist(adapter.get_tcp_position(),
                                                         adapter._initial_tcp_position), .01)
                            # Only the injected approach deadline is short. Recovery
                            # must execute the unchanged controller's normal return.
                            adapter.arm.state_timeout_steps = timeout_budget

                # Dispatch may happen while the source is still recovering.
                # The recipient's local grasp reservation must prevent any
                # physical departure into that same payload's occupied lane.
                if (recipient is not None and handoff_gate_checks == 0
                        and math.dist(recipient.get_tcp_position(), recipient._initial_tcp_position) > .01):
                    self.assertFalse(gate.has_reservation(first_arm.robot_id),
                                     'recipient entered the source local pickup reservation')
                    self.assertTrue(gate.has_reservation(recipient.robot_id))
                    self.assertIsNone(first_arm.last_collision)
                    self.assertIsNone(recipient.last_collision)
                    self.assertGreater(gate.minimum_observed_clearance, .025)
                    handoff_gate_checks += 1

                scheduler.tick(step)
                for task in scheduler.assigned_tasks.values():
                    if task.status == TaskStatus.PENDING_HANDOFF:
                        pending_steps.append(step)
                        self.assertFalse(removed)
                        self.assertEqual(p.getNumBodies(), scene_bodies + 1)
                        self.assertEqual(len(assignments), 1)
                        self.assertFalse(scheduler.failed_track_ids)
                        self.assertTrue(first_arm.handoff_eligible)
                        if not first_arm.is_idle():
                            self.assertTrue(gate.has_reservation(first_arm.robot_id),
                                            "source released its local recovery reservation")

                terminals.extend(scheduler.drain_finished())
                self.assertLessEqual(len(assignments), 2)
                self.assertFalse(removed, "nonterminal failure removed the physical payload")
                if terminals and all(arm.is_idle() for arm in arms):
                    if terminal_idle_since is None:
                        terminal_idle_since = step
                    # In the double-failure case, keep the real box visible for
                    # another second to prove the terminal ID cannot retry again.
                    if step - terminal_idle_since >= 240:
                        break
            else:
                self.fail(f"bounded integration run did not finish: {assignments}, {aborts}")

            self.assertEqual(len(assignments), 2)
            self.assertEqual(len(scheduler.assignment_log), 2)
            self.assertEqual([entry["attempt"] for entry in scheduler.assignment_log], [1, 2])
            self.assertEqual(observed_ids, {assignments[0]["track_id"]}, "tracker replaced the object ID")
            self.assertTrue(pending_steps)
            self.assertEqual(handoff_gate_checks, 1)
            self.assertEqual(len(terminals), 1, "first failure was exposed as a terminal event")
            task = terminals[0]
            self.assertEqual(task.handoff_count, 1)
            self.assertEqual(task.failed_arm_name, "arm1")
            self.assertEqual(task.arm_name, "arm2")
            self.assertEqual([event["event"] for event in scheduler.handoff_events],
                             ["pending", "started"])
            self.assertTrue(aborts.get("Arm 1"))
            self.assertIsNone(gate.owner)
            self.assertTrue(all(arm._at_initial_joint_pose() for arm in arms))
            self.assertTrue(all(arm.collision_count == 0 for arm in arms))
            self.assertEqual(p.getNumConstraints(), scene_constraints)
            for arm in arms:
                self.assertIsNone(arm.held_object_id)
                self.assertEqual(arm._virtual_grasp_constraint_id, -1)
                self.assertFalse(arm._virtual_collision_pairs)

            if fail_recipient:
                self.assertTrue(recipient_fault_queued)
                self.assertTrue(aborts.get("Arm 2"))
                self.assertEqual(task.status, TaskStatus.FAILED)
                self.assertEqual(scheduler.failed_track_ids, {task.track_id})
                self.assertEqual(scheduler.completed_count, 0)
                self.assertGreaterEqual(visible_after_final_failure, 4)
                self.assertTrue(p.getContactPoints(body, env.conveyor_id))
            else:
                self.assertEqual(task.status, TaskStatus.COMPLETED)
                self.assertEqual(scheduler.completed_count, 1)
                self.assertFalse(scheduler.failed_track_ids)
                zone = env.get_arm_drop_zone(task.arm_name, task.label)
                self.assertTrue(p.getContactPoints(body, zone.body_id))
                frame = env.camera.capture_frame()
                error = camera_delivery_error(frame, body, task.drop_pos, env.camera.config)
                self.assertLess(error, .035)
                for other in env.drop_zones:
                    if other is not zone:
                        self.assertFalse(p.getContactPoints(body, other.body_id))

            remove_payload(body)
            vision.forget_track(task.track_id)
            scheduler.release_destination(task.track_id)
            self.assertEqual(len(removed), 1)
            self.assertGreaterEqual(removed[0][1], task.finished_step)
            self.assertEqual(p.getNumBodies(), scene_bodies)
            self.assertEqual(p.getNumConstraints(), scene_constraints)
            self.assertFalse(scheduler._slots)
            self.assertFalse(scheduler.active_tasks)
            self.assertFalse(scheduler.pending_tasks)
        finally:
            if panel is not None:
                panel.close()
            env.close()

    def test_failed_grasp_hands_same_visual_object_to_other_station(self):
        self._exercise("grasp")

    def test_real_approach_timeout_recovers_before_handoff(self):
        self._exercise("timeout", label="Box_A")

    def test_second_failed_grasp_is_terminal_without_third_assignment(self):
        self._exercise("grasp", fail_recipient=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
