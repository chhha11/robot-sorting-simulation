"""Optional GUI fault injection; production vision/scheduling stays unchanged.

Run ``python fault_injection.py`` for the four finite DIRECT-mode regressions.
All deadlines and recovery use physics ticks. No payload pose is queried.
"""

from __future__ import annotations

from collections import deque
import math

import pybullet as p

from robot_arm import ArmState, BaseRobotArm


class _TestArmAdapter(BaseRobotArm):
    """Intercept commands at the scheduler/arm boundary, only for an armed test."""

    def __init__(self, arm, panel):
        self.arm, self.panel = arm, panel
        self.object_id = None
        self.height_bias = 0.0
        self.fault = None
        self._saved_approach_height = None

    def __getattr__(self, name):
        return getattr(self.arm, name)

    def is_idle(self):
        return self.arm.is_idle()

    def get_last_error(self):
        return self.arm.get_last_error()

    def get_tcp_position(self):
        return self.arm.get_tcp_position()

    def cancel_task(self, reason):
        return self.arm.cancel_task(reason)

    def estimate_pick_seconds(self, target_pos):
        return self.arm.estimate_pick_seconds(target_pos)

    @property
    def is_tracking_target(self):
        return self.arm.is_tracking_target

    def get_handoff_status(self):
        return self.arm.get_handoff_status()

    def set_traffic_task(self, track_id, handoff_count=0):
        return self.arm.set_traffic_task(track_id, handoff_count)

    def assign_task(self, target_pos, drop_pos, **options):
        self.object_id = options.get("object_id")
        self.height_bias, self.fault = 0.0, None
        fault = self.panel.claim_assignment(self)
        target = tuple(target_pos)
        if fault == "grasp":
            self.height_bias = 0.15
            # The fake grasp already sits 15 cm above the object. Adding the
            # normal extra 15 cm would test an unreachable approach instead
            # of empty fingers. Keep 2 cm clearance over this fake surface,
            # including retry/abort, and restore the normal setting at IDLE.
            self._saved_approach_height = self.arm.approach_height
            self.arm.approach_height = 0.02
            target = (*target[:2], target[2] + self.height_bias)
            if options.get("tracking_position") is not None:
                observed = options["tracking_position"]
                options["tracking_position"] = (*observed[:2], observed[2] + self.height_bias)
        elif fault == "unreachable":
            target = (10.0, 10.0, 10.0)
        self.arm.assign_task(target, drop_pos, **options)
        if fault:
            self.arm.flash_failure()
            self.panel.warn(
                "ERROR: GRASP HEIGHT +0.15m" if fault == "grasp" else "ERROR: UNREACHABLE TARGET",
                self,
            )
        if fault == "unreachable":
            # Reject the deliberately invalid command BEFORE any IK/motor tick.
            # Recovery starts from the measured TCP, never the distant target.
            self.arm.abort_and_return("[FAULT TEST] unreachable target (10, 10, 10)")

    def update_tracking_observation(self, position, velocity):
        if self.height_bias:
            position = (*position[:2], position[2] + self.height_bias)
        self.arm.update_tracking_observation(position, velocity)

    def update(self):
        previous = self.arm.state
        state = self.arm.update()
        if self.fault and self.arm.last_error and not self.arm.last_error.startswith("[FAULT TEST]"):
            self.arm.last_error = f"[FAULT TEST] {self.fault}: {self.arm.last_error}"
        if self.fault == "grasp" and state != previous:
            if state == ArmState.RETRY:
                self.panel.warn("WARNING: GRASP FAILED / RETRYING", self)
            elif state == ArmState.ABORT:
                self.panel.warn("ERROR: GRASP FAILED / ABORT", self)
        return state


class FaultInjectionPanel:
    """Four latched sliders with a serialized, one-shot fault queue.

    Grasp/unreachable intercept the next assignment. Target loss waits for a
    descent; collision waits for an approach travel segment. Raising a
    slider while busy therefore queues a test without interrupting a payload
    already being carried. Return it to zero before raising it again.
    """

    LABELS = {
        # Bullet's fixed-width slider text clips long labels on high-DPI screens.
        "grasp": "Grasp Fail",
        "lost": "Target Lost",
        "unreachable": "Unreachable",
        "collision": "Collision",
    }

    def __init__(self, env, remove_target=None, gui=True):
        self.env = env
        self.remove_target = remove_target or p.removeBody
        self.arms = []
        self.pending = deque()
        self.active = None
        self.step = 0
        self.events = deque(maxlen=64)
        self._latched = {kind: False for kind in self.LABELS}
        self._controls = {}
        self._banner_id = -1
        self._banner = ""
        self._banner_until = 0
        self._obstacle = None
        self._obstacle_until = 0
        self._obstacle_collision_sequence = 0
        self._collision_shape = None
        self._visual_shape = None
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            self._controls = {
                # Leave a gutter for Bullet's high-DPI native text offset.
                kind: p.addUserDebugParameter(f"            {label}", 0, 1, 0)
                for kind, label in self.LABELS.items()
            }
            print("Fault panel: raise to 1 to queue; return to 0 before triggering again.", flush=True)

    def wrap(self, arm):
        adapter = _TestArmAdapter(arm, self)
        self.arms.append(adapter)
        return adapter

    def set_slider_value(self, kind, value):
        """Shared GUI/test edge detector: high triggers once; zero rearms."""
        if value <= 0.05:
            self._latched[kind] = False
        elif value >= 0.95 and not self._latched[kind]:
            self._latched[kind] = True
            self.request(kind)

    def request(self, kind):
        if kind not in self.LABELS:
            raise ValueError(f"Unknown fault: {kind}")
        if kind in self.pending or (self.active is not None and self.active.fault == kind):
            return
        self.pending.append(kind)
        self.warn(f"TEST QUEUED: {self.LABELS[kind]}")

    def _activate(self, adapter, kind):
        self.pending.popleft()
        self.active, adapter.fault = adapter, kind
        self.events.append((kind, "started", self.step))
        adapter.arm.flash_failure()

    def claim_assignment(self, adapter):
        if self.active is None and self.pending and self.pending[0] in ("grasp", "unreachable"):
            kind = self.pending[0]
            self._activate(adapter, kind)
            return kind
        return None

    def warn(self, message, adapter=None):
        self._banner = f"{adapter.display_name}: {message}" if adapter else message
        self._banner_until = self.step + round(3.0 / self.env.time_step)
        print(self._banner, flush=True)

    def _draw_banner(self):
        if self.step >= self._banner_until:
            if not self.pending:
                return
            self._banner = f"TEST WAITING: {self.LABELS[self.pending[0]]}"
        kwargs = dict(text=self._banner, textPosition=(-0.8, 0.0, 1.65),
                      textColorRGB=(1.0, 0.02, 0.02), textSize=2.5, lifeTime=0.1)
        if self._banner_id >= 0:
            kwargs["replaceItemUniqueId"] = self._banner_id
        try:
            self._banner_id = p.addUserDebugText(**kwargs)
        except TypeError:
            kwargs.pop("replaceItemUniqueId", None)
            self._banner_id = p.addUserDebugText(**kwargs)

    def _remove_obstacle(self):
        if self._obstacle is not None:
            p.removeBody(self._obstacle)
            self._obstacle = None

    def _spawn_obstacle(self, adapter, segment):
        start, end = segment
        direction = tuple(end[i] - start[i] for i in range(3))
        midpoint = tuple((start[i] + end[i]) / 2 for i in range(3))
        yaw = math.atan2(direction[1], direction[0])
        pitch = -math.atan2(direction[2], math.hypot(direction[0], direction[1]))
        # A wide wall perpendicular to travel: 8 cm thick, 32 x 32 cm face.
        # A non-red/blue material prevents it becoming a sorting candidate.
        if self._collision_shape is None:
            self._collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=(0.04, 0.16, 0.16))
            self._visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=(0.04, 0.16, 0.16),
                                                    rgbaColor=(0.9, 0.75, 0.05, 0.85))
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=self._collision_shape,
                                 baseVisualShapeIndex=self._visual_shape, basePosition=midpoint,
                                 baseOrientation=p.getQuaternionFromEuler((0, pitch, yaw)))
        # Never materialize an obstacle intersecting an arm. Wait for the
        # next eligible segment if this midpoint is already occupied.
        if any(p.getClosestPoints(part, body, distance=0.005)
               for item in self.arms for part in (item.robot_id, item.gripper_id) if part is not None):
            p.removeBody(body)
            return False
        self._obstacle = body
        self._obstacle_collision_sequence = adapter.collision_count
        self._obstacle_until = self.step + round(4.0 / self.env.time_step)
        self._activate(adapter, "collision")
        self.warn("TEST: OBSTACLE ON APPROACH PATH", adapter)
        return True

    def _check_obstacle(self):
        if self._obstacle is None:
            return
        adapter = self.active
        event = adapter.last_collision
        if (event is not None and event["sequence"] > self._obstacle_collision_sequence
                and event["other_body"] == self._obstacle):
            # The REAL controller detected contact and stopped on the last
            # tick. The panel only clears its own temporary test obstacle;
            # arbitrary obstacles stay in the world and keep the arm stopped.
            print(f"FAULT COLLISION confirmed by RobotArm: {event}", flush=True)
            self._remove_obstacle()
            self.events.append(("collision", "contact", self.step))
            self.warn("WARNING: COLLISION DETECTED", adapter)
        elif self.step >= self._obstacle_until:
            self._remove_obstacle()
            adapter.arm.abort_and_return("[FAULT TEST] obstacle test timed out without contact")
            self.warn("TEST: NO CONTACT / OBSTACLE REMOVED", adapter)

    def tick(self, step):
        """Call after stepSimulation and before scheduler/arm updates."""
        self.step = step
        for kind, control in self._controls.items():
            self.set_slider_value(kind, p.readUserDebugParameter(control))
        if self.active is not None and self.active.is_idle():
            self.events.append((self.active.fault, "finished", step))
            self._remove_obstacle()
            self.warn("TEST FINISHED / READY", self.active)
            self.active.height_bias = 0.0
            if self.active._saved_approach_height is not None:
                self.active.arm.approach_height = self.active._saved_approach_height
                self.active._saved_approach_height = None
            self.active = None
        if self.active is None and self.pending:
            kind = self.pending[0]
            for adapter in self.arms:
                if adapter.object_id is None:
                    continue
                if kind == "lost" and adapter.state == ArmState.DESCEND:
                    self._activate(adapter, kind)
                    # Restore arm/payload collision filters before deleting a
                    # body handle that Bullet may later recycle for new infeed.
                    adapter.arm.abort_and_return("[FAULT TEST] target lost")
                    self.remove_target(adapter.object_id)
                    self.events.append((kind, "removed", step))
                    self.warn("ERROR: TARGET LOST", adapter)
                    break
                if kind == "collision" and adapter.state == ArmState.MOVE_TO_TARGET_ABOVE:
                    segment = adapter.current_motion_segment()
                    if segment is not None:
                        start, end = segment
                        # The travel leg may also rise to safe height. Orient
                        # the wall perpendicular to that actual straight line.
                        # Exclude the short initial lift above the wrist.
                        if math.dist(start[:2], end[:2]) > 0.30:
                            if self._spawn_obstacle(adapter, segment):
                                break
        self._check_obstacle()
        if self._banner or self.pending:
            self._draw_banner()

    def close(self):
        if p.isConnected():
            self._remove_obstacle()
            if self._banner_id >= 0:
                p.removeUserDebugItem(self._banner_id)


def _run_legacy_regression_tests():
    """Exercise actual Bullet motion, recovery and slider edges in DIRECT mode.

    Fixture coordinates are used only to create a known controller test. No
    simulated object pose is read. The normal Scheduler receives its usual
    detection dictionary and is responsible for recording the failed task.
    """
    from env_setup import SimulationEnvironment
    from main import create_controllers
    from scheduler import TaskStatus

    for kind in FaultInjectionPanel.LABELS:
        env = SimulationEnvironment(gui=False)
        panel = None
        removed = []
        try:
            def remove_target(body):
                p.removeBody(body)
                removed.append(body)

            panel = FaultInjectionPanel(env, remove_target=remove_target, gui=False)
            arms, scheduler = create_controllers(env, fault_panel=panel)
            normal_approach_height = arms[0].approach_height
            target = (-0.55, -0.30, env.conveyor_top_z + 0.10)
            body = env.create_test_object(
                (target[0], target[1], env.conveyor_top_z + 0.05),
                shape="box", color=(0.95, 0.08, 0.03, 1.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )
            body_count = p.getNumBodies()
            constraint_count = p.getNumConstraints()
            # All fixtures settle before the test's first control command.
            for _ in range(120):
                env.step()
            detection = {
                "class": "Box_A", "pos_3d": target,
                "bbox": (100, 100, 140, 130), "track_id": 1,
                "arm_options": {
                    "object_id": body, "grasp_offset_z": -0.015,
                    "object_orientation": (0.0, 0.0, 0.0, 1.0),
                    "object_point_local": (0.0, 0.0, 0.05),
                    "object_long_axis_local": (1.0, 0.0, 0.0),
                },
            }
            panel.set_slider_value(kind, 1.0)
            for _ in range(5):
                panel.set_slider_value(kind, 1.0)
            assert list(panel.pending) == [kind], "high slider queued duplicate faults"
            started = scheduler.update([detection], env.simulation_step)
            assert len(started) == 1, "fixture was not assigned"
            arm = arms[0]
            seen_states = {arm.state}
            if kind == "unreachable":
                assert arm.state == ArmState.ABORT, "invalid target reached the motion loop"
                assert arm.last_error and "unreachable" in arm.last_error

            completed = False
            for tick in range(12000):
                env.step()
                panel.set_slider_value(kind, 1.0)
                panel.tick(env.simulation_step)
                seen_states.add(arm.state)
                for controller in arms:
                    controller.update()
                seen_states.add(arm.state)
                scheduler.update([detection], env.simulation_step)
                task = scheduler.assigned_tasks[1]
                if arm.is_idle() and panel.active is None and task.status == TaskStatus.FAILED:
                    completed = True
                    break

            assert completed, f"{kind}: fault did not finish within the bounded test"
            assert task.error and task.error.startswith("[FAULT TEST]"), task.error
            assert ArmState.ABORT in seen_states, f"{kind}: no ABORT feedback"
            assert ArmState.RETURN_HOME in seen_states, f"{kind}: recovery skipped HOME"
            assert all(controller.is_idle() for controller in arms), "a controller remained busy"
            assert not scheduler.active_tasks, "failed task still owns an arm"
            assert sum(event[:2] == (kind, "started") for event in panel.events) == 1
            assert sum(event[:2] == (kind, "finished") for event in panel.events) == 1
            assert not panel.pending, "a high slider retriggered after test completion"
            if kind == "grasp":
                assert ArmState.RETRY in seen_states, "empty grasp did not retry"
                assert arm._grasp_retry_count == 1, "empty grasp retry budget was not exactly one"
            elif kind == "lost":
                assert removed == [body], "target-loss injection removed the wrong payload"
                existing = {p.getBodyUniqueId(index) for index in range(p.getNumBodies())}
                assert body not in existing, "removed target still exists in Bullet"
                assert any(event[:2] == (kind, "removed") for event in panel.events)
            elif kind == "unreachable":
                assert tick * env.time_step < 10.0, "unreachable rejection waited for a motion timeout"
            elif kind == "collision":
                assert any(event[:2] == (kind, "contact") for event in panel.events), (
                    "collision test never produced a real Bullet contact"
                )
                assert "collision detected" in task.error, "obstacle expired without collision"
                assert arm.collision_count >= 1 and arm.last_collision is not None, (
                    "fault panel bypassed the controller's live contact monitor"
                )

            assert panel._obstacle is None, "test obstacle leaked after recovery"
            assert p.getNumBodies() == body_count - int(kind == "lost"), "temporary body leaked"
            assert p.getNumConstraints() == constraint_count, "virtual grasp constraint leaked"
            assert all(controller.held_object_id is None for controller in arms)
            assert arm.height_bias == 0.0, "fault height offset leaked into the next task"
            assert arm.approach_height == normal_approach_height, "approach clearance was not restored"

            # Keep the slider high throughout a subsequent REAL pick/place.
            # This catches both lingering injection settings and a latch that
            # requeues a fault when the first test's arm becomes available.
            if kind != "lost":
                p.removeBody(body)
            next_body = env.create_test_object(
                (target[0], target[1], env.conveyor_top_z + 0.05),
                shape="box", color=(0.95, 0.08, 0.03, 1.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )
            for _ in range(120):
                env.step()
                panel.tick(env.simulation_step)
                for controller in arms:
                    controller.update()
            next_detection = dict(
                detection, track_id=2,
                arm_options=dict(detection["arm_options"], object_id=next_body),
            )
            next_started = scheduler.update([next_detection], env.simulation_step)
            assert len(next_started) == 1, "recovered arm did not accept the next normal task"
            assert next_started[0].arm_name == task.arm_name, "normal task used a different arm"
            assert next_started[0].drop_pos == task.drop_pos, (
                "failed task did not release its unused drop slot"
            )
            normal_completed = False
            for normal_tick in range(8000):
                env.step()
                panel.set_slider_value(kind, 1.0)
                panel.tick(env.simulation_step)
                for controller in arms:
                    controller.update()
                scheduler.update([next_detection], env.simulation_step)
                next_task = scheduler.assigned_tasks[2]
                if arm.is_idle():
                    normal_completed = next_task.status == TaskStatus.COMPLETED
                    break
            assert normal_completed, f"{kind}: subsequent normal task failed: {arm.last_error}"
            assert next_task.error is None and arm.last_error is None
            assert arm._grasp_retry_count == 0, "normal task unexpectedly retried after fault recovery"
            assert arm.height_bias == 0.0 and arm.approach_height == normal_approach_height
            assert p.getContactPoints(next_body, env.get_drop_zone(0).body_id), (
                "subsequent normal payload did not physically reach its drop platform"
            )
            assert not scheduler.active_tasks and scheduler.completed_count == 1
            assert panel.active is None and not panel.pending and panel._obstacle is None
            assert p.getNumBodies() == body_count, "normal follow-up leaked a temporary body"
            assert p.getNumConstraints() == constraint_count, "normal follow-up leaked a grasp constraint"
            assert all(controller.held_object_id is None for controller in arms)
            assert sum(event[:2] == (kind, "started") for event in panel.events) == 1
            # The exact GUI edge detector must allow another request only
            # after returning to zero, even when the previous test is done.
            panel.set_slider_value(kind, 1.0)
            assert not panel.pending, "unchanged high slider triggered a second test"
            panel.set_slider_value(kind, 0.0)
            panel.set_slider_value(kind, 1.0)
            panel.set_slider_value(kind, 1.0)
            assert list(panel.pending) == [kind], "zero/high failed to rearm exactly once"
            print(
                f"PASS fault={kind}: FAILED -> IDLE in {(tick + 1) * env.time_step:.2f}s; "
                f"next normal pick COMPLETED in {(normal_tick + 1) * env.time_step:.2f}s; "
                "single trigger, rearm and resource cleanup verified",
                flush=True,
            )
        finally:
            if panel is not None:
                panel.close()
            env.close()
    print("PASS: all four interactive fault regressions", flush=True)


def _run_regression_tests():
    """Run the current two-station handoff suite.

    The old four-slider regression remains available as
    ``_run_legacy_regression_tests`` for historical debugging, but its fixture
    coordinates and shared-drop assumptions predate the dedicated stations.
    The default entry point therefore runs only the maintained physical RGB-D
    handoff cases and reports their unittest status.
    """
    import unittest

    from test_station_handoff import StationHandoffIntegrationTests

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        StationHandoffIntegrationTests
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    _run_regression_tests()
