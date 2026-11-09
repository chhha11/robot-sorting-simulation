"""Traffic policy boundaries independent of geometric retreat integration."""

import contextlib
import io
import unittest

from robot_arm import ArmState, SharedWorkspaceGate


class PolicyArm:
    """A parked assembly exposing the real controller's traffic state."""

    def __init__(self, robot_id, state, *, handoff_count=0, payload=None):
        self.robot_id = robot_id
        self.state = state
        self.held_object_id = payload
        self.traffic_handoff_count = handoff_count
        self.traffic_track_id = robot_id
        self.traffic_yielding = False
        self._traffic_yield_requested = None
        self._collision_recovery_blocked = False
        self._update_ticks = 0
        self._time_step = 1 / 240
        self.display_name = f"Arm {robot_id}"

    def is_idle(self):
        # A real IDLE controller becomes unavailable to the Scheduler while
        # yielding; this must not turn a completed task into a new handoff.
        return (self.state == ArmState.IDLE and not self.traffic_yielding
                and self._traffic_yield_requested is None)

    def get_tcp_position(self):
        return (-.4 if self.robot_id == 1 else .4, 0.0, 1.25)


class TrafficPriorityTests(unittest.TestCase):
    def setUp(self):
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        self.gate = SharedWorkspaceGate()
        self.addCleanup(self.gate.close)

    def register(self, *arms):
        # The policy only needs parked positions, so no physics/shadow client
        # or simulated grasp is required to exercise token arbitration.
        self.gate.arms = {arm.robot_id: arm for arm in arms}

    def test_idle_completed_handoff_stays_below_active_work_when_yield_requested(self):
        idle = PolicyArm(1, ArmState.IDLE, handoff_count=1)
        working = PolicyArm(2, ArmState.MOVE_TO_TARGET_ABOVE)
        self.register(idle, working)
        for yielding, requested in ((False, 2), (True, None)):
            with self.subTest(yielding=yielding, requested=requested):
                idle.traffic_yielding = yielding
                idle._traffic_yield_requested = requested
                self.assertFalse(idle.is_idle(), "the fixture must expose traffic availability")
                self.assertGreater(self.gate.priority(working), self.gate.priority(idle))

    def test_loaded_lift_requester_beats_an_ordinary_approach_contender(self):
        loaded = PolicyArm(1, ArmState.LIFT, payload=42)
        ordinary = PolicyArm(2, ArmState.MOVE_TO_TARGET_ABOVE)
        self.register(loaded, ordinary)
        self.assertGreater(self.gate.priority(loaded), self.gate.priority(ordinary))
        # A yielded payload resumes in LIFT, outside the ordinary pickup
        # contender states. It still must participate in its own token vote.
        self.assertTrue(self.gate.acquire(loaded.robot_id))
        self.assertEqual(self.gate.owner, loaded.robot_id)
        self.assertFalse(self.gate.acquire(ordinary.robot_id))

    def test_first_tick_handoff_wins_before_its_controller_has_requested_token(self):
        ordinary = PolicyArm(1, ArmState.MOVE_TO_TARGET_ABOVE)
        handoff = PolicyArm(2, ArmState.MOVE_TO_TARGET_ABOVE, handoff_count=1)
        self.register(ordinary, handoff)
        self.assertNotIn(handoff.robot_id, self.gate.requests)
        # Main updates Arm 1 first. Arm 2's freshly published assignment must
        # already participate, or the old source can steal the shared cell.
        self.assertFalse(self.gate.acquire(ordinary.robot_id))
        self.assertTrue(self.gate.acquire(handoff.robot_id))
        self.assertEqual(self.gate.owner, handoff.robot_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
