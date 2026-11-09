"""Physical regressions for the shared-volume interlock at nearby standby poses."""
import contextlib
import io
import math
import unittest

import pybullet as p

from env_setup import SimulationEnvironment
from main import create_controllers
from robot_arm import ArmState


class NearHomeInterlockTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.quiet = contextlib.redirect_stdout(self.output)
        self.quiet.__enter__()
        self.env = SimulationEnvironment(gui=False)
        self.arms, _ = create_controllers(self.env)
        self.gate = self.arms[0].workspace_gate
        for _ in range(120):
            self.tick()

    def tearDown(self):
        self.env.close()
        self.quiet.__exit__(None, None, None)

    def tick(self, arms=None):
        self.env.step()
        for arm in self.arms if arms is None else arms:
            arm.update()

    def assign(self, arm, moving=False, object_id=None):
        name = f'arm{self.env.arm_ids.index(arm.robot_id) + 1}'
        zone = self.env.get_arm_drop_zone(name, 'Box_B')
        arm.assign_task((-.11, 0, self.env.conveyor_top_z + .10),
                        (zone.surface_position[0], zone.surface_position[1],
                         zone.surface_position[2] + .10),
                        object_id=object_id,
                        grasp_offset_z=-.015, object_point_local=(0, 0, .05),
                        object_long_axis_local=(1, 0, 0),
                        target_velocity=(.05, 0, 0) if moving else (0, 0, 0))

    def waiting_arm(self):
        owner, waiter = self.arms
        self.gate.acquire(owner.robot_id)
        self.assign(waiter, moving=True)
        self.tick([waiter])
        self.assertEqual(self.gate.owner, owner.robot_id)
        return owner, waiter

    def test_wait_holds_home_and_preserves_motion_timeout_budget(self):
        owner, waiter = self.waiting_arm()
        waiter.state_timeout_steps = 24
        before = waiter.get_measured_tcp_position()
        for _ in range(120):  # five times the deliberately short motion budget
            owner._compensate_arm_gravity()
            self.tick([waiter])
            self.assertEqual(self.gate.owner, owner.robot_id)
            self.assertEqual(waiter.state, ArmState.MOVE_TO_TARGET_ABOVE)
            self.assertIsNone(waiter.current_motion_segment())
            self.assertLess(math.dist(before, waiter.get_measured_tcp_position()), .005)
        self.assertIsNone(waiter.last_error)
        self.assertFalse(waiter._timed_out())
        self.gate.release(owner.robot_id)
        self.tick([waiter])
        self.assertEqual(self.gate.owner, waiter.robot_id)
        self.assertIsNone(waiter.last_error)

    def assert_cancelled_waiter_stays_home(self, external_fault):
        owner, waiter = self.waiting_arm()
        before = waiter.get_measured_tcp_position()
        reason = 'test target expired while waiting'
        if external_fault:
            waiter.abort_and_return(reason)
        else:
            waiter.cancel_task(reason)
        for _ in range(120):
            owner._compensate_arm_gravity()
            self.tick([waiter])
            self.assertEqual(self.gate.owner, owner.robot_id)
            self.assertLess(math.dist(before, waiter.get_measured_tcp_position()), .005,
                            'a cancelled waiter entered the occupied shared volume')
        self.assertTrue(waiter.is_idle(), self.output.getvalue())
        self.assertTrue(waiter._at_initial_joint_pose())
        self.assertEqual(waiter.last_error, reason)
        self.assertEqual(p.getNumConstraints(), len(self.env.gripper_attachments))

    def test_cancel_waiting_track_does_not_start_an_unreserved_return(self):
        self.assert_cancelled_waiter_stays_home(external_fault=False)

    def test_external_fault_while_waiting_does_not_leave_home(self):
        self.assert_cancelled_waiter_stays_home(external_fault=True)

    def test_cancel_before_first_control_tick_does_not_enter_occupied_space(self):
        owner, waiter = self.arms
        self.gate.acquire(owner.robot_id)
        self.assign(waiter, moving=True)
        before = waiter.get_measured_tcp_position()
        waiter.cancel_task('cancelled before first control tick')
        for _ in range(120):
            owner._compensate_arm_gravity()
            self.tick([waiter])
            self.assertEqual(self.gate.owner, owner.robot_id)
            self.assertLess(math.dist(before, waiter.get_measured_tcp_position()), .005)
        self.assertTrue(waiter.is_idle(), self.output.getvalue())

    def test_collision_at_waiting_home_cannot_become_dispatchable(self):
        owner, waiter = self.waiting_arm()
        finger = p.getLinkState(waiter.gripper_id, waiter.gripper_joint_indices[0],
                                computeForwardKinematics=True)
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=(.006,) * 3)
        obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                     basePosition=finger[4], baseOrientation=finger[5])
        p.performCollisionDetection()
        self.assertTrue(p.getContactPoints(bodyA=waiter.gripper_id, bodyB=obstacle))
        waiter.update()
        self.assertFalse(waiter.is_idle(), 'a physical collision was treated as ordinary cancellation')
        for _ in range(120):
            owner._compensate_arm_gravity()
            self.tick([waiter])
            self.assertFalse(waiter.is_idle(), 'a colliding standby arm accepted new work')
            self.assertEqual(self.gate.owner, owner.robot_id)
            self.assertIsNone(waiter.current_motion_segment())
        self.assertIsNotNone(waiter.last_collision)

    def test_cancel_active_static_pick_keeps_gate_until_safe_home(self):
        owner, waiter = self.arms
        self.assign(owner)
        for _ in range(1200):
            self.tick()
            if math.dist(owner.get_tcp_position(), owner._initial_tcp_position) > .12:
                break
        else:
            self.fail('fixture did not leave standby: ' + self.output.getvalue())
        self.assertEqual(self.gate.owner, owner.robot_id)
        owner.cancel_task('test cancelled active static pick')
        for _ in range(4000):
            self.tick()
            if owner.is_idle():
                self.assertTrue(owner._at_initial_joint_pose(),
                                'shared volume was released while arm was away from HOME')
                self.assertLess(math.dist(owner.get_tcp_position(), owner._initial_tcp_position), .015)
                break
            self.assertEqual(self.gate.owner, owner.robot_id)
        else:
            self.fail('cancel did not recover to HOME: ' + self.output.getvalue())
        self.assertIsNone(self.gate.owner)

    def test_nearby_pick_waits_only_until_owner_exits_belt_lane(self):
        owner, waiter = self.arms
        body = self.env.create_test_object((-.11, 0, self.env.conveyor_top_z + .05),
                                           shape='box', color=(.03, .18, .95, 1))
        for _ in range(120):
            self.tick()
        self.assign(owner, object_id=body)
        self.tick([owner])
        self.assign(waiter)
        seen, waiting_pose = set(), waiter.get_measured_tcp_position()
        minimum_gap = math.inf
        for _ in range(8000):
            self.env.step()
            state = owner.update()
            seen.add(state)
            self.assertIsNone(owner.last_error, self.output.getvalue())
            waiter.update()
            minimum_gap=min(minimum_gap,self.gate.minimum_observed_clearance)
            self.assertIsNone(waiter.last_error)
            if math.dist(waiting_pose, waiter.get_measured_tcp_position()) > .015:
                self.assertFalse(owner.is_idle(), 'waiter still waited for a complete cycle')
                self.assertIn(state, (ArmState.MOVE_TO_DROP_ABOVE, ArmState.DESCEND_TO_DROP,
                                      ArmState.RELEASE, ArmState.LIFT_AFTER_RELEASE,
                                      ArmState.RETURN_HOME))
                self.assertTrue(self.gate.has_reservation(waiter.robot_id))
                self.assertFalse(self.gate.has_reservation(owner.robot_id))
                break
            self.assertTrue(self.gate.has_reservation(owner.robot_id)
                            or self.gate.has_reservation(waiter.robot_id))
        else:
            self.fail(f'physical pick/return stopped at {owner.state.name}, '
                      f'TCP={owner.get_tcp_position()}, segment={owner.current_motion_segment()}: '
                      + self.output.getvalue())
        self.assertTrue({ArmState.GRASP, ArmState.LIFT, ArmState.MOVE_TO_DROP_ABOVE} <= seen)
        self.assertGreater(minimum_gap, .025, 'parallel admission lost physical clearance')
        self.assertIsNone(owner.last_collision)
        self.assertIsNone(waiter.last_collision)


if __name__ == '__main__':
    unittest.main(verbosity=2)
