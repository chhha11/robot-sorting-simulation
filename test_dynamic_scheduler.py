"""Route-A invariants: current TCP costs, shared destinations and task lifecycle."""
import contextlib
import io
import math
import unittest

from scheduler import ArmRoute, ConveyorScheduler, DetectedObject, Scheduler, TaskStatus


class FakeArm:
    def __init__(self, tcp):
        self.tcp = tcp
        self.idle = True
        self.last_error = None
        self.calls = []
        self.observations = []
        self.is_tracking_target = True
        self.cancelled = []

    def get_tcp_position(self):
        return self.tcp

    def is_idle(self):
        return self.idle

    def estimate_pick_seconds(self, target):
        return 2

    def assign_task(self, target, drop, **options):
        assert self.idle
        self.calls.append((target, drop, options))
        self.last_error = None
        self.idle = False

    def update_tracking_observation(self, *values):
        self.observations.append(values)

    def cancel_task(self, reason):
        self.cancelled.append(reason)
        self.last_error = reason
        self.idle = False  # recovering, not dispatchable


class DynamicSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def build(self, moving=True, seed=7):
        a, b = FakeArm((0, 0, .5)), FakeArm((1, 0, .5))
        cls = ConveyorScheduler if moving else Scheduler
        instance = cls({'left': ArmRoute('arm1', a), 'right': ArmRoute('arm2', b)},
                       drop_targets={'Box_A': ((.3, .37, .52), (.46, .37, .52)),
                                     'Box_B': ((.3, -.37, .52), (.46, -.37, .52))},
                       random_seed=seed)
        return instance, a, b

    def test_nearest_current_tcp_in_3d_for_both_colours(self):
        for moving in (False, True):
            s, a, b = self.build(moving)
            blue = DetectedObject(1, 'Box_B', (.1, 0, .5))
            task, = s.update([blue], 0)
            self.assertEqual(task.arm_name, 'arm1')
            self.assertEqual(task.drop_pos, s.drop_targets['Box_B'][0])
            self.assertAlmostEqual(s.assignment_log[-1]['cost'], .1)
            a.idle = True
            s.tick(1)
            if moving:
                s.retire(s.drain_finished()[0], True)
            a.tcp = (.9, 0, 2)  # XY-nearest loses on the actual 3D cost
            red = DetectedObject(2, 'Box_A', (.9, 0, .5))
            task, = s.update([red], 2)
            self.assertEqual(task.arm_name, 'arm2')
            self.assertEqual(task.drop_pos, s.drop_targets['Box_A'][0])

    def test_same_colour_burst_uses_both_arms_and_distinct_destination_slots(self):
        s, a, b = self.build()
        tasks = s.update([DetectedObject(1, 'Box_B', (.1, 0, .5)),
                          DetectedObject(2, 'Box_B', (.2, 0, .5))], 0)
        self.assertEqual({t.arm_name for t in tasks}, {'arm1', 'arm2'})
        self.assertEqual(len({t.drop_pos for t in tasks}), 2)
        self.assertTrue(all(t.drop_pos[1] < 0 for t in tasks))
        self.assertEqual(s.update([DetectedObject(1, 'Box_B', (.15, 0, .5))], 1), ())
        self.assertEqual(len(a.calls), 1)
        self.assertEqual(len(a.observations), 1)
        self.assertEqual(len(b.observations), 0)
        s.fail(1, 2, 'test fault')
        self.assertEqual(a.cancelled, ['test fault'])
        self.assertEqual(b.cancelled, [])

    def test_idle_peer_dispatches_while_other_arm_is_busy(self):
        """A busy arm never becomes a global dispatch barrier."""
        for moving in (False, True):
            with self.subTest(moving=moving):
                s, a, b = self.build(moving=moving)
                first, = s.update([DetectedObject(1, 'Box_A', (.1, 0, .5))], 0)
                self.assertEqual(first.arm_name, 'arm1')
                self.assertFalse(a.idle)
                self.assertTrue(b.idle)

                # The second frame contains a new object while arm1 is still
                # moving.  Arm2 must receive it immediately; dispatch must not
                # wait for arm1 to return HOME or for both arms to be idle.
                second, = s.update([DetectedObject(2, 'Box_B', (.9, 0, .5))], 1)
                self.assertEqual(second.arm_name, 'arm2')
                self.assertFalse(b.idle)
                self.assertEqual(set(s.active_tasks), {'arm1', 'arm2'})
                self.assertEqual(len(a.calls), 1)
                self.assertEqual(len(b.calls), 1)

    def test_destination_slot_preference_never_changes_nearest_arm(self):
        s, a, b = self.build()
        s.slot_preferences = {'arm1': {'Box_B': (1, 0)}, 'arm2': {'Box_B': (0, 1)}}
        tasks = s.update([DetectedObject(1, 'Box_B', (.1, 0, .5)),
                          DetectedObject(2, 'Box_B', (.9, 0, .5))], 0)
        self.assertEqual([t.arm_name for t in tasks], ['arm1', 'arm2'])
        self.assertEqual([t.drop_pos for t in tasks],
                         [s.drop_targets['Box_B'][1], s.drop_targets['Box_B'][0]])
        self.assertEqual(len(set(s._slots.values())), 2)

    def test_equal_cost_ties_randomize_and_seed_replays(self):
        selected = []
        for seed in range(20):
            s, a, b = self.build(seed=seed)
            task, = s.update([DetectedObject(1, 'Box_A', (.5, 0, .5))], 0)
            selected.append(task.arm_name)
        self.assertEqual(set(selected), {'arm1', 'arm2'})
        for _ in range(2):
            s, _, _ = self.build(seed=7)
            self.assertEqual(s.update([DetectedObject(1, 'Box_A', (.5, 0, .5))], 0)[0].arm_name,
                             selected[7])

    def test_busy_pending_and_no_dispatch_from_stale_observations(self):
        s, a, b = self.build()
        a.idle = b.idle = False
        item = DetectedObject(1, 'Box_B', (.1, 0, .5))
        self.assertFalse(s.update([item], 0))
        a.idle = True
        self.assertFalse(s.update([], 1))
        self.assertEqual(s.update([item], 2)[0].arm_name, 'arm1')

    def test_shared_bounds_are_not_left_right_ownership(self):
        s, a, b = self.build()
        s.pickup_bounds = (-.2, .25, -.08, .08)
        item = DetectedObject(1, 'Box_B', (0, 0, .52))
        self.assertEqual(s.update([item], 0)[0].arm_name, 'arm1')
        self.assertFalse(s.update([DetectedObject(2, 'Box_A', (.3, 0, .52))], 1))

    def test_failed_id_recovery_gate_and_completed_id_ledger(self):
        s, a, b = self.build()
        b.idle = False
        item = DetectedObject(1, 'Box_B', (.1, 0, .5))
        s.update([item], 0)
        a.last_error = 'collision'
        s.tick(1)
        task, = s.drain_finished()
        self.assertEqual(task.status, TaskStatus.FAILED)
        s.retire(task, True)  # late success cannot overwrite failure
        self.assertEqual(s.completed_count, 0)
        self.assertFalse(s.update([item, DetectedObject(2, 'Box_A', (.1, 0, .5))], 2))
        a.idle = True
        a.last_error = None
        self.assertEqual(s.update([DetectedObject(2, 'Box_A', (.1, 0, .5))], 3)[0].arm_name, 'arm1')
        a.idle = True
        s.tick(4)
        task, = s.drain_finished()
        s.retire(task, True)
        s._terminal_ids.clear()
        self.assertFalse(s.update([item, DetectedObject(2, 'Box_A', (.1, 0, .5))], 5))
        self.assertEqual(len(a.calls), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
