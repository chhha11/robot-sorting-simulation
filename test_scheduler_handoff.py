"""Exclusive destinations and one safe, observed handoff per visual identity."""
import contextlib
import io
import unittest
from types import SimpleNamespace

from scheduler import ArmRoute, ConveyorScheduler, DetectedObject, Scheduler, TaskStatus
from test_dynamic_scheduler import FakeArm


class HandoffArm(FakeArm):
    def __init__(self, tcp):
        super().__init__(tcp)
        self.handoff_eligible = False
        self.held_object_id = None
        self.last_collision = None
        self._collision_recovery_blocked = False

    def assign_task(self, *args, **kwargs):
        super().assign_task(*args, **kwargs)
        self.handoff_eligible = False


class TrafficHandoffArm(HandoffArm):
    """Record the priority the motion controller sees at command startup."""

    def __init__(self, tcp):
        super().__init__(tcp)
        self.traffic_task = None
        self.traffic_assignments = []

    def set_traffic_task(self, track_id, handoff_count):
        self.traffic_task = (track_id, handoff_count)

    def assign_task(self, *args, **kwargs):
        self.traffic_assignments.append(self.traffic_task)
        super().assign_task(*args, **kwargs)


class SchedulerHandoffTests(unittest.TestCase):
    def setUp(self):
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def build(self, moving=False, traffic=False, **options):
        arm_cls = TrafficHandoffArm if traffic else HandoffArm
        a, b = arm_cls((0, 0, .5)), arm_cls((1, 0, .5))
        cls = ConveyorScheduler if moving else Scheduler
        destinations = {
            'arm1': {'Box_A': ((.4, -.4, .7),), 'Box_B': ((.2, -.4, .7),)},
            'arm2': {'Box_A': ((.4, .4, .7),), 'Box_B': ((.2, .4, .7),)},
        }
        s = cls({'first': ArmRoute('arm1', a), 'second': ArmRoute('arm2', b)},
                arm_drop_targets=destinations, pickup_bounds=(-.4, 1.2, -.1, .1),
                random_seed=7, **options)
        return s, a, b

    @staticmethod
    def item(track_id=1, x=.1, label='Box_B'):
        return DetectedObject(track_id, label, (x, 0, .5), {'object_id': 42})

    def queue_handoff(self, s, a, track_id=1):
        task, = s.update([self.item(track_id)], 0)
        a.handoff_eligible = True
        a.last_error = 'empty fingers after bounded grasp retry'
        a.idle = False
        s.tick(1)
        self.assertEqual(task.status, TaskStatus.PENDING_HANDOFF)
        self.assertFalse(s.drain_finished())
        self.assertNotIn(track_id, s.failed_track_ids)
        return task

    def test_each_idle_arm_receives_traffic_identity_before_motion_command(self):
        """A local traffic token never becomes a task-allocation barrier."""
        for moving in (False, True):
            with self.subTest(moving=moving):
                s, a, b = self.build(moving=moving, traffic=True)
                first, = s.update([self.item(1, .1)], 0)
                self.assertEqual(first.arm_name, 'arm1')
                second, = s.update([self.item(2, .9)], 1)
                self.assertEqual(second.arm_name, 'arm2')
                self.assertEqual(a.traffic_assignments, [(1, 0)])
                self.assertEqual(b.traffic_assignments, [(2, 0)])
                self.assertEqual(set(s.active_tasks), {'arm1', 'arm2'})

    def test_handoff_priority_is_published_before_assign_and_not_inherited_by_new_work(self):
        for moving in (False, True):
            with self.subTest(moving=moving):
                s, a, b = self.build(moving=moving, traffic=True)
                task = self.queue_handoff(s, a, track_id=10)
                a.idle = True
                started = s.update([self.item(10, .2), self.item(2, .9, 'Box_A')], 2)
                self.assertEqual([item.track_id for item in started], [10])
                self.assertIs(started[0], task)
                self.assertEqual(b.traffic_assignments, [(10, 1)])
                # The returning source can take independent work on the next
                # frame, but must retain ordinary priority beneath the retry.
                started = s.update([self.item(10, .25), self.item(2, .9, 'Box_A')], 3)
                self.assertEqual([(item.track_id, item.arm_name) for item in started], [(2, 'arm1')])
                self.assertEqual(a.traffic_assignments, [(10, 0), (2, 0)])
                self.assertEqual(b.traffic_task, (10, 1))

    def test_pending_handoff_does_not_elevate_unrelated_new_task(self):
        s, a, b = self.build(traffic=True)
        self.queue_handoff(s, a, track_id=10)
        # A missing camera observation keeps ID10 pending.  ID2 is fresh and
        # independently dispatchable, and must not acquire handoff priority.
        a.idle = True
        started = s.update([self.item(2, .9, 'Box_A')], 2)
        self.assertEqual([(task.track_id, task.arm_name) for task in started], [(2, 'arm2')])
        self.assertEqual(b.traffic_assignments, [(2, 0)])
        self.assertEqual(s.assigned_tasks[10].status, TaskStatus.PENDING_HANDOFF)

    def test_each_arm_uses_its_own_colour_destination(self):
        for moving in (False, True):
            for label in ('Box_A', 'Box_B'):
                with self.subTest(moving=moving, label=label):
                    s, a, b = self.build(moving)
                    tasks = s.update([self.item(1, .1, label), self.item(2, .9, label)], 0)
                    self.assertEqual([t.arm_name for t in tasks], ['arm1', 'arm2'])
                    self.assertEqual([t.drop_pos for t in tasks],
                                     [s.arm_drop_targets[name][label][0] for name in ('arm1', 'arm2')])
                    self.assertEqual(len(set(s._slots.values())), 2)
                    self.assertEqual(len(a.calls), 1)
                    self.assertEqual(len(b.calls), 1)

    def test_full_private_destination_excludes_only_its_owner(self):
        s, a, b = self.build()
        first, = s.update([self.item()], 0)
        a.idle = True
        s.tick(1)  # Static delivered objects continue occupying their slots.
        self.assertEqual(first.status, TaskStatus.COMPLETED)
        second, = s.update([self.item(2, .05)], 2)
        self.assertEqual(second.arm_name, 'arm2')
        self.assertEqual(second.drop_pos, s.arm_drop_targets['arm2']['Box_B'][0])

    def test_only_verified_terminal_delivery_can_release_static_destination(self):
        s, a, _ = self.build()
        task, = s.update([self.item()], 0)
        self.assertFalse(s.release_destination(task.track_id))
        self.assertIn(task.track_id, s._slots)
        a.idle = True
        s.tick(1)
        self.assertTrue(s.release_destination(task.track_id))
        self.assertIs(s.assigned_tasks[task.track_id], task)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertNotIn(task.track_id, s._slots)
        next_task, = s.update([self.item(2, .05)], 2)
        self.assertEqual(next_task.arm_name, 'arm1')
        self.assertFalse(s.release_destination(9999))

    def test_handoff_requires_recovered_source_and_fresh_measurement(self):
        for moving in (False, True):
            with self.subTest(moving=moving):
                s, a, b = self.build(moving)
                task = self.queue_handoff(s, a)
                first_log = dict(s.assignment_log[0])
                self.assertFalse(s.update([self.item(x=.2)], 2))
                a.idle = True
                self.assertFalse(s.update([], 3), 'handoff used a stale observation')
                second, = s.update([self.item(x=.3)], 4)
                self.assertIs(second, task)
                self.assertEqual(second.handoff_count, 1)
                self.assertEqual(second.failed_arm_name, 'arm1')
                self.assertEqual(second.arm_name, 'arm2')
                self.assertEqual(second.status, TaskStatus.RUNNING)
                self.assertIsNone(second.error)
                self.assertEqual(second.drop_pos, s.arm_drop_targets['arm2']['Box_B'][0])
                self.assertEqual(b.calls[0][2]['object_id'], 42)
                self.assertEqual(b.calls[0][0], (.4 if moving else .3, 0, .5))
                if moving:
                    self.assertEqual(b.calls[0][2]['tracking_position'], (.3, 0, .5))
                self.assertEqual(len(a.calls), 1)
                self.assertEqual(len(b.calls), 1)
                self.assertEqual(s.assignment_log[0], first_log)
                self.assertEqual([row['attempt'] for row in s.assignment_log], [1, 2])
                self.assertEqual([row['handoff_count'] for row in s.assignment_log], [0, 1])
                b.idle = True
                s.tick(5)
                terminal, = s.drain_finished()
                self.assertIs(terminal, task)
                self.assertEqual(terminal.status, TaskStatus.COMPLETED)
                if moving:
                    s.retire(terminal, True)
                    self.assertEqual(s.missed_count, 0)
                    self.assertEqual(len(s.history), 1)
                self.assertEqual(s.completed_count, 1)
                self.assertFalse(s.update([self.item()], 6))

    def test_handoff_dispatches_while_source_is_returning_home(self):
        """A recovering source must not starve its idle handoff recipient."""
        s, a, b = self.build()
        task = self.queue_handoff(s, a)
        # RobotArm exposes ABORT while it retracts toward HOME.  The source
        # still owns the physical reservation, so the recipient's controller
        # may wait at the boundary; scheduling itself should happen now.
        a.state = SimpleNamespace(name='ABORT')
        started = s.update([self.item(x=.2)], 2)
        self.assertIs(started[0], task)
        self.assertEqual(task.arm_name, 'arm2')
        self.assertEqual(task.handoff_count, 1)
        self.assertFalse(a.idle)
        self.assertFalse(b.idle)

    def test_second_failure_is_final_and_never_bounces_to_first_arm(self):
        for moving in (False, True):
            with self.subTest(moving=moving):
                s, a, b = self.build(moving)
                task = self.queue_handoff(s, a)
                a.idle = True
                s.update([self.item(x=.2)], 2)
                b.handoff_eligible = True
                b.last_error = 'second empty grasp'
                s.tick(3)
                terminal, = s.drain_finished()
                self.assertIs(terminal, task)
                self.assertEqual(task.status, TaskStatus.FAILED)
                self.assertEqual(task.handoff_count, 1)
                self.assertIn(1, s.failed_track_ids)
                if moving:
                    s.retire(terminal, False)
                    s.retire(terminal, False)
                    self.assertEqual(s.missed_count, 1)
                b.idle = True
                self.assertFalse(s.update([self.item()], 4))
                self.assertEqual(len(a.calls) + len(b.calls), 2)

    def test_explicit_handoff_log_occurs_only_on_actual_second_assignment(self):
        s, a, _ = self.build()
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            self.queue_handoff(s, a, track_id=3)
            s.update([self.item(3, .2)], 2)
            self.assertNotIn('Handing off', log.getvalue())
            a.idle = True
            s.update([self.item(3, .2)], 3)
            s.update([self.item(3, .25)], 4)
        message = '[Scheduler] Arm 1 failed to grasp ID_3. Handing off to Arm 2...'
        self.assertEqual(log.getvalue().count(message), 1)

    def test_handoff_has_priority_over_smaller_new_ids(self):
        s, a, b = self.build()
        task = self.queue_handoff(s, a, track_id=10)
        a.idle = True
        started = s.update([self.item(2, .9), self.item(10, .2)], 2)
        self.assertIs(started[0], task)
        self.assertEqual(started[0].arm_name, 'arm2')
        self.assertEqual(started[0].handoff_count, 1)
        self.assertEqual(len(started), 1)
        # The source arm is intentionally reopened on the following frame.
        started = s.update([self.item(2, .9), self.item(10, .2)], 3)
        self.assertEqual([(item.track_id, item.arm_name) for item in started], [(2, 'arm1')])

    def test_handoff_source_is_reserved_for_this_dispatch_pass_only(self):
        s, a, b = self.build()
        task = self.queue_handoff(s, a, track_id=10)
        a.idle = True
        # The fresh handoff must start first. The lower-ID normal task would
        # otherwise immediately reuse arm1 in this same pass.
        started = s.update([self.item(2, .9, 'Box_A'), self.item(10, .2, 'Box_B')], 2)
        self.assertEqual([item.track_id for item in started], [10])
        self.assertIs(started[0], task)
        self.assertEqual(task.arm_name, 'arm2')
        self.assertEqual(len(a.calls), 1)
        self.assertEqual(len(b.calls), 1)
        # A subsequent camera frame reopens the source arm to normal greedy
        # work after the receiving arm has begun its task.
        next_started = s.update([self.item(2, .9, 'Box_A'), self.item(10, .25, 'Box_B')], 3)
        self.assertEqual(len(next_started), 1)
        self.assertEqual(next_started[0].track_id, 2)
        self.assertEqual(next_started[0].arm_name, 'arm1')
        self.assertEqual(len(a.calls), 2)

    def test_new_work_uses_idle_arm_while_handoff_waits_for_camera(self):
        s, a, b = self.build()
        self.queue_handoff(s, a, track_id=10)
        a.idle = True
        started = s.update([self.item(2, .9)], 2)
        self.assertEqual([(task.track_id, task.arm_name) for task in started], [(2, 'arm2')])
        self.assertEqual(len(b.calls), 1, 'an idle peer must not be held by a stale handoff')
        self.assertFalse(s.drain_finished())

    def test_new_work_uses_idle_peer_when_failed_source_is_still_recovering(self):
        """A pending retry must not turn the whole scheduler into a barrier."""
        s, a, b = self.build()
        self.queue_handoff(s, a, track_id=10)
        # Arm1 is still recovering.  The same-ID retry cannot start yet, but
        # arm2 is independently idle and should accept a different object.
        started = s.update([self.item(10, .2), self.item(2, .9, 'Box_A')], 2)
        self.assertEqual([(task.track_id, task.arm_name) for task in started], [(2, 'arm2')])
        self.assertEqual(len(a.calls), 1)
        self.assertEqual(len(b.calls), 1)
        self.assertNotIn(10, s.active_tasks.values())
        self.assertEqual(s.assigned_tasks[10].status, TaskStatus.PENDING_HANDOFF)

    def test_unsafe_failure_is_terminal_even_if_eligibility_flag_is_wrong(self):
        for unsafe in ('holding', 'collision', 'blocked', 'not_eligible'):
            with self.subTest(unsafe=unsafe):
                s, a, _ = self.build()
                task, = s.update([self.item()], 0)
                a.last_error = unsafe
                a.handoff_eligible = unsafe != 'not_eligible'
                if unsafe == 'holding':
                    a.held_object_id = 42
                elif unsafe == 'collision':
                    a.last_collision = {'other_body': 99}
                elif unsafe == 'blocked':
                    a._collision_recovery_blocked = True
                s.tick(1)
                terminal, = s.drain_finished()
                self.assertIs(terminal, task)
                self.assertEqual(task.status, TaskStatus.FAILED)
                self.assertEqual(task.handoff_count, 0)

    def test_force_fail_pending_handoff_is_final_without_cancelling_other_arm(self):
        for moving in (False, True):
            with self.subTest(moving=moving):
                s, a, b = self.build(moving)
                task = self.queue_handoff(s, a)
                s.fail(1, 2, 'payload fell to floor')
                terminal, = s.drain_finished()
                self.assertIs(terminal, task)
                self.assertEqual(task.status, TaskStatus.FAILED)
                self.assertFalse(a.cancelled)
                self.assertFalse(b.cancelled)
                self.assertNotIn(1, s.pending_tasks)
                self.assertNotIn(1, s._slots)

    def test_wait_deadline_and_lost_visual_identity_end_handoff_once(self):
        for moving in (False, True):
            for deadline in ('wait', 'vision'):
                with self.subTest(moving=moving, deadline=deadline):
                    options = {'handoff_timeout_steps': 3} if deadline == 'wait' else {'observation_timeout_steps': 3}
                    s, a, b = self.build(moving, **options)
                    task = self.queue_handoff(s, a)
                    if deadline == 'wait':
                        s.update([self.item()], 3)
                    s.tick(6)
                    terminal, = s.drain_finished()
                    self.assertIs(terminal, task)
                    self.assertEqual(task.status, TaskStatus.FAILED)
                    s.tick(7)
                    self.assertFalse(s.drain_finished())
                    self.assertFalse(b.calls)

    def test_object_leaving_pickup_area_cannot_be_handed_off(self):
        s, a, b = self.build()
        task = self.queue_handoff(s, a)
        a.idle = True
        self.assertFalse(s.update([self.item(x=1.3)], 2))
        terminal, = s.drain_finished()
        self.assertIs(terminal, task)
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertFalse(b.calls)

    def test_pending_task_cannot_be_retired_as_a_failed_delivery(self):
        s, a, _ = self.build(moving=True)
        task = self.queue_handoff(s, a)
        s.retire(task, False)
        self.assertEqual(task.status, TaskStatus.PENDING_HANDOFF)
        self.assertEqual(s.missed_count, 0)
        self.assertIs(s.assigned_tasks[1], task)

    def test_conveyor_default_deadlines_follow_its_physics_timestep(self):
        s, a, _ = self.build(moving=True, time_step=.1)
        task = self.queue_handoff(s, a)
        self.assertEqual(s.handoff_timeout_steps, 300)
        self.assertEqual(s.observation_timeout_steps, 60)
        s.tick(61)
        self.assertEqual(s.drain_finished(), (task,))
        self.assertEqual(task.status, TaskStatus.FAILED)

    def test_configured_belt_end_does_not_use_legacy_154m_cutoff(self):
        a, b = HandoffArm((0, 0, .5)), HandoffArm((1, 0, .5))
        s = ConveyorScheduler(
            {'first': ArmRoute('arm1', a), 'second': ArmRoute('arm2', b)},
            arm_drop_targets={
                'arm1': {'Box_B': ((.2, -.4, .7),)},
                'arm2': {'Box_B': ((.2, .4, .7),)},
            }, pickup_bounds=(-.4, 1.2, -.1, .1), conveyor_end_x=2.25,
            random_seed=7,
        )
        a.idle = b.idle = False
        far_on_belt = self.item(1, 1.7)
        self.assertFalse(s.update([far_on_belt], 0))
        s.tick(1)
        self.assertEqual(s.drain_finished(), (), 'x=1.7 is before the configured belt end')
        beyond_end = self.item(1, 2.3)
        s.update([beyond_end], 2)
        s.tick(3)
        task, = s.drain_finished()
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn('传送带尽头', task.error)

    def test_handoff_can_be_disabled_for_legacy_fault_regressions(self):
        s, a, _ = self.build(handoff_enabled=False)
        task, = s.update([self.item()], 0)
        a.handoff_eligible = True
        a.last_error = 'empty grasp'
        s.tick(1)
        self.assertEqual(s.drain_finished(), (task,))
        self.assertEqual(task.status, TaskStatus.FAILED)


if __name__ == '__main__':
    unittest.main(verbosity=2)
