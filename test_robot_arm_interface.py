"""Driver isolation and scheduler behavior through the public arm contract."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest

from robot_arm import ArmHandoffStatus, BaseRobotArm, RealRobotArm, create_robot_arm
from scheduler import ArmRoute, ConveyorScheduler, DetectedObject, Scheduler, TaskStatus


ROOT = Path(__file__).resolve().parent


class ContractArm(BaseRobotArm):
    """A driver with no Bullet identifiers, FSM attributes, or private flags."""

    def __init__(self, tcp):
        self.position = tuple(tcp)
        self.busy = False
        self.error = None
        self.result = ArmHandoffStatus()
        self.commands = []
        self.observations = []
        self.priorities = []

    def assign_task(self, target_pos, drop_pos, **options):
        self.busy = True
        self.error = None
        self.result = ArmHandoffStatus()
        self.commands.append((tuple(target_pos), tuple(drop_pos), options))

    def update(self):
        pass

    def is_idle(self):
        return not self.busy

    def get_last_error(self):
        return self.error

    def get_tcp_position(self):
        return self.position

    def cancel_task(self, reason):
        self.error = reason
        self.busy = False

    def estimate_pick_seconds(self, target_pos):
        return .5

    def update_tracking_observation(self, position, velocity):
        self.observations.append((tuple(position), tuple(velocity)))

    def get_handoff_status(self):
        return self.result

    def set_traffic_task(self, track_id, handoff_count=0):
        self.priorities.append((track_id, handoff_count))


class RobotArmInterfaceTests(unittest.TestCase):
    def setUp(self):
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def subprocess(self, source):
        return subprocess.run([sys.executable, '-c', source], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)

    @staticmethod
    def import_blocker():
        return '''
import importlib.abc
import sys
class RejectSimulation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pybullet', 'pybullet_data', 'sim_robot_arm'}:
            raise ModuleNotFoundError('blocked simulation dependency: ' + fullname)
sys.meta_path.insert(0, RejectSimulation())
'''

    def build(self, conveyor=False):
        first = ContractArm((0, 0, .5))
        second = ContractArm((1, 0, .5))
        destinations = {
            'arm1': {'Box_A': ((.2, -.5, .3),), 'Box_B': ((.4, -.5, .3),)},
            'arm2': {'Box_A': ((.8, .5, .3),), 'Box_B': ((.6, .5, .3),)},
        }
        kind = ConveyorScheduler if conveyor else Scheduler
        scheduler = kind({'left': ArmRoute('arm1', first),
                          'right': ArmRoute('arm2', second)},
                         arm_drop_targets=destinations,
                         pickup_bounds=(-.1, 1.1, -.1, .1), random_seed=7)
        return scheduler, first, second

    @staticmethod
    def item(identity=1, x=.1, label='Box_A'):
        return DetectedObject(identity, label, (x, 0, .5))

    def test_abstract_class_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            BaseRobotArm()

    def test_base_real_and_scheduler_import_without_simulation_dependencies(self):
        result = self.subprocess(self.import_blocker() + '''
from robot_arm import BaseRobotArm, RealRobotArm, create_robot_arm
from scheduler import Scheduler
assert issubclass(RealRobotArm, BaseRobotArm)
arm = create_robot_arm('real', display_name='Interface probe')
assert isinstance(arm, RealRobotArm)
assert not arm.is_idle()
assert 'pybullet' not in sys.modules
assert 'sim_robot_arm' not in sys.modules
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_simulation_alias_and_subclass_preserve_existing_imports(self):
        result = self.subprocess('''
import sys
from robot_arm import BaseRobotArm
assert 'sim_robot_arm' not in sys.modules
from robot_arm import RobotArm, SimRobotArm, SharedWorkspaceGate
assert RobotArm is SimRobotArm
assert issubclass(SimRobotArm, BaseRobotArm)
assert not SimRobotArm.__abstractmethods__
assert SharedWorkspaceGate is not None
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_placeholder_does_not_report_or_simulate_hardware_success(self):
        arm = RealRobotArm(display_name='Unconnected arm')
        self.assertFalse(arm.is_idle())
        self.assertFalse(arm.get_handoff_status().eligible)
        for command in (arm.ensure_ready,
                        lambda: arm.assign_task((0, 0, 0), (1, 0, 0)),
                        arm.update, arm.get_tcp_position,
                        lambda: arm.estimate_pick_seconds((0, 0, 0))):
            with self.subTest(command=command):
                with self.assertRaises(NotImplementedError):
                    command()

    def test_invalid_backend_rejected(self):
        with self.assertRaises(ValueError):
            create_robot_arm('missing-driver')

    def test_real_cli_fails_cleanly_before_loading_simulation(self):
        result = self.subprocess(self.import_blocker() + '''
import runpy
sys.argv = ['main.py', '--robot', 'real', '--headless', '--objects', '1']
runpy.run_path('main.py', run_name='__main__')
''')
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertNotIn('Traceback', output)
        self.assertNotIn('blocked simulation dependency', output)
        self.assertNotIn('unrecognized arguments', output)
        self.assertIn('real', output.lower())

    def test_public_driver_contract_dispatches_each_idle_arm_independently(self):
        for conveyor in (False, True):
            with self.subTest(conveyor=conveyor):
                scheduler, first, second = self.build(conveyor)
                near_left, = scheduler.update([self.item(1, .1)], 0)
                near_right, = scheduler.update([self.item(2, .9, 'Box_B')], 1)
                self.assertEqual((near_left.arm_name, near_right.arm_name), ('arm1', 'arm2'))
                self.assertTrue(first.busy)
                self.assertTrue(second.busy)
                self.assertEqual(first.commands[0][1], (.2, -.5, .3))
                self.assertEqual(second.commands[0][1], (.6, .5, .3))
                self.assertEqual(first.priorities, [(1, 0)])
                self.assertEqual(second.priorities, [(2, 0)])
                self.assertFalse(scheduler.update([self.item(1), self.item(2, .9, 'Box_B')], 2))
                if conveyor:
                    self.assertEqual(first.observations[-1], ((.1, 0, .5), (.05, 0, 0)))
                    self.assertEqual(first.commands[0][2]['tracking_position'], (.1, 0, .5))

    def test_public_failure_status_allows_only_one_same_identity_handoff(self):
        for conveyor in (False, True):
            with self.subTest(conveyor=conveyor):
                scheduler, first, second = self.build(conveyor)
                task, = scheduler.update([self.item()], 0)
                first.error = 'grasp empty'
                first.result = ArmHandoffStatus(eligible=True, recovering=True)
                scheduler.tick(1)
                self.assertEqual(task.status, TaskStatus.PENDING_HANDOFF)
                retry, = scheduler.update([self.item(x=.2)], 2)
                self.assertIs(retry, task)
                self.assertEqual((task.track_id, task.arm_name, task.handoff_count), (1, 'arm2', 1))
                self.assertEqual(second.priorities, [(1, 1)])
                self.assertEqual(second.commands[-1][1], (.8, .5, .3))
                second.error = 'grasp empty again'
                second.result = ArmHandoffStatus(eligible=True, recovering=True)
                scheduler.tick(3)
                self.assertEqual(task.status, TaskStatus.FAILED)
                self.assertIn(1, scheduler.failed_track_ids)
                self.assertFalse(scheduler.update([self.item()], 4))
                self.assertEqual(len(first.commands) + len(second.commands), 2)

    def test_public_cancellation_reaches_driver(self):
        scheduler, first, _ = self.build()
        task, = scheduler.update([self.item()], 0)
        scheduler.fail(task.track_id, 1, 'track expired')
        self.assertEqual(first.get_last_error(), 'track expired')
        self.assertTrue(first.is_idle())
        self.assertEqual(task.status, TaskStatus.FAILED)

    def test_default_handoff_status_requires_explicit_driver_confirmation(self):
        scheduler, first, second = self.build()
        task, = scheduler.update([self.item()], 0)
        # A new hardware adapter need not implement recovery yet. Error text
        # alone must never make the scheduler repeat a possibly lifted grasp.
        first.get_handoff_status = lambda: BaseRobotArm.get_handoff_status(first)
        first.error = 'grasp failed'
        scheduler.tick(1)
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertEqual(task.handoff_count, 0)
        self.assertFalse(second.commands)


if __name__ == '__main__':
    unittest.main()
