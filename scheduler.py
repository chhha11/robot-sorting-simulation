"""Distance-greedy dispatch, per-arm destinations, and one safe visual handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import math
import random
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from robot_arm import BaseRobotArm, get_arm_handoff_status, get_arm_last_error

Position = Tuple[float, float, float]


# Keep older callers' type imports working while exposing the common driver
# contract. Runtime registration remains duck typed for adapters and tests.
ArmInterface = BaseRobotArm


@dataclass(frozen=True)
class DetectedObject:
    """A stable VISUAL identity and its latest measured world pick point.

    arm_options adapts optional robot features (e.g. simulation-only virtual
    attachment). The scheduler never uses those options to locate an object.
    """

    track_id: int
    label: str
    target_pos: Position
    arm_options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_vision(cls, item: Mapping[str, Any]) -> "DetectedObject":
        """Adapt the common BaseVision dictionary without knowing its backend.

        arm_options is optional application metadata for the gripper adapter;
        it never changes the measured class, position or visual identity.
        """
        return cls(item["track_id"], item["class"], tuple(item["pos_3d"]),
                   item.get("arm_options", {}))


@dataclass(frozen=True)
class ArmRoute:
    name: str
    arm: BaseRobotArm
    drop_positions: Sequence[Position] = ()


class TaskStatus(Enum):
    RUNNING = "RUNNING"
    PENDING_HANDOFF = "PENDING_HANDOFF"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class AssignedTask:
    track_id: int
    label: str
    arm_name: str
    target_pos: Position
    drop_pos: Position
    started_step: int
    status: TaskStatus = TaskStatus.RUNNING
    finished_step: Optional[int] = None
    error: Optional[str] = None
    handoff_count: int = 0
    failed_arm_name: Optional[str] = None
    handoff_requested_step: Optional[int] = None
    handoff_error: Optional[str] = None


class Scheduler:
    """Route A: each fresh object chooses the nearest idle arm by TCP distance.

    Arm registration and class destinations are independent. Slots are reserved
    per destination; private destinations additionally belong to one arm.
    A session-long identity ledger allows only the explicit, bounded handoff
    path to start a second attempt for the same visual object.
    """

    def __init__(self, routes: Mapping[str, ArmRoute], *, drop_targets=None,
                 pickup_bounds=None, random_seed=None, slot_preferences=None,
                 arm_drop_targets=None, handoff_enabled=True,
                 handoff_timeout_steps=7200, observation_timeout_steps=1440):
        self.routes = dict(routes)
        self.arms = {route.name: route for route in routes.values()}
        if len(self.arms) != len(routes) or len({id(r.arm) for r in routes.values()}) != len(routes):
            raise ValueError("register each physical arm exactly once")
        self.arm_drop_targets = ({name: dict(destinations) for name, destinations in arm_drop_targets.items()}
                                 if arm_drop_targets is not None else None)
        if self.arm_drop_targets is not None:
            if set(self.arm_drop_targets) - self.arms.keys():
                raise ValueError("private destinations must name a registered arm")
            self.drop_targets = {}
            for destinations in self.arm_drop_targets.values():
                for label, positions in destinations.items():
                    if not positions:
                        raise ValueError("private class destinations must have at least one slot")
                    self.drop_targets.setdefault(label, positions)
        else:
            self.drop_targets = dict(drop_targets if drop_targets is not None else
                                     {label: route.drop_positions for label, route in routes.items()})
        if not self.arms or not self.drop_targets or any(not slots for slots in self.drop_targets.values()):
            raise ValueError("arms and nonempty class destinations are required")
        self.pickup_bounds = pickup_bounds
        self.slot_preferences = slot_preferences or {}
        self._rng = random.Random(random_seed)
        self.assigned_tasks: Dict[int, AssignedTask] = {}
        self.active_tasks: Dict[str, int] = {}
        self.pending_tasks: Dict[int, DetectedObject] = {}
        self._slots = {}
        self._finished = []
        self.failed_track_ids = set()
        self._dispatched_track_ids = set()
        self.assignment_log = deque(maxlen=256)
        self.handoff_events = deque(maxlen=256)
        self._seen = {}
        self.handoff_enabled = bool(handoff_enabled)
        self.handoff_timeout_steps = int(handoff_timeout_steps)
        self.observation_timeout_steps = int(observation_timeout_steps)
        if self.handoff_timeout_steps <= 0 or self.observation_timeout_steps <= 0:
            raise ValueError("handoff and observation timeouts must be positive")

    def _destination_slots(self, arm_name, label):
        if self.arm_drop_targets is None:
            return self.drop_targets.get(label, ())
        return self.arm_drop_targets.get(arm_name, {}).get(label, ())

    def _destination_key(self, arm_name, label):
        return (arm_name, label) if self.arm_drop_targets is not None else label

    def _available_slots(self, arm_name, label):
        slots = self._destination_slots(arm_name, label)
        key = self._destination_key(arm_name, label)
        occupied = {slot for destination, slot in self._slots.values() if destination == key}
        return [index for index in range(len(slots)) if index not in occupied]

    def release_destination(self, track_id):
        """Free a terminal task's slot after the caller verifies and collects it."""
        task = self.assigned_tasks.get(track_id)
        if task is None or task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            return False
        return self._slots.pop(track_id, None) is not None

    @staticmethod
    def _canonical_label(label):
        return {"red": "Box_A", "blue": "Box_B"}.get(label.lower(), label)

    def _poll_finished(self, step):
        for arm_name, track_id in list(self.active_tasks.items()):
            arm = self.arms[arm_name].arm
            error = get_arm_last_error(arm)
            if error or arm.is_idle():
                record = self.assigned_tasks[track_id]
                if error and self._can_handoff(record, arm):
                    self._request_handoff(record, step, error)
                else:
                    self._finish_task(record, step, error)

    def _can_handoff(self, record, arm):
        # The arm latches this before releasing/resetting a failed grasp. A
        # human-readable error alone cannot prove that no payload was lifted.
        status = get_arm_handoff_status(arm)
        return (self.handoff_enabled and record.handoff_count == 0
                and status.eligible
                and not status.has_payload
                and not status.collision
                and not status.recovery_blocked
                and any(name != record.arm_name and self._destination_slots(name, record.label)
                        for name in self.arms))

    def _handoff_event(self, event, record, step):
        self.handoff_events.append({
            'event': event, 'track_id': record.track_id, 'step': step,
            'from_arm': record.failed_arm_name,
            'to_arm': record.arm_name if event == 'started' else None,
            'handoff_count': record.handoff_count, 'error': record.error,
        })

    def _request_handoff(self, record, step, error):
        record.status = TaskStatus.PENDING_HANDOFF
        record.error = record.handoff_error = str(error)
        record.failed_arm_name = record.arm_name
        record.handoff_requested_step = step
        record.finished_step = None
        self.active_tasks.pop(record.arm_name, None)
        self._slots.pop(record.track_id, None)
        if record.track_id in self._seen:
            self.pending_tasks[record.track_id] = self._seen[record.track_id][0]
        self._handoff_event('pending', record, step)
        print(f"[Scheduler] ID_{record.track_id} PENDING_HANDOFF from {record.arm_name}: {error}", flush=True)

    def _observed_position(self, item, seen_step, step):
        return item.target_pos

    def _expire_handoffs(self, step):
        for record in list(self.assigned_tasks.values()):
            if record.status != TaskStatus.PENDING_HANDOFF:
                continue
            observed = self._seen.get(record.track_id)
            source = self.arms[record.failed_arm_name].arm
            status = get_arm_handoff_status(source)
            reason = None
            if step - record.handoff_requested_step > self.handoff_timeout_steps:
                reason = "handoff waiting deadline exceeded"
            elif observed is None or step - observed[1] > self.observation_timeout_steps:
                reason = "visual track expired while waiting for handoff"
            elif status.collision or status.recovery_blocked:
                reason = "source arm could not safely recover for handoff"
            elif not self._inside_pickup(self._observed_position(*observed, step)):
                reason = "object left the shared pickup area before handoff"
            if reason:
                self._finish_task(record, step, reason)

    def _finish_task(self, record, step, error=None):
        if record.status not in (TaskStatus.RUNNING, TaskStatus.PENDING_HANDOFF):
            return
        was_handoff = record.status == TaskStatus.PENDING_HANDOFF
        record.finished_step, record.error = step, error
        record.status = TaskStatus.FAILED if error else TaskStatus.COMPLETED
        if error:
            self.failed_track_ids.add(record.track_id)
            self._slots.pop(record.track_id, None)
        if self.active_tasks.get(record.arm_name) == record.track_id:
            del self.active_tasks[record.arm_name]
        self.pending_tasks.pop(record.track_id, None)
        self._finished.append(record)
        if was_handoff:
            self._handoff_event('failed', record, step)
        print(f"[Scheduler] {record.arm_name} ID_{record.track_id} {record.status.value}", flush=True)

    def tick(self, step):
        self._poll_finished(step)
        self._expire_handoffs(step)

    def drain_finished(self):
        events, self._finished = tuple(self._finished), []
        return events

    def _inside_pickup(self, point):
        if self.pickup_bounds is None:
            return True
        xmin, xmax, ymin, ymax = self.pickup_bounds
        return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax

    def _ingest(self, observations, step):
        for observation in observations:
            item = observation if isinstance(observation, DetectedObject) else DetectedObject.from_vision(observation)
            label = self._canonical_label(item.label)
            if label not in self.drop_targets or item.track_id in self.failed_track_ids:
                continue
            if item.track_id in self._dispatched_track_ids:
                task = self.assigned_tasks.get(item.track_id)
                if task and label == task.label:
                    if task.status == TaskStatus.RUNNING:
                        self._refresh_running(task, item, step)
                    elif task.status == TaskStatus.PENDING_HANDOFF:
                        self.pending_tasks[item.track_id] = item
                        self._remember(item, step)
                continue
            self.pending_tasks[item.track_id] = item
            self._remember(item, step)

    def _refresh_running(self, task, item, step):
        self._remember(item, step)

    def _remember(self, item, step):
        self._seen[item.track_id] = (item, step)

    def _fresh(self, item, step):
        return True

    def _command(self, route, item, drop, step):
        route.arm.assign_task(item.target_pos, drop, **dict(item.arm_options))
        return item.target_pos

    @staticmethod
    def _set_traffic_task(route, item, handoff_count):
        """Publish task priority before its first motion update.

        Task allocation stays independent for every idle arm.  The shared
        workspace gate owns physical entry/release and reads these identities
        to prefer an explicit handoff over new work.  Legacy arm adapters can
        omit this optional control interface.
        """
        setter = getattr(route.arm, "set_traffic_task", None)
        if callable(setter):
            setter(track_id=item.track_id, handoff_count=handoff_count)

    def _dispatch(self, step):
        started = []
        # A handoff releases its source arm from ``active_tasks`` before the
        # new assignment is issued. Keep that source reserved until this
        # dispatch pass finishes, so a same-frame fresh object cannot send it
        # back into the shared gate before the receiving arm has entered it.
        handoff_sources = set()
        def pending_handoff(item):
            record = self.assigned_tasks.get(item.track_id)
            return record if record and record.status == TaskStatus.PENDING_HANDOFF else None

        # A recoverable object gets the next safe opportunity, even when a
        # newly detected object has a smaller ID or is nearer the standby arm.
        items = sorted(self.pending_tasks.values(), key=lambda item: (not bool(pending_handoff(item)), item.track_id))
        for item in items:
            handoff = pending_handoff(item)
            fresh = (self._seen.get(item.track_id, (None, None))[1] == step
                     if handoff else self._fresh(item, step))
            if not fresh or not self._inside_pickup(item.target_pos):
                continue
            if item.track_id in getattr(self, 'suppressed_tracks', set()):
                continue
            label = self._canonical_label(item.label)
            if handoff:
                source = self.arms[handoff.failed_arm_name].arm
                # Once an arm has latched a safe pre-grasp failure it is
                # already on its ABORT/RETURN_HOME recovery path.  Dispatch
                # the retry as soon as the other arm is idle; the physical
                # workspace interlock will hold that arm at the boundary until
                # the source has actually cleared the pickup lane.  Waiting
                # for ``source.is_idle()`` here serialized the whole system
                # and could let a fresh object consume the only recipient.
                status = get_arm_handoff_status(source)
                if ((not source.is_idle() and not status.recovering)
                        or status.has_payload
                        or status.recovery_blocked):
                    continue
            # A pending handoff has priority whenever its fresh observation
            # and recovered source make it dispatchable.  It must not reserve
            # every other arm while it waits for a camera frame or source
            # recovery: an idle arm can continue with independent work.  This
            # is the key distinction between task priority and a global
            # scheduler barrier.  ``handoff_sources`` below still protects a
            # source from being reused in the *same* pass in which its retry
            # is issued.
            costs = {name: math.dist(route.arm.get_tcp_position(), item.target_pos)
                     for name, route in self.arms.items()
                     if name != (handoff.failed_arm_name if handoff else None)
                     and name not in handoff_sources
                     and name not in self.active_tasks and route.arm.is_idle()
                     and self._available_slots(name, label)}
            if not costs:
                continue
            minimum = min(costs.values())
            tied = [name for name, cost in costs.items() if math.isclose(cost, minimum, rel_tol=0, abs_tol=1e-9)]
            name = self._rng.choice(tied)
            slots = self._destination_slots(name, label)
            available_slots = self._available_slots(name, label)
            preferred = self.slot_preferences.get(name, {}).get(label, range(len(slots)))
            slot = next((i for i in preferred if i in available_slots), available_slots[0])
            drop = tuple(slots[slot])
            next_handoff_count = handoff.handoff_count + 1 if handoff else 0
            self._set_traffic_task(self.arms[name], item, next_handoff_count)
            target = self._command(self.arms[name], item, drop, step)
            if handoff:
                task = handoff
                task.arm_name, task.target_pos, task.drop_pos = name, tuple(target), drop
                task.started_step = step
                task.status = TaskStatus.RUNNING
                task.finished_step = task.error = None
                task.handoff_count += 1
                self._handoff_event('started', task, step)
                handoff_sources.add(task.failed_arm_name)
                print(f"[Scheduler] Arm {task.failed_arm_name.removeprefix('arm')} failed to grasp "
                      f"ID_{task.track_id}. Handing off to Arm {name.removeprefix('arm')}...", flush=True)
            else:
                task = AssignedTask(item.track_id, label, name, tuple(target), drop, step)
            self._dispatched_track_ids.add(item.track_id)
            self.assigned_tasks[item.track_id] = task
            self.active_tasks[name] = item.track_id
            self._slots[item.track_id] = (self._destination_key(name, label), slot)
            started.append(task)
            self.assignment_log.append({'track_id': item.track_id, 'arm': name, 'costs': costs,
                                        'cost': costs[name], 'drop': drop, 'step': step,
                                        'attempt': task.handoff_count + 1, 'handoff_count': task.handoff_count})
            colour = 'Red' if label == 'Box_A' else 'Blue' if label == 'Box_B' else label
            print(f"[Scheduler] Object ID_{item.track_id} ({colour}) assigned to Arm {name.removeprefix('arm')}. "
                  f"(Cost: {costs[name]:.3f}m) idle_costs={costs}, drop={drop}, "
                  f"attempt={task.handoff_count + 1}", flush=True)
        for task in started:
            self.pending_tasks.pop(task.track_id, None)
        return tuple(started)

    def update(self, detections, step):
        self.tick(step)
        self.pending_tasks = {track_id: item for track_id, item in self.pending_tasks.items()
                              if (record := self.assigned_tasks.get(track_id))
                              and record.status == TaskStatus.PENDING_HANDOFF}
        self._ingest(detections, step)
        self._expire_handoffs(step)
        return self._dispatch(step)

    def fail(self, track_id, step, reason):
        """Force a terminal failure for lost/fallen payloads; never request handoff."""
        if track_id in self.failed_track_ids:
            return
        task = self.assigned_tasks.get(track_id)
        if task is not None:
            if task.status not in (TaskStatus.RUNNING, TaskStatus.PENDING_HANDOFF):
                return
            if task.status == TaskStatus.RUNNING:
                self.arms[task.arm_name].arm.cancel_task(reason)
        elif track_id in self.pending_tasks:
            item = self.pending_tasks.pop(track_id)
            label = self._canonical_label(item.label)
            # An unassigned missed object owns no physical arm.
            task = AssignedTask(track_id, label, 'unassigned', item.target_pos,
                                tuple(self.drop_targets[label][0]), step)
            self.assigned_tasks[track_id] = task
        else:
            return
        self._finish_task(task, step, reason)

    @property
    def completed_count(self):
        return sum(task.status == TaskStatus.COMPLETED for task in self.assigned_tasks.values())


class ConveyorScheduler(Scheduler):
    """Fresh observations drive distance dispatch; motion prediction drives grasp."""

    def __init__(self, routes, velocity=(.05, 0, 0), time_step=1/240,
                 conveyor_end_x=1.54, **options):
        if not math.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be finite and positive")
        if not math.isfinite(conveyor_end_x):
            raise ValueError("conveyor_end_x must be finite")
        options.setdefault('handoff_timeout_steps', max(1, round(30 / time_step)))
        options.setdefault('observation_timeout_steps', max(1, round(6 / time_step)))
        super().__init__(routes, **options)
        self.velocity, self.time_step = tuple(velocity), time_step
        self.conveyor_end_x = float(conveyor_end_x)
        self._seen = {}
        self._terminal_ids = set()
        self._terminal_order = deque()
        self.history = deque(maxlen=256)
        self._completed_count = 0
        self.missed_count = 0
        self.suppressed_tracks = set()

    def predict(self, item, seen_step, step, lead=0):
        elapsed = (step-seen_step) * self.time_step + lead
        return tuple(item.target_pos[i] + self.velocity[i] * elapsed for i in range(3))

    def _observed_position(self, item, seen_step, step):
        return self.predict(item, seen_step, step)

    def _remember(self, item, step):
        self._seen[item.track_id] = (item, step)

    def _fresh(self, item, step):
        return self._seen[item.track_id][1] == step

    def _refresh_running(self, task, item, step):
        self.arms[task.arm_name].arm.update_tracking_observation(item.target_pos, self.velocity)
        self._remember(item, step)

    def _command(self, route, item, drop, step):
        lead = route.arm.estimate_pick_seconds(item.target_pos)
        target = self.predict(item, step, step, lead)
        route.arm.assign_task(target, drop, target_velocity=self.velocity,
                              tracking_position=item.target_pos, **dict(item.arm_options))
        return target

    def tick(self, step):
        super().tick(step)
        for track_id, item in list(self.pending_tasks.items()):
            _, seen = self._seen[track_id]
            position = self.predict(item, seen, step)
            if position[0] > self.conveyor_end_x:
                self.fail(track_id, step, "滑出传送带尽头")
            elif step - seen > self.observation_timeout_steps:
                self.fail(track_id, step, "视觉追踪超时")

    def update(self, detections, step):
        self.tick(step)
        self._ingest(detections, step)
        self._expire_handoffs(step)
        return self._dispatch(step)

    def retire(self, task, delivered, reason=None):
        if (task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                or task.track_id in self._terminal_ids or self.assigned_tasks.get(task.track_id) is not task):
            return
        delivered = bool(delivered and task.status == TaskStatus.COMPLETED)
        task.status = TaskStatus.COMPLETED if delivered else TaskStatus.FAILED
        task.error = None if delivered else reason or task.error or "放置区未收到物体"
        if not delivered:
            self.failed_track_ids.add(task.track_id)
        self._completed_count += int(delivered)
        self.missed_count += int(not delivered)
        self.assigned_tasks.pop(task.track_id, None)
        self.pending_tasks.pop(task.track_id, None)
        self._seen.pop(task.track_id, None)
        self._slots.pop(task.track_id, None)
        self.suppressed_tracks.discard(task.track_id)
        self.history.append(task)
        self._terminal_ids.add(task.track_id)
        self._terminal_order.append(task.track_id)
        if len(self._terminal_order) > 512:
            self._terminal_ids.discard(self._terminal_order.popleft())
        print(f"{'SORTED' if delivered else '漏抓'}: {task.label} #{task.track_id}, "
              f"{task.arm_name}, {task.error or 'confirmed drop / slot recycled'}", flush=True)

    @property
    def completed_count(self):
        return self._completed_count


if __name__ == '__main__':
    import unittest
    unittest.main(module='test_dynamic_scheduler', verbosity=2)
