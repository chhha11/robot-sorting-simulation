"""Non-blocking UR5 arm controller for the sorting demo.

The controller owns only arm motion and gripper commands.  A caller advances
PyBullet and invokes :meth:`update` once per simulation step.

Run ``python robot_arm.py --headless`` for the two-arm speed/path/placement
regression test. Cycle logs distinguish simulation time from wall time (which
includes the caller's camera, inference and GUI work). The default speed
budget is for the supplied static fixtures using virtual grasp.
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Iterable, Optional, Sequence, Tuple

import pybullet as p

from robot_arm import ArmState, ArmHandoffStatus, BaseRobotArm


class SharedWorkspaceGate:
    """One low pickup corridor token, plus complete-body motion forecasting.

    A token covers entry, grasp and departure, not a complete work cycle.
    Private drop/overhead return motion can overlap the next pickup. Actual
    geometry still has the final say; priority resolves a conflict by moving
    the losing arm away instead of leaving both controllers in a hold.
    """

    def __init__(self, target_clearance=.24, minimum_clearance=.035,
                 center=(0.0, 0.0), radius=.30, ceiling=1.10):
        self.names = {}
        self.arms = {}
        self.reservations = {}
        self.target_clearance = float(target_clearance)
        self.minimum_clearance = float(minimum_clearance)
        self._sequence = 0
        self._client = None
        self._shadows = {}
        self.minimum_observed_clearance = math.inf
        self.wait_events = []
        self.center = tuple(float(v) for v in center)
        self.radius = float(radius)
        self.ceiling = float(ceiling)
        self.tool_margin = .08
        self.requests = {}
        self.traffic_events = []

    def _event(self, event, arm, **details):
        stamp = arm._update_ticks * arm._time_step
        item = dict(time=stamp, event=event, arm_id=arm.robot_id,
                    arm_name=arm.display_name, track_id=arm.traffic_track_id, **details)
        self.traffic_events.append(item)
        identity = f" ID_{arm.traffic_track_id}" if arm.traffic_track_id is not None else ""
        print(f"[Traffic] t={stamp:.3f}s {arm.display_name}{identity}: {event} "
              + " ".join(f"{key}={value}" for key, value in details.items()), flush=True)

    def in_spatial_zone(self, position, margin=0.0):
        return (position[2] < self.ceiling + margin
                and math.dist(position[:2], self.center) < self.radius + margin)

    def priority(self, arm):
        """Payload, handoff, then stable token/FIFO order; no tick-order ties."""
        sequence = self.requests.get(arm.robot_id,
                                    self.reservations.get(arm.robot_id, self._sequence + 1))
        executing = arm.state != ArmState.IDLE
        return (arm.held_object_id is not None, arm.traffic_handoff_count > 0 and executing,
                executing, arm.robot_id in self.reservations, -sequence, -arm.robot_id)

    @property
    def owner(self):
        """The single pickup token owner, independent of busy task states."""
        return next(iter(self.reservations), None)

    def has_reservation(self, arm_id):
        return arm_id in self.reservations

    def register(self, arm, urdf_path):
        if self._client is None:
            self._client = p.connect(p.DIRECT)
        client = self._client
        self.arms[arm.robot_id] = arm
        self.names[arm.robot_id] = arm.display_name
        position, orientation = p.getBasePositionAndOrientation(arm.robot_id)
        shadow = p.loadURDF(str(urdf_path), position, orientation,
                            useFixedBase=True, physicsClientId=client)
        tool = None
        if arm.gripper_id is not None:
            palm = p.createCollisionShape(p.GEOM_BOX, halfExtents=(.055,.055,.018), physicsClientId=client)
            finger = p.createCollisionShape(p.GEOM_BOX, halfExtents=(.012,.018,.055), physicsClientId=client)
            tool = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=palm,
                linkMasses=[0, 0], linkCollisionShapeIndices=[finger, finger],
                linkVisualShapeIndices=[-1, -1],
                linkPositions=[(0,.040,.070),(0,-.040,.070)],
                linkOrientations=[(0,0,0,1)]*2,
                linkInertialFramePositions=[(0,0,0)]*2,
                linkInertialFrameOrientations=[(0,0,0,1)]*2,
                linkParentIndices=[0,0], linkJointTypes=[p.JOINT_PRISMATIC]*2,
                linkJointAxis=[(0,1,0)]*2, physicsClientId=client)
        self._shadows[arm.robot_id] = dict(robot=shadow, tool=tool, payload=None,
                                          payload_source=None)

    def close(self):
        if self._client is not None:
            try:
                p.disconnect(self._client)
            except p.error:
                pass
            self._client = None

    def __del__(self):
        self.close()

    def acquire(self, arm_id):
        if arm_id in self.reservations:
            return True
        arm = self.arms.get(arm_id)
        if arm_id not in self.requests:
            self._sequence += 1
            self.requests[arm_id] = self._sequence
        if self.reservations:
            return False
        # Do not admit an arm while the old owner is still physically in the
        # corridor, even after a cancellation or external release request.
        for other_id, other in self.arms.items():
            if other_id != arm_id and self.in_spatial_zone(other.get_tcp_position(), self.tool_margin):
                return False
        contenders = [candidate for candidate in self.arms.values()
                      if candidate.state in (ArmState.MOVE_TO_TARGET_ABOVE, ArmState.DESCEND,
                                             ArmState.GRASP, ArmState.RETRY)
                      and not candidate.traffic_yielding
                      and candidate._traffic_yield_requested is None
                      and not candidate._collision_recovery_blocked]
        if arm is not None and arm not in contenders:
            contenders.append(arm)
        if arm is not None and contenders and max(contenders, key=self.priority) is not arm:
            return False
        self.reservations[arm_id] = self.requests[arm_id]
        if arm is not None:
            self._event("TOKEN_ACQUIRED", arm, handoff_count=arm.traffic_handoff_count)
        return True

    def release(self, arm_id):
        arm = self.arms.get(arm_id)
        if arm is not None and self.in_spatial_zone(arm.get_tcp_position(), self.tool_margin):
            return False
        released = self.reservations.pop(arm_id, None)
        self.requests.pop(arm_id, None)
        if released is not None and arm is not None:
            self._event("TOKEN_RELEASED", arm)
        return True

    def release_pickup_if_clear(self, arm):
        # Keep the short local reservation until fingers/payload have exited
        # the central pickup corridor. The geometric guard still covers the
        # complete belt and all subsequent private-station/return motion.
        if arm.state in (ArmState.MOVE_TO_DROP_ABOVE, ArmState.DESCEND_TO_DROP,
                         ArmState.RELEASE, ArmState.LIFT_AFTER_RELEASE,
                         ArmState.RETURN_HOME, ArmState.IDLE):
            if (not self.in_spatial_zone(arm.get_tcp_position(), self.tool_margin)
                    and (abs(arm.get_tcp_position()[1]) > .22 or arm.state == ArmState.IDLE)):
                self.release(arm.robot_id)

    @staticmethod
    def _bodies(arm):
        return [body for body in (arm.robot_id, arm.gripper_id, arm.held_object_id)
                if body is not None]

    @staticmethod
    def _distance(first, second, maximum, client=None):
        kwargs = {} if client is None else dict(physicsClientId=client)
        return min((point[8] for a in first for b in second
                    for point in p.getClosestPoints(a, b, distance=maximum, **kwargs)),
                   default=maximum)

    def _forecast_joints(self, arm, horizon, evaluating=False):
        current = tuple(p.getJointState(arm.robot_id, j)[0] for j in arm.arm_joint_indices)
        tcp = arm.get_tcp_position()
        if arm.traffic_yielding:
            if not arm._path_waypoints:
                return current
            target = arm._path_waypoints[-1]
        elif (arm.is_idle() or arm._workspace_waiting
              or (arm._proximity_waiting and not evaluating)):
            return current
        elif (arm.state == ArmState.MOVE_TO_TARGET_ABOVE
              and arm._approach_phase == 0 and not self.has_reservation(arm.robot_id)):
            return current
        elif arm._path_waypoints:
            target = arm._path_waypoints[-1]
        elif arm.state == ArmState.MOVE_TO_TARGET_ABOVE and arm._approach_phase == 0:
            target = (*tcp[:2], arm.shared_transit_height or tcp[2])
        elif (arm.state == ArmState.RETURN_HOME and arm.shared_cell_motion
              and not arm._tcp_return_complete):
            home = arm._initial_tcp_position
            if arm._return_points:
                target = arm._return_points[0]
            else:
                height = max(tcp[2], arm.shared_transit_height or home[2])
                if arm.station_motion and not arm.last_error:
                    height = max(height, home[2])
                    target = (*tcp[:2], height)
                else:
                    target = (tcp[0], home[1], height)
        elif arm.state == ArmState.RETURN_HOME:
            return tuple(a + max(-arm.normal_joint_max_velocity*horizon,
                                 min(arm.normal_joint_max_velocity*horizon, b-a))
                         for a,b in zip(current,arm._initial_joint_positions))
        else:
            return current
        speed = arm.soft_landing_speed if arm.motion_profile == 'SOFT_LANDING' else arm.cartesian_speed
        fraction = min(1., speed*horizon/max(math.dist(tcp,target),1e-12))
        position = tuple(a+(b-a)*fraction for a,b in zip(tcp,target))
        ee_target = arm._tcp_to_ee_position(position, arm.downward_orientation)
        solution = p.calculateInverseKinematics(arm.robot_id,arm.end_effector_link_index,
            targetPosition=ee_target,targetOrientation=arm.downward_orientation,
            lowerLimits=arm.joint_lower_limits,upperLimits=arm.joint_upper_limits,
            jointRanges=arm.joint_ranges,restPoses=current,maxNumIterations=50,
            residualThreshold=.0005)
        return tuple(float(v) for v in solution[:6])

    def _request_yield(self, arm, peer, reason):
        if arm._collision_waiting or arm._collision_recovery_blocked:
            return
        if arm._traffic_yield_requested is None and not arm.traffic_yielding:
            arm._traffic_yield_requested = peer.robot_id
            arm._traffic_yield_reason = reason

    def _route_clear(self, arm, points):
        """Check the complete retreat against each stationary peer assembly.

        Every <= 2 cm sample includes the UR5, tool and attached payload.
        This is planning only in the private Bullet client, never teleporting
        a live robot or loosening actual contact interception.
        """
        if arm.robot_id not in self._shadows:
            return True
        others = [other for other in self.arms.values() if other is not arm]
        if not others:
            return True
        joints = tuple(p.getJointState(arm.robot_id, j)[0] for j in arm.arm_joint_indices)
        peer_bodies = []
        for other in others:
            peer_joints = tuple(p.getJointState(other.robot_id, j)[0] for j in other.arm_joint_indices)
            peer_bodies.extend(self._sync_shadow(other, peer_joints))
        first = self._sync_shadow(arm, joints)
        previous_gap = self._distance(first, peer_bodies, .30, self._client)
        required_gap = min(self.minimum_clearance, previous_gap) - .0001
        start = arm.get_tcp_position()
        may_exit_zone = self.in_spatial_zone(start, self.tool_margin)
        for point in points:
            count = max(1, math.ceil(math.dist(start, point) / .02))
            for sample in range(1, count + 1):
                t = sample / count
                tcp = tuple(a + (b - a) * t for a, b in zip(start, point))
                inside = self.in_spatial_zone(tcp, self.tool_margin)
                if not self.has_reservation(arm.robot_id) and inside and not may_exit_zone:
                    return False
                if not inside:
                    may_exit_zone = False
                solution = p.calculateInverseKinematics(
                    arm.robot_id, arm.end_effector_link_index,
                    targetPosition=arm._tcp_to_ee_position(tcp, arm.downward_orientation),
                    targetOrientation=arm.downward_orientation,
                    lowerLimits=arm.joint_lower_limits, upperLimits=arm.joint_upper_limits,
                    jointRanges=arm.joint_ranges, restPoses=joints,
                    maxNumIterations=100, residualThreshold=.0003)
                joints = tuple(float(v) for v in solution[:6])
                if any(angle < lower - .001 or angle > upper + .001
                       for angle, lower, upper in zip(joints, arm.joint_lower_limits, arm.joint_upper_limits)):
                    return False
                first = self._sync_shadow(arm, joints)
                ee = p.getLinkState(self._shadows[arm.robot_id]['robot'],
                                    arm.end_effector_link_index,
                                    computeForwardKinematics=True, physicsClientId=self._client)
                actual = p.multiplyTransforms(ee[4], ee[5], arm.tcp_offset_local, (0, 0, 0, 1))[0]
                if math.dist(actual, tcp) > .015:
                    return False
                gap = self._distance(first, peer_bodies, .30, self._client)
                if gap < required_gap:
                    return False
                previous_gap = gap
            start = point
        return True

    def plan_yield(self, arm):
        current, home = arm.get_tcp_position(), arm._initial_tcp_position
        base_y = p.getBasePositionAndOrientation(arm.robot_id)[0][1]
        safe_z = max(current[2], home[2])
        # A side parking point keeps even an IDLE arm out of the central
        # approach column while its peer passes underneath the HOME plane.
        parking = (home[0], math.copysign(.18, base_y), home[2])
        vertical = (current[0], current[1], safe_z)
        outer = (current[0], math.copysign(max(.18, abs(current[1])), base_y), current[2])
        previous = getattr(arm, '_segment_start', current)
        routes = [
            [vertical, (home[0], current[1], safe_z), parking],
            [outer, (outer[0], outer[1], safe_z), parking],
            [previous, (previous[0], previous[1], safe_z), parking],
        ]
        for route in routes:
            reduced = []
            start = current
            for point in route:
                if math.dist(start, point) > .004:
                    reduced.append(tuple(point))
                    start = point
            if reduced and self._route_clear(arm, reduced):
                return reduced
        return None

    def _sync_shadow(self, arm, joints):
        record, client = self._shadows[arm.robot_id], self._client
        robot, tool = record['robot'], record['tool']
        for joint, position in zip(arm.arm_joint_indices,joints):
            p.resetJointState(robot,joint,position,physicsClientId=client)
        ee = p.getLinkState(robot,arm.end_effector_link_index,
                            computeForwardKinematics=True,physicsClientId=client)
        bodies = [robot]
        if tool is not None:
            p.resetBasePositionAndOrientation(tool,ee[4],ee[5],physicsClientId=client)
            for shadow_joint,real_joint in enumerate(arm.gripper_joint_indices):
                p.resetJointState(tool,shadow_joint,p.getJointState(arm.gripper_id,real_joint)[0],physicsClientId=client)
            bodies.append(tool)
        payload = arm.held_object_id
        if payload is not None:
            if record['payload_source'] != payload:
                if record['payload'] is not None:
                    p.removeBody(record['payload'],physicsClientId=client)
                shape = p.getCollisionShapeData(payload,-1)[0]
                if shape[2] == p.GEOM_BOX:
                    half = tuple(value/2 for value in shape[3])
                else:
                    low,high=p.getAABB(payload)
                    half=tuple((b-a)/2 for a,b in zip(low,high))
                collision=p.createCollisionShape(p.GEOM_BOX,halfExtents=half,physicsClientId=client)
                record['payload']=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=collision,physicsClientId=client)
                record['payload_source']=payload
            real_ee=p.getLinkState(arm.robot_id,arm.end_effector_link_index,computeForwardKinematics=True)
            inverse=p.invertTransform(real_ee[4],real_ee[5])
            payload_pose=p.getBasePositionAndOrientation(payload)
            local=p.multiplyTransforms(*inverse,*payload_pose)
            predicted=p.multiplyTransforms(ee[4],ee[5],*local)
            p.resetBasePositionAndOrientation(record['payload'],*predicted,physicsClientId=client)
            bodies.append(record['payload'])
        return bodies

    def motion_allowed(self, arm):
        if arm.robot_id not in self.arms:
            return True, None
        self.release_pickup_if_clear(arm)
        # Enforce entry on every planned path, including a replan/return,
        # rather than trusting an earlier assignment-time reservation.
        if not self.has_reservation(arm.robot_id) and not arm.traffic_yielding:
            current = arm.get_tcp_position()
            destination = arm._path_waypoints[-1] if arm._path_waypoints else current
            samples = max(1, math.ceil(math.dist(current, destination) / .02))
            enters = any(self.in_spatial_zone(tuple(a + (b-a)*fraction
                                                    for a,b in zip(current,destination)), self.tool_margin)
                         for fraction in (step / samples for step in range(samples + 1)))
            if enters and not self.acquire(arm.robot_id):
                peer = self.arms.get(self.owner) or next(
                    (other for other in self.arms.values() if other is not arm
                     and self.in_spatial_zone(other.get_tcp_position(), self.tool_margin)), None)
                if peer is not None:
                    if self.in_spatial_zone(current, self.tool_margin):
                        self._request_yield(arm, peer, "central corridor occupied")
                    return False, (peer.robot_id, .30, .30)
                return False, (self.owner, .30, .30)
        for other_id, other in self.arms.items():
            if other_id == arm.robot_id:
                continue
            present=self._distance(self._bodies(arm),self._bodies(other),.30)
            self.minimum_observed_clearance=min(self.minimum_observed_clearance,present)
            if present >= .30:
                continue
            current_a=tuple(p.getJointState(arm.robot_id,j)[0] for j in arm.arm_joint_indices)
            current_b=tuple(p.getJointState(other.robot_id,j)[0] for j in other.arm_joint_indices)
            future_a=self._forecast_joints(arm,.12,evaluating=True)
            future_b=self._forecast_joints(other,.12)
            simultaneous=self_only=.30
            for fraction in (.5,1.):
                first=self._sync_shadow(arm,tuple(a+(b-a)*fraction for a,b in zip(current_a,future_a)))
                second=self._sync_shadow(other,tuple(a+(b-a)*fraction for a,b in zip(current_b,future_b)))
                simultaneous=min(simultaneous,self._distance(first,second,.30,self._client))
                second=self._sync_shadow(other,current_b)
                self_only=min(self_only,self._distance(first,second,.30,self._client))
            # A stationary peer can block even a higher-priority path. It
            # must be asked to retreat; stopping both sides is not a plan.
            if self_only < self.minimum_clearance and self_only < present-.0001:
                if not arm.traffic_yielding:
                    if self.priority(arm) >= self.priority(other):
                        self._request_yield(other, arm, "priority clearance")
                    else:
                        self._request_yield(arm, other, "priority clearance")
                return False,(other_id,present,self_only)
            other_has_motion=any(abs(a-b)>1e-5 for a,b in zip(current_b,future_b))
            if (simultaneous < self.minimum_clearance and simultaneous < present-.0001
                    and other_has_motion):
                if self.priority(arm) < self.priority(other) and not arm.traffic_yielding:
                    self._request_yield(arm, other, "priority crossing")
                    return False,(other_id,present,simultaneous)
                if not other.traffic_yielding:
                    self._request_yield(other, arm, "priority crossing")
        return True,None


class SimRobotArm(BaseRobotArm):
    """Control one loaded UR5 body with a discrete state machine."""

    def __init__(
        self,
        robot_id: int,
        end_effector_link_name: str = "ee_link",
        position_tolerance: float = 0.005,
        command_tolerance: float = 0.005,
        approach_height: float = 0.15,
        lift_height: float = 0.25,
        state_timeout_steps: int = 7200,
        grasp_offset: float = -0.02,
        gripper_id: Optional[int] = None,
        gripper_joint_indices: Optional[Sequence[int]] = None,
        tcp_offset_local: Sequence[float] = (0.0, 0.0, 0.0),
        grasp_offset_z: Optional[float] = None,
        fingertip_offset_local: Sequence[float] = (0.0, 0.0, 0.055),
        virtual_grasp: bool = True,
        target_euler: Optional[Sequence[float]] = None,
        grasp_retry_limit: int = 1,
        grasp_timeout_steps: Optional[int] = None,
        display_name: Optional[str] = None,
        trail_color: Sequence[float] = (1.0, 0.0, 0.0),
    ):
        self.robot_id = robot_id
        self.position_tolerance = position_tolerance
        self.command_tolerance = command_tolerance
        self.approach_height = approach_height
        self.lift_height = lift_height
        self.state_timeout_steps = state_timeout_steps
        if int(grasp_retry_limit) < 0:
            raise ValueError("grasp_retry_limit must be non-negative")
        self.grasp_retry_limit = int(grasp_retry_limit)
        if grasp_timeout_steps is not None and int(grasp_timeout_steps) < 1:
            raise ValueError("grasp_timeout_steps must be positive")
        self.grasp_timeout_steps = (
            int(grasp_timeout_steps) if grasp_timeout_steps is not None else None
        )
        self.display_name = display_name or f"Arm {robot_id}"
        trail = tuple(float(value) for value in trail_color)
        if len(trail) != 3 or not all(0.0 <= value <= 1.0 for value in trail):
            raise ValueError("trail_color must contain three RGB values in [0, 1]")
        self.trail_color = trail
        self.virtual_grasp = bool(virtual_grasp)
        self._time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]
        self.soft_landing_distance = 0.03
        self.soft_landing_force = 10.0
        self.soft_landing_max_velocity = 0.25
        self.normal_joint_force = 300.0
        self.normal_joint_max_velocity = 2.0
        self.cartesian_speed = 0.8  # m/s, free-space segments
        self.soft_landing_speed = 0.06  # m/s, final descent / initial lift 3 cm
        self.cartesian_acceleration = 3.0  # m/s^2; ramp at starts/stops
        self.tracking_limit = 0.003  # bound reference lead; never jump a segment
        self.motor_position_gain = 0.35
        self.ik_max_iterations = 100
        # A virtual payload has no finger contact. Physical mode retains a
        # slower close so speeding up the demo does not increase squeeze force.
        self.gripper_close_speed = 0.08 if self.virtual_grasp else 0.02
        self._task_gripper_close_speed = min(0.02, self.gripper_close_speed)
        self.gripper_open_speed = 0.12
        self.gripper_settle_seconds = 0.08
        self.motion_profile = "NORMAL"
        self.last_error: Optional[str] = None
        self.handoff_eligible = False
        self.failure_phase = None
        # ``grasp_offset`` remains as a backwards-compatible alias.  New
        # callers can use the explicit Z-axis name to make the visual-to-TCP
        # height adjustment clear.
        self.grasp_offset_z = float(
            grasp_offset if grasp_offset_z is None else grasp_offset_z
        )
        self.grasp_offset = self.grasp_offset_z  # compatibility alias
        self._task_grasp_offset_z = self.grasp_offset_z
        self._last_grasp_offset_z = self.grasp_offset_z
        # Euler angles describe the EE/gripper frame in WORLD coordinates,
        # in radians.  Both grippers are mounted with identity rotation.
        # Preserve the measured yaw for the first vertical lift, then turn
        # above the target at safe height before descending with the task quaternion.
        self._default_target_euler = (
            self._validate_target_euler(target_euler) if target_euler is not None else None
        )
        self.target_euler = self._default_target_euler or (math.pi, 0.0, 0.0)
        self.target_orientation = p.getQuaternionFromEuler(self.target_euler)
        self.orientation_tolerance = math.radians(1.0)
        self.orientation_speed = 1.8  # rad/s; safe-height wrist turn
        self.downward_orientation = p.getQuaternionFromEuler((math.pi, 0.0, 0.0))
        self._object_long_axis_world = (1.0, 0.0, 0.0)
        self._orientation_hold_position = None
        self._pose_debug_ids = []
        self.vertical_clearance = 0.25
        self.waypoints_per_segment = 12
        self._segment_start = None
        self._segment_length = 0.0
        self._segment_progress = 0.0
        self._segment_speed = 0.0
        self._cycle_steps = 0
        self._cycle_started_at = None
        self.last_cycle_sim_seconds = None
        self.last_cycle_wall_seconds = None
        self.stage_sim_seconds = {}
        self._motion_tracking = False
        self._conveyor_task = False
        self._tracking_velocity = (0.0, 0.0, 0.0)
        self._tracking_observation = None
        self._tracking_observed_step = 0
        self._moving_descent_z = None
        self.max_tracking_age = 3.5  # wrist can occlude the box for ~2.6s at this camera angle
        self.last_grasp_error = None
        self.pick_time_estimate = 4.2  # approach + descend + finger close, simulation seconds
        self._path_waypoints = []
        self._path_index = 0
        self._approach_phase = 0
        self._target_safe_position: Optional[Tuple[float, float, float]] = None
        self._drop_safe_position: Optional[Tuple[float, float, float]] = None
        self._release_hold_steps = 0
        self._path_debug_ids = []
        self._contact_debug_line_id = -1
        self._last_tcp_debug_step = -1
        self.debug_interval_steps = max(1, round(0.05 / self._time_step))
        self._contact_reported_keys = set()
        self._gripper_opening = False
        self._gripper_open_wait_steps = 0
        self._gripper_open_targets = (0.035, -0.035)
        # Require the real joints to open fully before starting a descent.
        self._gripper_open_tolerance = 0.002
        self._task_object_id: Optional[int] = None
        self._last_object_id: Optional[int] = None
        # Optional simulation-only fallback used after the slow close.  The
        # visual pipeline still supplies the pick point; this
        # constraint only prevents a light object from being launched by
        # small contact impulses in the simplified PyBullet gripper.
        self._virtual_grasp_constraint_id = -1
        self._virtual_grasp_object_id: Optional[int] = None
        self._virtual_collision_pairs = []
        self._object_orientation = (0.0, 0.0, 0.0, 1.0)
        self._object_point_local = (0.0, 0.0, 0.0)
        self._last_object_frame = (self._object_orientation, self._object_point_local)
        self._grasp_hold_position = None
        self._grasp_close_steps = 0
        self._grasp_timeout_steps = 0
        self._grasp_retry_count = 0
        self._retry_phase = 0
        self._retry_open_steps = 0
        self._retry_safe_height = None
        self._abort_safe_position = None
        self._abort_returning_above_home = False
        self._status_debug_id = -1
        self._last_trail_position = None
        self._last_visual_state = None
        self._last_visual_color = None
        self._failure_highlight_steps = 0
        self.last_collision = None
        self.collision_count = 0
        self._collision_waiting = False
        self._collision_recovery_blocked = False
        self._collision_clear_steps = 0
        self._collision_hold_joints = None
        self._collision_hold_fingers = ()
        self._collision_warning_id = -1
        self._collision_warning_steps = 0
        self._cartesian_command = None
        self.state = ArmState.IDLE
        self._state_steps = 0
        self._target_pos: Optional[Tuple[float, float, float]] = None
        self._drop_pos: Optional[Tuple[float, float, float]] = None
        self._last_target_pos: Optional[Tuple[float, float, float]] = None
        self._last_drop_pos: Optional[Tuple[float, float, float]] = None
        self._tcp_reported = False
        self.workspace_gate = None
        self.shared_cell_motion = False  # shared-cell opt-in: avoid sweeping a neighbouring arm
        self.near_pickup_home = False
        self.shared_transit_height = None
        self.station_motion = False  # overhead HOME, own-side direct transfer
        self._workspace_waiting = False
        self._proximity_waiting = False
        self._proximity_hold_joints = None
        self.traffic_track_id = None
        self.traffic_handoff_count = 0
        self.traffic_yielding = False
        self.traffic_yield_count = 0
        self._traffic_yield_requested = None
        self._traffic_yield_reason = None
        self._traffic_yield_points = []
        self._traffic_resume_state = None
        self._traffic_wait_position = None
        self._traffic_stalled_steps = 0
        self._traffic_retry_step = 0
        self._traffic_yield_peer = None
        self._traffic_yield_started = 0
        self._traffic_parking = False
        self._update_ticks = 0
        self.state_events = []
        self._approach_points = []
        self._transport_points = None
        self._return_points = None
        self._tcp_return_complete = False
        self._initial_joint_positions = []

        self.arm_joint_indices = []
        self.gripper_joint_indices = []
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        self.joint_ranges = []
        self.joint_rest_poses = []
        self.end_effector_link_index = -1
        self.gripper_id = gripper_id
        self.tcp_offset_local = _as_position(tcp_offset_local)
        self.fingertip_offset_local = _as_position(fingertip_offset_local)
        self._discover_joints(end_effector_link_name)
        self.downward_orientation = self._initial_downward_orientation()
        self._loaded_orientation = self.downward_orientation
        if self.gripper_id is not None:
            self.gripper_joint_indices = list(
                gripper_joint_indices
                if gripper_joint_indices is not None
                else self._discover_gripper_joints()
            )
        self._initial_joint_positions = [
            p.getJointState(self.robot_id, index)[0] for index in self.arm_joint_indices
        ]
        self._initial_tcp_position = self.get_tcp_position()
        self._gravity_link_masses = [
            (index, p.getDynamicsInfo(self.robot_id, index)[0])
            for index in range(p.getNumJoints(self.robot_id))
        ]
        self._command_initial_joint_pose()

    @staticmethod
    def _validate_target_euler(euler: Sequence[float]) -> Tuple[float, float, float]:
        values = tuple(float(value) for value in euler)
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError("target_euler must be three finite WORLD roll/pitch/yaw radians")
        rotation = p.getMatrixFromQuaternion(p.getQuaternionFromEuler(values))
        if -rotation[8] < math.cos(math.radians(5.0)):
            raise ValueError("target_euler must keep gripper local +Z pointing down (within 5 deg)")
        return values

    @staticmethod
    def _orientation_error(first: Sequence[float], second: Sequence[float]) -> float:
        # q and -q represent the same rotation.
        dot = abs(sum(a * b for a, b in zip(first, second)))
        return 2.0 * math.acos(min(1.0, max(0.0, dot)))

    def _get_ee_orientation(self):
        return p.getLinkState(
            self.robot_id, self.end_effector_link_index, computeForwardKinematics=True
        )[5]

    def _at_orientation(self) -> bool:
        return self._orientation_error(
            self._get_ee_orientation(), self.downward_orientation
        ) <= self.orientation_tolerance

    def _initial_downward_orientation(self):
        """Return the arm's loaded vertical-down orientation with fixed yaw."""
        link_state = p.getLinkState(
            self.robot_id, self.end_effector_link_index, computeForwardKinematics=True
        )
        candidate = link_state[5]
        tool_axis_world = p.multiplyTransforms(
            (0.0, 0.0, 0.0), candidate,
            (0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0),
        )[0]
        if tool_axis_world[2] < -0.7:
            return tuple(float(value) for value in candidate)
        return p.getQuaternionFromEuler((math.pi, 0.0, 0.0))

    def _discover_joints(self, end_effector_link_name: str) -> None:
        for index in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, index)
            joint_type = info[2]
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            if link_name == end_effector_link_name:
                self.end_effector_link_index = index
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                if any(token in joint_name.lower() for token in ("gripper", "finger", "hand")):
                    self.gripper_joint_indices.append(index)
                else:
                    self.arm_joint_indices.append(index)
                    lower, upper = float(info[8]), float(info[9])
                    if lower >= upper:
                        lower, upper = -6.283, 6.283
                    self.joint_lower_limits.append(lower)
                    self.joint_upper_limits.append(upper)
                    self.joint_ranges.append(upper - lower)
                    self.joint_rest_poses.append(float(p.getJointState(self.robot_id, index)[0]))
        if len(self.arm_joint_indices) != 6:
            raise ValueError(
                f"Expected six UR5 arm joints, found {len(self.arm_joint_indices)}"
            )
        if self.end_effector_link_index < 0:
            raise ValueError(f"End-effector link {end_effector_link_name!r} was not found")

    def _discover_gripper_joints(self) -> Sequence[int]:
        return [
            index
            for index in range(p.getNumJoints(self.gripper_id))
            if p.getJointInfo(self.gripper_id, index)[2] == p.JOINT_PRISMATIC
        ]

    def _configure_gripper_contact(self) -> None:
        """Use high friction and zero restitution on the simulated fingers."""
        if self.gripper_id is None:
            return
        for link_index in [-1, *self.gripper_joint_indices]:
            try:
                p.changeDynamics(
                    self.gripper_id,
                    link_index,
                    lateralFriction=1.0,
                    spinningFriction=0.02,
                    rollingFriction=0.02,
                    restitution=0.0,
                )
            except Exception:
                # Real hardware adapters may not expose PyBullet dynamics.
                pass

    @staticmethod
    def _configure_object_contact(object_id: Optional[int]) -> None:
        if object_id is None:
            return
        try:
            p.changeDynamics(
                int(object_id), -1,
                lateralFriction=1.2,
                spinningFriction=0.02,
                rollingFriction=0.02,
                restitution=0.0,
            )
        except Exception:
            pass

    def assign_task(
        self,
        target_pos: Sequence[float],
        drop_pos: Sequence[float],
        grasp_offset: Optional[float] = None,
        grasp_offset_z: Optional[float] = None,
        object_id: Optional[int] = None,
        object_orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        object_point_local: Sequence[float] = (0.0, 0.0, 0.0),
        target_euler: Optional[Sequence[float]] = None,
        object_long_axis_local: Optional[Sequence[float]] = None,
        target_velocity: Sequence[float] = (0.0, 0.0, 0.0),
        tracking_position: Optional[Sequence[float]] = None,
    ) -> None:
        """Assign world-space pick/drop points when the arm is idle.

        ``grasp_offset_z`` adjusts the TCP height relative to the visual point;
        negative values make the fingers descend slightly farther.  The older
        ``grasp_offset`` spelling is accepted for existing callers.  When an
        ``object_id`` is supplied, contact diagnostics are filtered to that
        body and its friction/restitution are configured for soft contact.
        In virtual mode, ``object_orientation`` is the estimated world
        orientation, and ``object_point_local`` locates the visual pick point
        in the object's frame.  They are calibration/vision inputs, never
        read back from the simulated object.  The default pick point is its
        origin; for a sideways cylinder's top, pass (-radius, 0, 0) and its
        known Y=pi/2 orientation.

        ``drop_pos`` is mandatory and independent of the pick point. Like
        ``target_pos``, it denotes the object's top-surface reference point:
        its Z receives the same grasp offset to preserve the held object's
        pose. For a 10 cm tall box resting on a calibrated platform, supply
        the platform surface Z + 0.10, not the platform surface Z itself.

        ``target_euler`` overrides the constructor's WORLD roll/pitch/yaw
        (radians), never base-relative angles.  If neither is provided, local
        +Z points down and local Y (the fingers' opening axis) is perpendicular
        to the object's long edge.  Supply ``object_long_axis_local=(1,0,0)``
        for a box or ``(0,0,1)`` for a cylinder, together with its known/visual
        ``object_orientation``.  Omitting the axis assumes the fixed demo's
        long edge is WORLD X.  No simulated object pose is queried.
        """
        if not self.is_idle():
            raise RuntimeError("Cannot assign a task while the arm is busy")
        if self.virtual_grasp and object_id is not None and self.gripper_id is None:
            raise ValueError("virtual grasp requires an attached gripper")
        orientation = tuple(float(value) for value in object_orientation)
        if len(orientation) != 4 or not all(math.isfinite(v) for v in orientation):
            raise ValueError("object_orientation must contain four finite values")
        norm = math.sqrt(sum(v * v for v in orientation))
        if norm < 1e-9:
            raise ValueError("object_orientation must be a nonzero quaternion")
        self._object_orientation = tuple(v / norm for v in orientation)
        long_axis = (1.0, 0.0, 0.0)
        if object_long_axis_local is not None:
            axis = _as_position(object_long_axis_local)
            if not all(math.isfinite(v) for v in axis):
                raise ValueError("object_long_axis_local must contain finite values")
            long_axis = p.multiplyTransforms(
                (0.0, 0.0, 0.0), self._object_orientation,
                axis, (0.0, 0.0, 0.0, 1.0),
            )[0]
        horizontal_length = math.hypot(long_axis[0], long_axis[1])
        if horizontal_length < 1e-6:
            raise ValueError("object long edge must have a nonzero projection on the belt")
        self._object_long_axis_world = (
            long_axis[0] / horizontal_length, long_axis[1] / horizontal_length, 0.0
        )
        selected_euler = target_euler if target_euler is not None else self._default_target_euler
        if selected_euler is not None:
            self.target_euler = self._validate_target_euler(selected_euler)
        else:
            # The fingers are symmetric: yaw and yaw+pi grasp the SAME two
            # sides, but have different arm IK solutions. Choose the branch
            # closest to this arm's loaded wrist yaw. At an exact 90-deg tie,
            # prefer the negative turn, preserving the mirrored UR5 posture.
            long_yaw = math.atan2(long_axis[1], long_axis[0])
            loaded_yaw = p.getEulerFromQuaternion(self._loaded_orientation)[2]
            delta_yaw = math.remainder(long_yaw - loaded_yaw, math.pi)
            if abs(delta_yaw - math.pi / 2.0) < 1e-6:
                delta_yaw -= math.pi
            self.target_euler = (math.pi, 0.0, loaded_yaw + delta_yaw)
        # Rz(yaw)*Rx(pi): local X follows the long edge; local Y spans its
        # two sides.  A shared WORLD target works for bases differing by pi.
        self.target_orientation = p.getQuaternionFromEuler(self.target_euler)
        self._object_point_local = _as_position(object_point_local)
        self._last_object_frame = (self._object_orientation, self._object_point_local)
        self.last_error = None
        self.handoff_eligible = False
        self.failure_phase = None
        self.last_collision = None
        self._collision_waiting = False
        self._collision_recovery_blocked = False
        self._collision_hold_joints = None
        self._target_pos = _as_position(target_pos)
        self._drop_pos = _as_position(drop_pos)
        if grasp_offset_z is not None:
            self._task_grasp_offset_z = float(grasp_offset_z)
        elif grasp_offset is not None:
            self._task_grasp_offset_z = float(grasp_offset)
        else:
            self._task_grasp_offset_z = self.grasp_offset_z
        self._last_target_pos = self._target_pos
        self._last_drop_pos = self._drop_pos
        self._last_grasp_offset_z = self._task_grasp_offset_z
        self._task_object_id = object_id
        self._last_object_id = object_id
        self._configure_object_contact(object_id)
        self._tcp_reported = False
        self._reset_motion_plan()
        self._tracking_velocity = _as_position(target_velocity)
        self._motion_tracking = any(abs(v) > 1e-8 for v in self._tracking_velocity)
        self._conveyor_task = self._motion_tracking
        self._tracking_observation = _as_position(
            tracking_position if tracking_position is not None else target_pos
        )
        self._tracking_observed_step = 0
        self._moving_descent_z = None
        self.last_grasp_error = None
        # A new assignment gets one fresh retry budget. The budget is not
        # reset by the RETRY/ABORT cleanup paths.
        self._grasp_retry_count = 0
        self._grasp_timeout_steps = 0
        self._retry_phase = 0
        self._retry_open_steps = 0
        self._retry_safe_height = None
        self._abort_safe_position = None
        self._failure_highlight_steps = 0
        self._set_state(ArmState.MOVE_TO_TARGET_ABOVE)
        print(
            f"RobotArm {self.robot_id} target_euler WORLD={self.target_euler}, "
            f"object long axis WORLD={self._object_long_axis_world}"
        )

    @property
    def held_object_id(self):
        return self._virtual_grasp_object_id

    def set_traffic_task(self, track_id, handoff_count=0):
        """Set scheduling metadata before assign_task without changing a track."""
        self.traffic_track_id = track_id
        self.traffic_handoff_count = int(handoff_count)

    def _traffic_hold(self):
        if self._proximity_hold_joints is None:
            self._proximity_hold_joints = tuple(p.getJointState(self.robot_id, joint)[0]
                                                for joint in self.arm_joint_indices)
        self._segment_speed = 0.0
        for joint, angle in zip(self.arm_joint_indices, self._proximity_hold_joints):
            p.setJointMotorControl2(self.robot_id, joint, p.POSITION_CONTROL,
                targetPosition=angle, force=self.normal_joint_force,
                maxVelocity=.05, positionGain=.35)

    def _traffic_wait_tick(self, conflict):
        """Waits do not consume stage deadlines; two stationary seconds replan."""
        current = self.get_tcp_position()
        if (self._traffic_wait_position is None
                or math.dist(current, self._traffic_wait_position) > .002):
            self._traffic_wait_position = current
            self._traffic_stalled_steps = 0
        else:
            self._traffic_stalled_steps += 1
        self._traffic_hold()
        if self._traffic_stalled_steps * self._time_step >= 2.0:
            peer = conflict[0] if conflict else None
            self.workspace_gate._event("WAIT_TIMEOUT", self,
                                       stationary_seconds=round(self._traffic_stalled_steps*self._time_step, 3))
            self._traffic_yield_requested = peer if peer is not None else self.robot_id
            self._traffic_yield_reason = "two-second stationary timeout"
            self._traffic_retry_step = self._update_ticks
            self._traffic_stalled_steps = 0

    def _start_traffic_yield(self):
        gate = self.workspace_gate
        if not gate or self._traffic_yield_requested is None:
            return False
        if self._update_ticks < self._traffic_retry_step:
            self._traffic_wait_tick((self._traffic_yield_requested, .0, .0))
            return True
        route = gate.plan_yield(self)
        if route is None:
            # Recheck changing peer geometry without ever issuing an unsafe
            # blind retreat. Real-contact ABORT handling remains independent.
            self._traffic_retry_step = self._update_ticks + max(1, round(.10/self._time_step))
            self._traffic_hold()
            self._traffic_wait_tick((self._traffic_yield_requested, .0, .0))
            return True
        self._traffic_resume_state = self.state
        self._traffic_yield_peer = self._traffic_yield_requested
        self._traffic_yield_requested = None
        self._traffic_yield_started = self._update_ticks
        self._traffic_parking = False
        self._traffic_returning = False
        self.traffic_yielding = True
        self.traffic_yield_count += 1
        self._traffic_yield_points = route
        self._path_waypoints = []
        self._path_index = 0
        self._proximity_waiting = False
        self._proximity_hold_joints = None
        gate._event("YIELD_STARTED", self, peer_id=self._traffic_yield_peer,
                    reason=self._traffic_yield_reason, resume_state=self.state.name,
                    start_tcp=self.get_tcp_position(), parking_tcp=route[-1])
        return True

    def _traffic_yield_update(self):
        """A nonblocking retreat/park/resume subroutine; never drops the task."""
        gate = self.workspace_gate
        self._state_steps -= 1
        if self.held_object_id is None:
            self.open_gripper()
        if not self._path_waypoints and self._traffic_yield_points:
            self._begin_cartesian_segment(self._traffic_yield_points.pop(0))
        if self._path_waypoints:
            allowed, conflict = gate.motion_allowed(self)
            if not allowed:
                self._proximity_waiting = True
                self._traffic_wait_tick(conflict)
                if self._traffic_yield_requested is not None:
                    # A peer may have moved into the planned retreat. Plan
                    # again from the measured TCP, retaining the resume FSM.
                    route = gate.plan_yield(self)
                    self._traffic_yield_requested = None
                    if route:
                        self._traffic_yield_points = route
                        self._path_waypoints = []
                        gate._event("REPLAN", self, reason="retreat clearance changed")
                return self.state
            self._proximity_waiting = False
            self._proximity_hold_joints = None
            self._traffic_stalled_steps = 0
            if self._follow_cartesian_segment():
                self._path_waypoints = []
                self._path_index = 0
            return self.state
        if not self._traffic_parking:
            self._traffic_parking = True
            self._traffic_parking_started = self._update_ticks
            gate.release(self.robot_id)
            gate._event("YIELD_CLEAR", self, tcp=self.get_tcp_position())
        if self._traffic_returning:
            self._finish_traffic_yield()
            return self.state
        peer = gate.arms.get(self._traffic_yield_peer)
        elapsed = (self._update_ticks-self._traffic_parking_started)*self._time_step
        if elapsed < .20:
            self._traffic_hold()
            return self.state
        if peer is not None and not peer.is_idle():
            # The winner only needs to clear pickup/our return corridor, not
            # complete its private placement and whole HOME cycle.
            passing = (gate.has_reservation(peer.robot_id)
                       or gate.in_spatial_zone(peer.get_tcp_position(), gate.tool_margin)
                       or (peer.state in (ArmState.MOVE_TO_TARGET_ABOVE, ArmState.DESCEND,
                                          ArmState.GRASP, ArmState.RETRY, ArmState.LIFT)
                           and not peer.traffic_yielding))
            if passing:
                self._traffic_hold()
                return self.state
        home = self._initial_tcp_position
        current = self.get_tcp_position()
        route = [(current[0], home[1], current[2]), home]
        if not gate._route_clear(self, route):
            self._traffic_hold()
            return self.state
        self._traffic_returning = True
        self._traffic_yield_points = route
        self._proximity_hold_joints = None
        return self.state

    def _finish_traffic_yield(self):
        previous = self._traffic_resume_state
        self.traffic_yielding = False
        self._traffic_yield_requested = None
        self._traffic_resume_state = None
        self._traffic_yield_points = []
        self._path_waypoints = []
        self._path_index = 0
        self._proximity_waiting = False
        self._workspace_waiting = False
        self._proximity_hold_joints = None
        self._traffic_wait_position = None
        self._traffic_stalled_steps = 0
        self._approach_points = []
        self._transport_points = None
        self._return_points = None
        self._tcp_return_complete = False
        if self.held_object_id is not None:
            self._set_state(ArmState.LIFT)
        elif previous in (ArmState.MOVE_TO_TARGET_ABOVE, ArmState.DESCEND,
                           ArmState.GRASP, ArmState.RETRY):
            self._approach_phase = 0
            self._gripper_opening = False
            self._set_state(ArmState.MOVE_TO_TARGET_ABOVE)
        elif previous == ArmState.IDLE:
            self._command_initial_joint_pose()
            self._set_state(ArmState.IDLE)
        else:
            self._set_state(ArmState.RETURN_HOME)
        self.workspace_gate._event("REPLAN", self, reason="priority peer clear",
                                   resumed_state=self.state.name)

    def _cancel_traffic_yield(self, reason):
        if self.traffic_yielding or self._traffic_yield_requested is not None:
            if self.workspace_gate:
                self.workspace_gate._event("YIELD_CANCELLED", self, reason=reason)
            self._path_waypoints = []
            self._path_index = 0
            self._workspace_waiting = False
            if self.state == ArmState.MOVE_TO_TARGET_ABOVE and self._approach_phase == 0:
                # A suspended approach already left HOME; it must recover,
                # rather than using the unstarted-at-HOME cancellation path.
                self._approach_phase = 1
        self.traffic_yielding = False
        self._traffic_yield_requested = None
        self._traffic_resume_state = None
        self._traffic_yield_points = []
        self._proximity_waiting = False
        self._proximity_hold_joints = None
        self._traffic_wait_position = None
        self._traffic_stalled_steps = 0


    @property
    def is_tracking_target(self):
        return self._motion_tracking and self.state in (
            ArmState.MOVE_TO_TARGET_ABOVE, ArmState.DESCEND, ArmState.GRASP
        )

    def estimate_pick_seconds(self, target_pos):
        """Time to closed fingers; learned from completed approach phases."""
        return self.pick_time_estimate

    def predict_tracked_position(self, seconds_ahead=0.0):
        elapsed = (self._cycle_steps - self._tracking_observed_step) * self._time_step
        return tuple(self._tracking_observation[i] + self._tracking_velocity[i]
                     * (elapsed + seconds_ahead) for i in range(3))

    def update_tracking_observation(self, position, velocity):
        """Accept a fresh RGB/depth observation; extrapolate between frames.

        A partial, occluded box can shift the apparent centroid. Reject large
        innovations instead of steering the gripper into an adjacent box.
        """
        if not self.is_tracking_target:
            return
        predicted = self.predict_tracked_position()
        measured = _as_position(position)
        if math.dist(predicted, measured) > 0.025:
            return
        self._tracking_observation = tuple(0.65 * predicted[i] + 0.35 * measured[i] for i in range(3))
        self._tracking_velocity = _as_position(velocity)
        self._tracking_observed_step = self._cycle_steps

    def cancel_task(self, reason):
        """Cancel a lost conveyor target and retreat using the normal FSM."""
        if not self.is_idle():
            self._abort_task(reason)

    def abort_and_return(self, reason: str) -> None:
        """Report an external fault, release the payload and retreat via ABORT.

        Callers must remove a temporary obstruction before requesting retreat.
        This uses the same bounded recovery as a failed grasp.
        """
        if not self.is_idle() and self.state not in (ArmState.ABORT, ArmState.RETURN_HOME):
            self._abort_grasp(reason)

    def flash_failure(self) -> None:
        """Highlight a fault for one simulated second without changing motion."""
        self._failure_highlight_steps = max(1, round(1.0 / self._time_step))

    def current_motion_segment(self):
        """Return measured TCP and active Cartesian endpoint, or None.

        Exposes read-only diagnostics for the GUI's obstacle test. The start
        always comes from the physical fingers, never a cached command.
        """
        if not self._path_waypoints:
            return None
        return self.get_measured_tcp_position(), tuple(self._path_waypoints[-1])

    def _follow_moving_descent(self, target):
        """Descend vertically in the belt frame while following WORLD X/Y."""
        current = self.get_tcp_position()
        if self._moving_descent_z is None:
            self._moving_descent_z = current[2]
            self._segment_speed = 0.0
        soft = current[2] - target[2] <= self.soft_landing_distance
        speed = self.soft_landing_speed if soft else self.cartesian_speed
        remaining = max(0.0, self._moving_descent_z - target[2])
        speed = min(speed, math.sqrt(2 * self.cartesian_acceleration * remaining))
        if not soft:
            braking = max(0.0, current[2] - target[2] - self.soft_landing_distance - 0.003)
            speed = min(speed, math.sqrt(self.soft_landing_speed ** 2
                                         + 2 * self.cartesian_acceleration * braking))
        self._segment_speed = min(speed, self._segment_speed + self.cartesian_acceleration * self._time_step)
        # Never descend beside an object that the horizontal servo cannot follow.
        if math.dist(current[:2], target[:2]) < 0.008 and abs(current[2] - self._moving_descent_z) < 0.004:
            self._moving_descent_z = max(target[2], self._moving_descent_z - self._segment_speed * self._time_step)
        self._command_pose((target[0], target[1], self._moving_descent_z), soft_landing=soft)
        return self._at_position(target) and self._at_orientation()

    def _clear_path_debug(self) -> None:
        """Remove debug geometry belonging to the previous Cartesian segment."""
        if not p.isConnected():
            self._path_debug_ids.clear()
            return
        for debug_id in self._path_debug_ids:
            try:
                p.removeUserDebugItem(debug_id)
            except Exception:
                # Debug drawing is diagnostic only; it must never stop motion.
                pass
        self._path_debug_ids.clear()

    def _clear_contact_debug(self) -> None:
        """Remove the temporary target-to-TCP diagnostic line."""
        if self._contact_debug_line_id >= 0 and p.isConnected():
            try:
                p.removeUserDebugItem(self._contact_debug_line_id)
            except Exception:
                pass
        self._contact_debug_line_id = -1
        self._last_tcp_debug_step = -1
        self._contact_reported_keys.clear()

    def _reset_motion_plan(self) -> None:
        """Reset all Cartesian-segment bookkeeping for a new task."""
        self._cancel_traffic_yield("motion reset")
        self._cycle_steps = 0
        self._cycle_started_at = perf_counter()
        self.last_cycle_sim_seconds = None
        self.last_cycle_wall_seconds = None
        self.stage_sim_seconds = {}
        self._remove_virtual_grasp_constraint()
        self._task_gripper_close_speed = (
            self.gripper_close_speed
            if self.virtual_grasp and self._task_object_id is not None
            else min(0.02, self.gripper_close_speed)
        )
        self._clear_path_debug()
        self._clear_contact_debug()
        for debug_id in self._pose_debug_ids:
            p.removeUserDebugItem(debug_id)
        self._pose_debug_ids.clear()
        self.downward_orientation = tuple(self._get_ee_orientation())
        self._orientation_hold_position = None
        self._path_waypoints = []
        self._path_index = 0
        self._approach_phase = 0
        self._target_safe_position = None
        self._drop_safe_position = None
        self._release_hold_steps = 0
        self._workspace_waiting = False
        self._proximity_waiting = False
        self._proximity_hold_joints = None
        self._approach_points = []
        self._transport_points = None
        self._return_points = None
        self._tcp_return_complete = False
        self._gripper_opening = False
        self._gripper_open_wait_steps = 0
        self._grasp_hold_position = None
        self._grasp_close_steps = 0
        self._cartesian_command = None
        self._retry_phase = 0
        self._retry_open_steps = 0
        self._retry_safe_height = None
        self._abort_safe_position = None

    def _prepare_virtual_contact(self) -> None:
        """Suppress only this arm's contact with its assigned virtual payload.

        Do this before descending, so the animation cannot launch the object
        during the close interval.  Belt, floor, other objects and other arms
        retain their collisions.  The fixed grasp is created AFTER closing.
        """
        if not self.virtual_grasp or self._task_object_id is None:
            return
        if self._virtual_collision_pairs:
            return
        # Only the simulated finger squeeze uses virtual contact. Preserve
        # palm, wrist and arm collisions even against the assigned payload.
        if self.gripper_id is not None:
            for link_index in self.gripper_joint_indices:
                p.setCollisionFilterPair(
                    self.gripper_id, self._task_object_id, link_index, -1, enableCollision=0,
                )
                self._virtual_collision_pairs.append(
                    (self.gripper_id, self._task_object_id, link_index, -1)
                )

    def _create_virtual_grasp_constraint(self) -> bool:
        """Attach at the real TCP using an object pose estimated from vision."""
        if not self.virtual_grasp or self._task_object_id is None:
            return False
        if self._virtual_grasp_constraint_id >= 0:
            return True
        if self.gripper_id is None or self._target_pos is None:
            return False
        # A distant or failed IK target must never become a successful grasp.
        grasp_target = (
            self._target_pos[0], self._target_pos[1],
            self._target_pos[2] + self._task_grasp_offset_z,
        )
        tcp = self.get_measured_tcp_position()
        if math.dist(tcp, grasp_target) > self.position_tolerance:
            raise RuntimeError("real TCP is outside the virtual grasp tolerance")
        if self._conveyor_task and not p.getClosestPoints(
            self.gripper_id, self._task_object_id, distance=0.015
        ):
            raise RuntimeError("payload is absent at the fingers; refusing a distant attachment")
        self.last_grasp_error = math.dist(tcp, grasp_target)
        ee = p.getLinkState(
            self.robot_id, self.end_effector_link_index,
            computeForwardKinematics=True,
        )
        # Constraint frames are relative to the parent's inertial/COM frame.
        # No object transform is queried.  Translate the observed surface
        # point to the body origin using the supplied object calibration.
        offset = p.rotateVector(self._object_orientation, self._object_point_local)
        estimated_origin = tuple(self._target_pos[i] - offset[i] for i in range(3))
        inverse_parent = p.invertTransform(ee[0], ee[1])
        inverse_object = p.invertTransform(estimated_origin, self._object_orientation)
        parent_frame = p.multiplyTransforms(
            inverse_parent[0], inverse_parent[1], tcp, ee[5]
        )
        child_frame = p.multiplyTransforms(
            inverse_object[0], inverse_object[1], tcp, ee[5]
        )
        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.end_effector_link_index,
            childBodyUniqueId=self._task_object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0.0, 0.0, 0.0),
            parentFramePosition=parent_frame[0],
            childFramePosition=child_frame[0],
            parentFrameOrientation=parent_frame[1],
            childFrameOrientation=child_frame[1],
        )
        self._virtual_grasp_constraint_id = constraint_id
        self._virtual_grasp_object_id = self._task_object_id
        try:
            p.changeConstraint(constraint_id, maxForce=200.0, erp=0.2)
        except p.error:
            self._remove_virtual_grasp_constraint()
            raise
        print(
            f"RobotArm {self.robot_id}: VIRTUAL_GRASP attach "
            f"constraint={constraint_id}, object={self._task_object_id}"
        )
        return True

    def _restore_virtual_collision_pairs(self) -> None:
        """Restore precisely the arm/payload pairs disabled for this task."""
        if p.isConnected():
            for body_a, body_b, link_a, link_b in self._virtual_collision_pairs:
                # The target may have been deleted by a test replay.
                try:
                    p.setCollisionFilterPair(
                        body_a, body_b, link_a, link_b, enableCollision=1
                    )
                except p.error:
                    pass
        self._virtual_collision_pairs.clear()

    def _remove_virtual_grasp_constraint(self, restore_collisions: bool = True) -> None:
        """Detach on release, cancellation, replay, or error without leaking."""
        constraint_id = self._virtual_grasp_constraint_id
        if constraint_id >= 0 and p.isConnected():
            try:
                p.removeConstraint(constraint_id)
                print(f"RobotArm {self.robot_id}: VIRTUAL_GRASP release {constraint_id}")
            except p.error:
                pass
        self._virtual_grasp_constraint_id = -1
        self._virtual_grasp_object_id = None
        if restore_collisions:
            self._restore_virtual_collision_pairs()

    def _compensate_arm_gravity(self) -> None:
        """Model gravity compensation independently of the low-force servo.

        Ten Nm cannot support the UR5's own links at full extension.  Apply
        their weight compensation at each measured COM, leaving payload,
        gripper and all object/environment contacts under normal gravity.
        """
        for index, mass in self._gravity_link_masses:
            if mass > 0.0:
                com = p.getLinkState(self.robot_id, index)[0]
                p.applyExternalForce(
                    self.robot_id, index, (0.0, 0.0, mass * 9.81), com,
                    flags=p.WORLD_FRAME,
                )

    @staticmethod
    def _debug_ids(value):
        """Normalize the scalar/list return variants of PyBullet debug APIs."""
        if isinstance(value, (tuple, list)):
            return [int(item) for item in value]
        return [int(value)]

    def _begin_cartesian_segment(
        self, end_position: Sequence[float], point_count: Optional[int] = None
    ) -> None:
        """Build and draw a straight Cartesian segment from the real TCP pose.

        The start is deliberately read on every segment from ``getLinkState``;
        no cached pose is used.  Each interpolated waypoint is later sent to
        IK with the same downward-facing orientation.
        """
        start = self.get_tcp_position()
        self._cartesian_command = start
        self._segment_start = start
        self._segment_progress = 0.0
        self._segment_speed = 0.0
        end = _as_position(end_position)
        count = max(2, int(point_count or self.waypoints_per_segment))
        distance = sum((start[i] - end[i]) ** 2 for i in range(3)) ** 0.5
        self._segment_length = distance
        if distance <= 1.0e-6:
            self._path_waypoints = [end]
        else:
            self._path_waypoints = [
                tuple(
                    start[axis] * (1.0 - t) + end[axis] * t
                    for axis in range(3)
                )
                for t in (index / count for index in range(1, count + 1))
            ]
        self._path_index = 0

        # Display the currently planned segment in blue.  The debug geometry
        # is cleared only when a new segment starts, so it remains visible
        # while that segment is being executed.
        self._clear_path_debug()
        if p.isConnected() and self._path_waypoints:
            point_id = p.addUserDebugPoints(
                pointPositions=self._path_waypoints,
                pointColorsRGB=[[0.0, 0.25, 1.0]] * len(self._path_waypoints),
                pointSize=5.0,
                lifeTime=0.0,
            )
            self._path_debug_ids.extend(self._debug_ids(point_id))
            # All waypoints are collinear; one line draws the same path and
            # avoids 11 synchronous GUI requests (and 11 later removals).
            line_id = p.addUserDebugLine(
                lineFromXYZ=start, lineToXYZ=end,
                lineColorRGB=[0.0, 0.25, 1.0], lineWidth=1.5, lifeTime=0.0,
            )
            self._path_debug_ids.extend(self._debug_ids(line_id))

    def _follow_cartesian_segment(
        self, soft_landing_z: Optional[float] = None,
        soft_lift_z: Optional[float] = None,
    ) -> bool:
        """Track a continuous straight line; settle only at the segment end.

        The 12 blue points visualize the line, not 12 stop-and-go commands.
        Reference speed ramps in simulation time and waits if the real TCP
        falls behind. Low-force control uses measured height, never a future
        waypoint. Both descents and the first 3 cm of lifts stay gentle.
        """
        if self._path_index >= len(self._path_waypoints):
            return True
        destination = self._path_waypoints[-1]
        current = self.get_tcp_position()
        soft_landing = (
            soft_landing_z is not None
            and current[2] - soft_landing_z <= self.soft_landing_distance
        ) or (
            soft_lift_z is not None
            and current[2] - soft_lift_z <= self.soft_landing_distance
        )
        speed = self.soft_landing_speed if soft_landing else self.cartesian_speed
        remaining = max(0.0, self._segment_length - self._segment_progress)
        # Brake smoothly before entering the contact band; motor force and
        # joint limit change ONLY once the measured TCP enters that band.
        if soft_landing_z is not None and not soft_landing:
            brake_distance = max(
                0.0, current[2] - soft_landing_z - self.soft_landing_distance
                - self.tracking_limit,
            )
            speed = min(speed, math.sqrt(
                self.soft_landing_speed ** 2 + 2 * self.cartesian_acceleration * brake_distance
            ))
        speed = min(speed, math.sqrt(2 * self.cartesian_acceleration * remaining))
        self._segment_speed = min(
            speed, self._segment_speed + self.cartesian_acceleration * self._time_step
        )
        command = self._cartesian_command or current
        # Do not let the reference run ahead of the physical TCP.
        if math.dist(current, command) <= self.tracking_limit:
            self._segment_progress = min(
                self._segment_length,
                self._segment_progress + self._segment_speed * self._time_step,
            )
            fraction = min(1.0, self._segment_progress / max(self._segment_length, 1e-12))
            command = tuple(
                self._segment_start[i] + (destination[i] - self._segment_start[i]) * fraction
                for i in range(3)
            )
            if self._segment_length <= 1e-6:
                command = destination
            self._cartesian_command = command
        self._command_pose(command, soft_landing=soft_landing)
        # Intermediate waypoints are passed without a settle condition.
        self._path_index = min(
            len(self._path_waypoints) - 1,
            int(self._segment_progress / max(self._segment_length, 1e-12)
                * len(self._path_waypoints)),
        )
        tolerance = min(self.command_tolerance, self.position_tolerance)
        if (
            math.dist(command, destination) < 1e-8
            and self._at_position(destination, tolerance)
            and self._at_orientation()
        ):
            self._path_index = len(self._path_waypoints)
        return self._path_index >= len(self._path_waypoints)

    def _align_orientation_at_safe_height(self) -> bool:
        """Turn slowly in place after the initial lift, using real pose feedback."""
        if self._orientation_hold_position is None:
            self._orientation_hold_position = self.get_tcp_position()
            self.downward_orientation = tuple(self._get_ee_orientation())
        current = self._get_ee_orientation()
        command = self.downward_orientation
        target = self.target_orientation
        # Follow the shorter quaternion arc, without Euler-angle wrapping.
        dot = sum(a * b for a, b in zip(command, target))
        if dot < 0.0:
            target = tuple(-value for value in target)
            dot = -dot
        remaining = 2.0 * math.acos(min(1.0, max(0.0, dot)))
        if self._orientation_error(current, command) <= math.radians(3.0):
            fraction = min(1.0, self.orientation_speed * self._time_step / max(remaining, 1e-12))
            if dot > 0.9995:
                interpolated = tuple(a + (b - a) * fraction for a, b in zip(command, target))
            else:
                half_angle = math.acos(min(1.0, dot))
                first_weight = math.sin((1.0 - fraction) * half_angle) / math.sin(half_angle)
                second_weight = math.sin(fraction * half_angle) / math.sin(half_angle)
                interpolated = tuple(
                    a * first_weight + b * second_weight for a, b in zip(command, target)
                )
            norm = math.sqrt(sum(value * value for value in interpolated))
            self.downward_orientation = tuple(value / norm for value in interpolated)
        self._command_pose(self._orientation_hold_position)
        return (
            self._orientation_error(self.downward_orientation, self.target_orientation) < 1e-5
            and self._orientation_error(current, self.target_orientation) <= self.orientation_tolerance
            and self._at_position(self._orientation_hold_position, self.command_tolerance)
        )

    def replay_task(self) -> None:
        """Restart the last task from the current measured posture."""
        if self._last_target_pos is None or self._last_drop_pos is None:
            raise RuntimeError("No previous task is available to replay")
        self._target_pos = self._last_target_pos
        self._drop_pos = self._last_drop_pos
        self._task_grasp_offset_z = self._last_grasp_offset_z
        self._task_object_id = self._last_object_id
        self._object_orientation, self._object_point_local = self._last_object_frame
        self.last_error = None
        self.handoff_eligible = False
        self.failure_phase = None
        self._grasp_retry_count = 0
        self._grasp_timeout_steps = 0
        # Detach before a replay can issue any new motion command.
        self._reset_motion_plan()
        self.open_gripper()
        self._state_steps = 0
        self._tcp_reported = False
        self._set_state(ArmState.MOVE_TO_TARGET_ABOVE)

    def open_gripper(self) -> None:
        """Command the two prismatic fingers to open."""
        self._command_gripper(
            self._gripper_open_targets, force=10.0, max_velocity=self.gripper_open_speed
        )

    def close_gripper(self) -> None:
        """Command the two prismatic fingers to close around the TCP."""
        self._command_gripper((0.0, 0.0), force=5.0, max_velocity=self._task_gripper_close_speed)

    def _command_gripper(
        self, targets: Sequence[float], force: float, max_velocity: float
    ) -> None:
        if self.gripper_id is None:
            return
        for joint_index, target in zip(self.gripper_joint_indices, targets):
            p.setJointMotorControl2(
                self.gripper_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=float(target),
                force=float(force),
                maxVelocity=float(max_velocity),
            )

    def _is_gripper_open(self) -> bool:
        """Return whether both fingers reached the commanded open positions."""
        if self.gripper_id is None or len(self.gripper_joint_indices) < 2:
            return True
        return all(
            abs(
                p.getJointState(self.gripper_id, joint_index)[0] - float(target)
            )
            <= self._gripper_open_tolerance
            for joint_index, target in zip(
                self.gripper_joint_indices, self._gripper_open_targets
            )
        )

    def _grasp_object_present(self) -> Optional[bool]:
        """Check whether the assigned payload is near the gripper.

        The body handle is supplied by the visual/simulation adapter. This
        method only asks Bullet for nearest/contact geometry; it never reads
        the payload pose. ``None`` means no payload handle was supplied, so a
        real hardware adapter can use its own tactile/current feedback.
        """
        if self._task_object_id is None:
            return None
        if self.gripper_id is None and self.robot_id is None:
            return False
        try:
            distance = 0.05
            if self.gripper_id is not None:
                if p.getContactPoints(self.gripper_id, self._task_object_id):
                    return True
                if p.getClosestPoints(self.gripper_id, self._task_object_id, distance):
                    return True
            if p.getContactPoints(self.robot_id, self._task_object_id):
                return True
            return bool(p.getClosestPoints(self.robot_id, self._task_object_id, distance))
        except (p.error, TypeError, ValueError):
            # A removed/missed simulation body is a failed grasp, not a
            # reason to strand the state machine.
            return False

    def _grasp_failure(self, reason: str) -> None:
        """Retry a failed grasp once, then enter the scheduler-visible abort."""
        if self._grasp_retry_count < self.grasp_retry_limit:
            self._grasp_retry_count += 1
            print(
                f"RobotArm {self.robot_id} grasp retry "
                f"{self._grasp_retry_count}/{self.grasp_retry_limit}: {reason}",
                flush=True,
            )
            self._remove_virtual_grasp_constraint()
            self.open_gripper()
            current = self.get_tcp_position()
            self._retry_safe_height = max(
                current[2],
                self._target_safe_position[2] if self._target_safe_position else
                current[2] + min(self.vertical_clearance, self.approach_height),
            )
            self._retry_phase = 0
            self._retry_open_steps = 0
            self._path_waypoints = []
            self._path_index = 0
            self._begin_cartesian_segment(
                (current[0], current[1], self._retry_safe_height)
            )
            self._set_state(ArmState.RETRY)
            return
        self._abort_grasp(reason)

    def _abort_grasp(self, reason: str) -> None:
        """Enter ABORT, retain the failure for Scheduler, and retreat safely."""
        self._cancel_traffic_yield(reason)
        # Latch BEFORE cleanup removes the grasp constraint or changes state.
        # Only an uncollected target can be offered to the other station.
        self.failure_phase = self.state.name
        unsafe = any(word in reason.lower() for word in
                     ("lost", "expired", "collision", "removed", "cancel", "掉落", "滑出", "追踪"))
        self.handoff_eligible = (
            self.state in (ArmState.MOVE_TO_TARGET_ABOVE, ArmState.DESCEND,
                           ArmState.GRASP, ArmState.RETRY)
            and self.held_object_id is None and self.last_collision is None and not unsafe
        )
        if self._cancel_waiting_home(reason):
            return
        print(f"RobotArm {self.robot_id} grasp aborted: {reason}", flush=True)
        self.last_error = reason
        self._failure_highlight_steps = round(1.0 / self._time_step)
        self._remove_virtual_grasp_constraint()
        self._motion_tracking = False
        self.open_gripper()
        current = self.get_tcp_position()
        # An originally IDLE arm may be yielding for a peer without a pick
        # task. Cancellation/contact must still finish physical recovery;
        # the generic missing-target shortcut must not mark it IDLE midair.
        if self._target_pos is None:
            self._target_pos = current
        if self._drop_pos is None:
            self._drop_pos = current
        # Reuse the validated approach plane. A fault can occur while already
        # above the object; another unconditional lift could be unreachable.
        safe_height = max(
            current[2],
            self._target_safe_position[2] if self._target_safe_position else
            current[2] + (0.0 if self.station_motion else min(self.vertical_clearance, self.approach_height)),
            (self.shared_transit_height or current[2]) if self.near_pickup_home else current[2],
        )
        self._abort_safe_position = (current[0], current[1], safe_height)
        # After descending/grasping, first clear the drop platforms on the
        # way back. Early-approach failures retain their loaded wrist posture
        # and the original short return; the extra Cartesian leg can cross a
        # singularity before that posture has aligned for grasping.
        self._abort_returning_above_home = self.near_pickup_home or self.state not in (
            ArmState.DESCEND, ArmState.GRASP, ArmState.RETRY,
        )
        self._path_waypoints = []
        self._path_index = 0
        self._begin_cartesian_segment(self._abort_safe_position)
        self._set_state(ArmState.ABORT)

    def _grasp_timeout_limit(self) -> int:
        """Allow the commanded close plus settling, with a finite upper bound."""
        if self.grasp_timeout_steps is not None:
            return self.grasp_timeout_steps
        # A default of 2.5 seconds is long enough for the physical gripper's
        # slow close and short enough to report a stalled actuator promptly.
        return max(self._grasp_close_steps + round(0.5 / self._time_step),
                   round(2.5 / self._time_step))

    def _command_pose(
        self, position: Sequence[float], soft_landing: bool = False
    ) -> None:
        # PyBullet's IK targetPosition and getLinkState(...)[4] are both in
        # world coordinates.  The caller therefore supplies world coordinates;
        # no second base-frame transform is applied here.
        orientation = self.downward_orientation
        self.motion_profile = "SOFT_LANDING" if soft_landing else "NORMAL"
        ee_target = self._tcp_to_ee_position(position, orientation)
        # Use this arm's current joint angles as the solver seed. This keeps
        # nearby targets on the same IK branch and prevents solution jumping.
        current_joint_positions = [state[0] for state in p.getJointStates(
            self.robot_id, self.arm_joint_indices
        )]
        solution = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_link_index,
            targetPosition=ee_target,
            targetOrientation=orientation,
            lowerLimits=self.joint_lower_limits,
            upperLimits=self.joint_upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=current_joint_positions,
            maxNumIterations=self.ik_max_iterations,
            residualThreshold=0.0005,
        )
        for joint_index, target, lower, upper in zip(
            self.arm_joint_indices,
            solution[:6],
            self.joint_lower_limits,
            self.joint_upper_limits,
        ):
            target = min(max(float(target), lower), upper)
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=target,
                positionGain=self.motor_position_gain,
                # The last three centimetres are deliberately gentle.  A
                # low force and velocity lets the fingertips settle around a
                # simplified collision shape instead of kicking it sideways.
                force=self.soft_landing_force if soft_landing else self.normal_joint_force,
                maxVelocity=(
                    self.soft_landing_max_velocity if soft_landing
                    else self.normal_joint_max_velocity
                ),
            )

    def _tcp_to_ee_position(
        self, tcp_position: Sequence[float], ee_orientation: Sequence[float]
    ) -> Tuple[float, float, float]:
        offset_world = p.multiplyTransforms(
            (0.0, 0.0, 0.0), ee_orientation,
            self.tcp_offset_local, (0.0, 0.0, 0.0, 1.0)
        )[0]
        return tuple(float(tcp_position[i]) - offset_world[i] for i in range(3))

    def get_tcp_position(self) -> Tuple[float, float, float]:
        """Return the control TCP from the current physical EE link state."""
        link_state = p.getLinkState(
            self.robot_id, self.end_effector_link_index, computeForwardKinematics=True
        )
        return tuple(
            p.multiplyTransforms(
                link_state[4], link_state[5], self.tcp_offset_local,
                (0.0, 0.0, 0.0, 1.0),
            )[0]
        )

    def get_measured_tcp_position(self) -> Tuple[float, float, float]:
        """Return the physical midpoint of the two measured finger tips."""
        if self.gripper_id is not None and len(self.gripper_joint_indices) >= 2:
            fingertip_positions = []
            for link_index in self.gripper_joint_indices[:2]:
                link_state = p.getLinkState(
                    self.gripper_id, link_index, computeForwardKinematics=True
                )
                fingertip_positions.append(
                    p.multiplyTransforms(
                        link_state[4], link_state[5], self.fingertip_offset_local,
                        (0.0, 0.0, 0.0, 1.0),
                    )[0]
                )
            return tuple(
                float((fingertip_positions[0][axis] + fingertip_positions[1][axis]) / 2.0)
                for axis in range(3)
            )
        return self.get_tcp_position()

    def _at_position(
        self, position: Sequence[float], tolerance: Optional[float] = None
    ) -> bool:
        current = self.get_tcp_position()
        error = sum((float(a) - float(b)) ** 2 for a, b in zip(current, position)) ** 0.5
        return error <= (self.position_tolerance if tolerance is None else tolerance)

    def _set_state(self, state: ArmState) -> None:
        previous = self.state
        self.state = state
        self._state_steps = 0
        if previous != state:
            seconds = self._update_ticks * self._time_step
            self.state_events.append((seconds, previous.name, state.name))
            print(f"[State] t={seconds:.3f}s {self.display_name}: {previous.name} -> {state.name}", flush=True)

    def _timed_out(self) -> bool:
        return self._state_steps > self.state_timeout_steps

    def _cancel_waiting_home(self, reason: str) -> bool:
        """An unstarted reservation can be cancelled without entering the shared space."""
        unstarted = (self.state == ArmState.MOVE_TO_TARGET_ABOVE
                     and self._approach_phase == 0 and not self._path_waypoints)
        if not (self.near_pickup_home and (self._workspace_waiting or unstarted) and self.workspace_gate
                and not self.workspace_gate.has_reservation(self.robot_id)):
            return False
        self.last_error = reason
        if self.last_collision is not None:
            # A contact at standby is not a cancelled reservation. Keep this
            # arm unavailable without sending a retreat through the neighbour.
            self._block_collision_recovery("collision while waiting at HOME")
            return True
        if not self._at_initial_joint_pose():
            self._block_collision_recovery("waiting arm displaced from HOME")
            return True
        self._reset_motion_plan()
        self._motion_tracking = False
        self._target_pos = self._drop_pos = None
        self._command_initial_joint_pose()
        self._set_state(ArmState.IDLE)
        return True

    def _abort_task(self, reason: str) -> None:
        """Stop in place on a timeout instead of advancing into a collision."""
        self._cancel_traffic_yield(reason)
        if self.near_pickup_home:
            if self._cancel_waiting_home(reason):
                return
            if self.state in (ArmState.ABORT, ArmState.RETURN_HOME):
                self._block_collision_recovery(reason)
            else:
                self._abort_grasp(reason)
            return
        if self.last_collision is not None and self.state == ArmState.RETURN_HOME:
            self._block_collision_recovery(reason)
            return
        print(f"RobotArm {self.robot_id} motion aborted: {reason}")
        self.last_error = reason
        self._failure_highlight_steps = round(1.0 / self._time_step)
        hold = self.get_tcp_position()
        self._command_pose(hold, soft_landing=True)
        self.open_gripper()
        self._reset_motion_plan()
        self._motion_tracking = False
        if self._conveyor_task and self.state in (
            ArmState.MOVE_TO_TARGET_ABOVE, ArmState.DESCEND, ArmState.GRASP,
        ):
            # A failed moving pick must free the reservation AND leave the
            # arm ready for the next object. Retreat vertically before home.
            self._task_object_id = None
            self._target_pos = hold
            self._drop_pos = hold
            self._drop_safe_position = (hold[0], hold[1], max(hold[2], 0.70))
            self._set_state(ArmState.LIFT_AFTER_RELEASE)
            return
        self._target_pos = None
        self._drop_pos = None
        self._set_state(ArmState.IDLE)

    def _print_tcp_error(self, target: Sequence[float]) -> None:
        actual = self.get_measured_tcp_position()
        error = sum((float(a) - float(b)) ** 2 for a, b in zip(actual, target)) ** 0.5
        print(
            f"RobotArm {self.robot_id} grasp TCP: "
            f"target=({target[0]:.6f}, {target[1]:.6f}, {target[2]:.6f}), "
            f"actual=({actual[0]:.6f}, {actual[1]:.6f}, {actual[2]:.6f}), "
            f"error={error:.6f} m"
        )

    def get_gripper_alignment(self):
        """Measure the finger frame and opening axis; never read the object pose."""
        if self.gripper_id is None or len(self.gripper_joint_indices) < 2:
            raise RuntimeError("gripper-axis diagnostics require two attached fingers")
        first, second = [
            p.getLinkState(self.gripper_id, index, computeForwardKinematics=True)
            for index in self.gripper_joint_indices[:2]
        ]
        rotation = p.getMatrixFromQuaternion(first[5])
        axes = tuple(tuple(rotation[column::3]) for column in range(3))
        separation = tuple(first[4][i] - second[4][i] for i in range(3))
        length = math.sqrt(sum(value * value for value in separation))
        if length < 1e-9:
            raise RuntimeError("cannot measure opening axis from coincident finger links")
        opening_axis = tuple(value / length for value in separation)
        joint_axis = p.getJointInfo(self.gripper_id, self.gripper_joint_indices[0])[13]
        joint_axis_world = p.multiplyTransforms(
            (0.0, 0.0, 0.0), first[5], joint_axis, (0.0, 0.0, 0.0, 1.0)
        )[0]
        return {
            "euler_world": p.getEulerFromQuaternion(first[5]),
            "axes_world": axes,
            "joint_axis_local": joint_axis,
            "joint_axis_world": joint_axis_world,
            "opening_axis_world": opening_axis,
            "long_axis_world": self._object_long_axis_world,
            # Zero means the opening/closing direction is perpendicular to
            # the long edge; one means the fingers are at the two ends.
            "long_axis_dot": abs(sum(a * b for a, b in zip(
                opening_axis, self._object_long_axis_world
            ))),
            "down_alignment": -axes[2][2],
        }

    def _print_gripper_alignment(self) -> None:
        if self.gripper_id is None or len(self.gripper_joint_indices) < 2:
            return
        measured = self.get_gripper_alignment()
        print(
            f"RobotArm {self.robot_id} measured finger frame WORLD: "
            f"euler={measured['euler_world']}, "
            f"X={measured['axes_world'][0]}, Y={measured['axes_world'][1]}, "
            f"Z={measured['axes_world'][2]}"
        )
        print(
            f"RobotArm {self.robot_id} joint opening axis local={measured['joint_axis_local']}, "
            f"WORLD={measured['joint_axis_world']}; "
            f"finger separation axis={measured['opening_axis_world']}; "
            f"abs(dot(opening, long))={measured['long_axis_dot']:.6f} (want 0), "
            f"dot(tool_Z, down)={measured['down_alignment']:.6f} (want 1)"
        )
        origin = self.get_measured_tcp_position()
        for debug_id in self._pose_debug_ids:
            p.removeUserDebugItem(debug_id)
        self._pose_debug_ids.clear()
        for axis, color in zip(measured["axes_world"], ((1, 0, 0), (0, 1, 0), (0, 0, 1))):
            endpoint = tuple(origin[i] + 0.10 * axis[i] for i in range(3))
            self._pose_debug_ids.append(p.addUserDebugLine(origin, endpoint, color, lineWidth=3))
        # Yellow: object's known/visual long edge.  Magenta: measured opening.
        for axis, color in (
            (self._object_long_axis_world, (1, 1, 0)),
            (measured["opening_axis_world"], (1, 0, 1)),
        ):
            self._pose_debug_ids.append(p.addUserDebugLine(
                tuple(origin[i] - 0.08 * axis[i] for i in range(3)),
                tuple(origin[i] + 0.08 * axis[i] for i in range(3)), color, lineWidth=2,
            ))

    def _draw_tcp_diagnostic(
        self, target: Sequence[float], force: bool = False,
    ) -> Tuple[float, float, float]:
        """Refresh the measured-TCP line at 20 Hz, independent of motor ticks."""
        actual = self.get_measured_tcp_position()
        if not p.isConnected():
            return actual
        if not force and self._last_tcp_debug_step >= 0 and (
            self._cycle_steps - self._last_tcp_debug_step < self.debug_interval_steps
        ):
            return actual
        self._last_tcp_debug_step = self._cycle_steps
        kwargs = dict(
            lineFromXYZ=target,
            lineToXYZ=actual,
            lineColorRGB=[1.0, 0.35, 0.0],
            lineWidth=3.0,
            # A camera render can outlive a 0.1-second debug item. Keep one
            # persistent ID, explicitly removed when the descent finishes.
            lifeTime=0.0,
        )
        try:
            if self._contact_debug_line_id >= 0:
                kwargs["replaceItemUniqueId"] = self._contact_debug_line_id
            self._contact_debug_line_id = p.addUserDebugLine(**kwargs)
        except TypeError:
            # Older PyBullet builds may not expose replaceItemUniqueId.
            kwargs.pop("replaceItemUniqueId", None)
            self._contact_debug_line_id = p.addUserDebugLine(**kwargs)
        return actual

    def _report_contacts(self) -> bool:
        """Report target proximity; actual collision protection runs every tick."""
        if not p.isConnected() or self._task_object_id is None:
            return False
        object_contact = False
        for source_body, source_name in ((self.robot_id, "arm"), (self.gripper_id, "gripper")):
            if source_body is None:
                continue
            # Avoid scanning every scene body: _unexpected_contacts already
            # monitors external contacts without a per-object query loop.
            for near in p.getClosestPoints(source_body, self._task_object_id, distance=0.01):
                object_contact = True
                key = (source_body, self._task_object_id, int(near[3]), int(near[4]), "near")
                if key in self._contact_reported_keys:
                    continue
                self._contact_reported_keys.add(key)
                print(
                    f"RobotArm {self.robot_id} NEAR_CONTACT: {source_name} "
                    f"link={near[3]} -> body={near[2]} link={near[4]}, "
                    f"distance={near[8]:.5f} m"
                )
        return object_contact

    def _unexpected_contacts(self):
        """Inspect this arm and gripper, with a link-specific grasp whitelist.

        Payload/support contacts are deliberately outside these two sensors:
        a box resting on the belt or its drop zone is not an arm collision.
        Another arm, a platform edge, or an arbitrary new obstacle is never
        ignored based on its body type or on a test-panel registration.
        """
        allowed_finger_states = {
            ArmState.DESCEND, ArmState.GRASP, ArmState.LIFT,
            ArmState.MOVE_TO_DROP_ABOVE, ArmState.DESCEND_TO_DROP,
            ArmState.RELEASE, ArmState.LIFT_AFTER_RELEASE,
        }
        own_bodies = {self.robot_id, self.gripper_id}
        contacts = []
        for body in (self.robot_id, self.gripper_id):
            if body is None:
                continue
            for contact in p.getContactPoints(bodyA=body):
                other, link = int(contact[2]), int(contact[3])
                # The fixed tool mount is internal to this assembly. Its
                # collision pairs are also disabled when attaching the tool.
                if other in own_bodies:
                    continue
                if contact[8] > 0.0 and contact[9] <= 0.0:
                    continue  # cached proximity, not physical contact
                if (body == self.gripper_id and link in self.gripper_joint_indices
                        and other in (self._task_object_id, self.held_object_id)
                        and self.state in allowed_finger_states):
                    continue
                contacts.append(contact)
        return contacts

    def _hold_collision_pose(self):
        """Hold measured joints; never continue driving the interrupted path."""
        for index, angle in zip(self.arm_joint_indices, self._collision_hold_joints):
            p.setJointMotorControl2(
                self.robot_id, index, p.POSITION_CONTROL, targetPosition=angle,
                force=self.normal_joint_force, maxVelocity=0.05,
                positionGain=0.15, velocityGain=1.0,
            )
        if self.gripper_id is not None:
            for index, angle in zip(self.gripper_joint_indices, self._collision_hold_fingers):
                p.setJointMotorControl2(self.gripper_id, index, p.POSITION_CONTROL,
                                        targetPosition=angle, force=10, maxVelocity=0.01)

    def _draw_collision_warning(self):
        if not (self._collision_waiting or self._collision_recovery_blocked) and self._collision_warning_steps <= 0:
            return
        self._collision_warning_steps = max(0, self._collision_warning_steps - 1)
        tcp = self.get_measured_tcp_position()
        status = ("RECOVERY BLOCKED" if self._collision_recovery_blocked else
                  "STOP - WAITING FOR CLEARANCE" if self._collision_waiting else "ABORT")
        kwargs = dict(text=f"{self.display_name}: COLLISION / {status}",
                      textPosition=(tcp[0], tcp[1], tcp[2] + 0.4),
                      textColorRGB=(1, 0, 0), textSize=2.0, lifeTime=0.1)
        if self._collision_warning_id >= 0:
            kwargs["replaceItemUniqueId"] = self._collision_warning_id
        try:
            self._collision_warning_id = p.addUserDebugText(**kwargs)
        except TypeError:
            kwargs.pop("replaceItemUniqueId", None)
            self._collision_warning_id = p.addUserDebugText(**kwargs)

    def _block_collision_recovery(self, reason):
        self._cancel_traffic_yield(reason)
        """A failed retreat is not a completed recovery; require an explicit reset."""
        self.handoff_eligible = False
        self.last_error = f"{self.last_error}; collision recovery blocked: {reason}"
        self._collision_recovery_blocked = True
        self._collision_waiting = False
        self._collision_hold_joints = tuple(p.getJointState(self.robot_id, j)[0]
                                            for j in self.arm_joint_indices)
        self._collision_hold_fingers = tuple(p.getJointState(self.gripper_id, j)[0]
                                             for j in self.gripper_joint_indices) if self.gripper_id is not None else ()
        self._path_waypoints = []
        self._path_index = 0
        self._set_state(ArmState.ABORT)
        self._hold_collision_pose()
        self._draw_collision_warning()
        print(f"\033[91mWARNING {self.display_name}: {self.last_error}; restart/reset required\033[0m",
              flush=True)

    def _check_live_collision(self):
        """Return True if a contact or a clearance wait preempts this tick."""
        if self._collision_recovery_blocked:
            self._hold_collision_pose()
            self._draw_collision_warning()
            return True
        contacts = self._unexpected_contacts()
        if self._collision_waiting:
            self._collision_clear_steps = 0 if contacts else self._collision_clear_steps + 1
            self._hold_collision_pose()
            if self._collision_clear_steps >= max(1, round(0.05 / self._time_step)):
                self._collision_waiting = False
                current = self.get_measured_tcp_position()
                safe_z = max(current[2], self._abort_safe_position[2])
                self._abort_safe_position = (current[0], current[1], safe_z)
                self._begin_cartesian_segment(self._abort_safe_position)
                self._state_steps = 0
                print(f"{self.display_name}: collision contact cleared; retreating", flush=True)
            self._draw_collision_warning()
            return True
        if not contacts:
            self._draw_collision_warning()
            return False
        contact = min(contacts, key=lambda row: row[8])
        self.collision_count += 1
        point = tuple((contact[5][i] + contact[6][i]) / 2 for i in range(3))
        self.last_collision = {
            "sequence": self.collision_count, "state": self.state.name,
            "source_body": int(contact[1]), "source_link": int(contact[3]),
            "other_body": int(contact[2]), "other_link": int(contact[4]),
            "position": point, "distance": float(contact[8]), "force": float(contact[9]),
        }
        reason = (f"collision detected: body={contact[1]} link={contact[3]} -> "
                  f"body={contact[2]} link={contact[4]}")
        print(f"\033[91mWARNING {self.display_name}: {reason}; point={point}\033[0m", flush=True)
        p.addUserDebugPoints([point], [[1, 0, 0]], pointSize=28, lifeTime=1.0)
        self._collision_hold_joints = tuple(p.getJointState(self.robot_id, j)[0]
                                             for j in self.arm_joint_indices)
        self._collision_hold_fingers = tuple(p.getJointState(self.gripper_id, j)[0]
                                              for j in self.gripper_joint_indices) if self.gripper_id is not None else ()
        self._abort_grasp(reason)
        if self._collision_recovery_blocked:
            # Standby collision handling already installed a stationary hold.
            # There is no retreat plane to resume after contact clearance.
            return True
        self._collision_waiting = True
        self._collision_clear_steps = 0
        self._collision_warning_steps = round(3.0 / self._time_step)
        self._hold_collision_pose()
        self._draw_collision_warning()
        return True

    def _highlight_unexpected_collision(self, position, key) -> None:
        """Draw a large red, one-second collision marker once per contact."""
        if key in self._contact_reported_keys or not p.isConnected():
            return
        self._contact_reported_keys.add(key)
        print(
            f"RobotArm {self.robot_id} UNEXPECTED_COLLISION: "
            f"body={key[1]}, arm_link={key[2]}, other_link={key[3]}, "
            f"position={tuple(round(float(value), 4) for value in position)}",
            flush=True,
        )
        p.addUserDebugPoints(
            pointPositions=[tuple(float(value) for value in position)],
            pointColorsRGB=[[1.0, 0.0, 0.0]],
            pointSize=24.0,
            lifeTime=1.0,
        )

    def _visual_state(self):
        """Return display label and RGB color for the current FSM state."""
        if self._collision_recovery_blocked:
            return "ABORT / RECOVERY BLOCKED", (1.0, 0.0, 0.0)
        if self._collision_waiting:
            return "ABORT / COLLISION HOLD", (1.0, 0.0, 0.0)
        if self.traffic_yielding or self._traffic_yield_requested is not None:
            return "YIELDING / SAFE HEIGHT", (1.0, 0.65, 0.0)
        if self._workspace_waiting:
            return "WAITING / CENTRAL TOKEN", (1.0, 0.8, 0.0)
        if self._proximity_waiting:
            return "WAITING / CLEARANCE", (1.0, 0.8, 0.0)
        if self.state == ArmState.RETRY:
            return "RETRYING...", (1.0, 0.8, 0.0)
        if self.state == ArmState.ABORT:
            return "ABORT", (1.0, 0.0, 0.0)
        if self.state == ArmState.IDLE and self.last_error:
            return "FAILED", (1.0, 0.0, 0.0)
        if self.state == ArmState.IDLE:
            return "IDLE", (0.25, 1.0, 0.35)
        return "MOVING", (0.15, 0.45, 1.0)

    def _update_debug_visuals(self) -> None:
        """Update status text, body color and short-lived TCP trail."""
        if not p.isConnected():
            return
        label, text_color = self._visual_state()
        try:
            base_position = p.getBasePositionAndOrientation(self.robot_id)[0]
        except p.error:
            return
        text_position = (
            float(base_position[0]), float(base_position[1]), float(base_position[2] + 0.8)
        )
        text_kwargs = dict(
            text=f"{self.display_name}: {label}",
            textPosition=text_position,
            textColorRGB=list(text_color),
            textSize=1.4,
            lifeTime=0.1,
        )
        try:
            if self._status_debug_id >= 0:
                text_kwargs["replaceItemUniqueId"] = self._status_debug_id
            self._status_debug_id = p.addUserDebugText(**text_kwargs)
        except TypeError:
            text_kwargs.pop("replaceItemUniqueId", None)
            self._status_debug_id = p.addUserDebugText(**text_kwargs)

        # Red is held for one simulated second after a grasp abort, then the
        # body returns to its normal grey/green idle appearance.
        if self._failure_highlight_steps > 0 or self._collision_waiting or self._collision_recovery_blocked:
            body_color = (1.0, 0.0, 0.0, 1.0)
            self._failure_highlight_steps = max(0, self._failure_highlight_steps - 1)
        elif self.state == ArmState.RETRY:
            body_color = (1.0, 0.75, 0.0, 1.0)
        elif self.traffic_yielding or self._workspace_waiting or self._proximity_waiting:
            body_color = (1.0, 0.65, 0.0, 1.0)
        elif self.state == ArmState.IDLE:
            body_color = (0.55, 0.60, 0.58, 1.0)
        else:
            body_color = (0.12, 0.35, 0.95, 1.0)
        if body_color != self._last_visual_color:
            for link_index in [-1, *range(p.getNumJoints(self.robot_id))]:
                try:
                    p.changeVisualShape(self.robot_id, link_index, rgbaColor=body_color)
                except p.error:
                    pass
            self._last_visual_color = body_color

        moving_states = {
            ArmState.MOVE_TO_TARGET_ABOVE, ArmState.DESCEND, ArmState.GRASP,
            ArmState.RETRY, ArmState.ABORT, ArmState.LIFT,
            ArmState.MOVE_TO_DROP_ABOVE, ArmState.DESCEND_TO_DROP,
            ArmState.RELEASE, ArmState.LIFT_AFTER_RELEASE,
            ArmState.RETURN_HOME,
        }
        if self.state not in moving_states:
            self._last_trail_position = None
            return
        current = self.get_measured_tcp_position()
        if self._last_trail_position is not None:
            p.addUserDebugLine(
                lineFromXYZ=self._last_trail_position,
                lineToXYZ=current,
                lineColorRGB=list(self.trail_color),
                lineWidth=2.0,
                lifeTime=0.5,
            )
        self._last_trail_position = current

    def update(self) -> ArmState:
        """Advance the controller by one tick and return its current state."""
        self._update_ticks += 1
        self._state_steps += 1
        self._compensate_arm_gravity()
        # The visual overlay is driven by the same non-blocking tick as the
        # controller, so it remains synchronized with the FSM in GUI runs.
        self._update_debug_visuals()
        if self.traffic_yielding or self._traffic_yield_requested is not None:
            if self._check_live_collision():
                return self.state
            if self.state != ArmState.IDLE:
                self._cycle_steps += 1
                self.stage_sim_seconds[self.state.name] = (
                    self.stage_sim_seconds.get(self.state.name, 0.0) + self._time_step)
            if not self.traffic_yielding:
                self._start_traffic_yield()
            if self.traffic_yielding:
                return self._traffic_yield_update()
            self._state_steps -= 1
            return self.state
        if self.state == ArmState.IDLE:
            # Standby is still physical: a neighbour/obstacle striking this
            # assembly cannot be hidden by the early IDLE return.
            if self._check_live_collision():
                return self.state
            if self.workspace_gate:
                self.workspace_gate.release(self.robot_id)
            return self.state
        # Collision waits consume simulation time too, while their recovery
        # timeout remains paused until contact has cleared.
        self._cycle_steps += 1
        self.stage_sim_seconds[self.state.name] = (
            self.stage_sim_seconds.get(self.state.name, 0.0) + self._time_step
        )
        # Contact data belongs to the caller's just-completed physics step.
        # Check before issuing any new Cartesian or joint-space motion, also
        # during RETRY/ABORT/RETURN_HOME. A persistent contact holds ABORT and
        # cannot fall through the generic timeout into a dispatchable IDLE.
        if self._check_live_collision():
            return self.state
        if self.workspace_gate:
            allowed, conflict = self.workspace_gate.motion_allowed(self)
            if not allowed:
                if not self._proximity_waiting:
                    self._proximity_hold_joints = tuple(p.getJointState(self.robot_id,j)[0]
                                                        for j in self.arm_joint_indices)
                    other, actual, predicted = conflict
                    print(f"[Clearance] t={self._update_ticks*self._time_step:.3f}s {self.display_name} pauses; "
                          f"peer={self.workspace_gate.names.get(other,other)}, "
                          f"actual={actual:.3f}m predicted={predicted:.3f}m",flush=True)
                    self.workspace_gate.wait_events.append((self._update_ticks*self._time_step,
                                                            self.robot_id,other,actual,predicted))
                self._proximity_waiting = True
                self._state_steps -= 1
                self._traffic_wait_tick(conflict)
                return self.state
            if self._proximity_waiting:
                print(f"[Clearance] t={self._update_ticks*self._time_step:.3f}s {self.display_name} resumes",flush=True)
            self._proximity_waiting = False
            self._proximity_hold_joints = None
            self._traffic_wait_position = None
            self._traffic_stalled_steps = 0
        if self._target_pos is None or self._drop_pos is None:
            self._set_state(ArmState.IDLE)
            return self.state

        if self.is_tracking_target and self.state in (ArmState.DESCEND, ArmState.GRASP):
            if (self._cycle_steps - self._tracking_observed_step) * self._time_step > self.max_tracking_age:
                self._abort_task("visual track expired during moving pick")
                return self.state
            self._target_pos = self.predict_tracked_position()

        grasp_target = (
            self._target_pos[0],
            self._target_pos[1],
            self._target_pos[2] + self._task_grasp_offset_z,
        )
        drop_target = (
            self._drop_pos[0], self._drop_pos[1],
            self._drop_pos[2] + self._task_grasp_offset_z,
        )

        if self.state == ArmState.MOVE_TO_TARGET_ABOVE:
            # Phase 0: lift vertically from the *actual* current TCP.  Phase
            # 1: translate in X/Y, still at safe height. Phase 2: turn above
            # the target, away from the fully stretched home's reach limit.
            # Phase 3: fully open, then descend with the fixed task quaternion.
            if self._approach_phase == 0:
                if self.near_pickup_home and self.workspace_gate:
                    if not self.workspace_gate.acquire(self.robot_id):
                        if not self._workspace_waiting:
                            print(f"[Motion] {self.display_name} waiting at pickup HOME; "
                                  f"owner={self.workspace_gate.names.get(self.workspace_gate.owner, self.workspace_gate.owner)}",
                                  flush=True)
                        self._workspace_waiting = True
                        self._state_steps -= 1  # a reservation wait is not a motion timeout
                        self._command_initial_joint_pose()
                        return self.state
                    self._workspace_waiting = False
                if not self._path_waypoints:
                    current = self.get_tcp_position()
                    if self.near_pickup_home:
                        if self._motion_tracking:
                            self._target_pos = self.predict_tracked_position(1.0)
                            grasp_target = (*self._target_pos[:2],
                                            self._target_pos[2] + self._task_grasp_offset_z)
                        start_z = max(self.shared_transit_height if self.station_motion else current[2],
                                      grasp_target[2] + self.approach_height)
                    else:
                        start_z = current[2] + self.vertical_clearance
                    start_up = (current[0], current[1], start_z)
                    self._begin_cartesian_segment(start_up)
                if self._follow_cartesian_segment():
                    self._path_waypoints = []
                    self._path_index = 0
                    self._approach_phase = 1
                    current = self.get_tcp_position()
                    safe_z = max(
                        current[2],
                        grasp_target[2] + self.approach_height,
                    )
                    self._target_safe_position = (
                        grasp_target[0], grasp_target[1], safe_z
                    )
                    if self.shared_cell_motion and not self.near_pickup_home:
                        base_y = p.getBasePositionAndOrientation(self.robot_id)[0][1]
                        outer_y = base_y + math.copysign(.34, base_y)
                        self._approach_points = [(current[0], outer_y, safe_z),
                                                 (.26, outer_y, safe_z),
                                                 (.26, grasp_target[1], safe_z),
                                                 self._target_safe_position]
                        self._begin_cartesian_segment(self._approach_points.pop(0))
                    else:
                        self._begin_cartesian_segment(self._target_safe_position)
            elif self._approach_phase == 1:
                if self._follow_cartesian_segment():
                    self._path_waypoints = []
                    self._path_index = 0
                    if self.shared_cell_motion and len(self._approach_points) == 3:
                        if self.workspace_gate and not self.workspace_gate.acquire(self.robot_id):
                            if not self._workspace_waiting:
                                print(f"[Motion] {self.display_name} waiting outside shared volume; "
                                      f"owner={self.workspace_gate.names.get(self.workspace_gate.owner, self.workspace_gate.owner)}", flush=True)
                            self._workspace_waiting = True
                            # Keep motors, collision checks and RGB-D updates active.
                            return self.state
                        self._workspace_waiting = False
                        if self._motion_tracking:
                            self._target_pos = self.predict_tracked_position(3.0)
                            safe_z = self._target_safe_position[2]
                            self._target_safe_position = (*self._target_pos[:2], safe_z)
                            base_y = p.getBasePositionAndOrientation(self.robot_id)[0][1]
                            outer_y = base_y + math.copysign(.34, base_y)
                            self._approach_points = [(.26, outer_y, safe_z),
                                                     (.26, self._target_pos[1], safe_z),
                                                     self._target_safe_position]
                    if self._approach_points:
                        self._begin_cartesian_segment(self._approach_points.pop(0))
                    else:
                        self._approach_phase = 2
            elif self._approach_phase == 2:
                if self._align_orientation_at_safe_height():
                    self.downward_orientation = self.target_orientation
                    self._approach_phase = 3
                    # Fully open the fingers while still at the safe height.
                    # Descent is not started until both prismatic joints have
                    # reached their open targets.
                    self._configure_gripper_contact()
                    self.open_gripper()
                    self._gripper_opening = True
                    self._gripper_open_wait_steps = 0
            elif self._approach_phase == 3 and self._gripper_opening:
                if self._motion_tracking:
                    moving = self.predict_tracked_position()
                    self._target_pos = moving
                    grasp_target = (moving[0], moving[1], moving[2] + self._task_grasp_offset_z)
                    safe = (moving[0], moving[1], self._target_safe_position[2])
                    current = self.get_tcp_position()
                    distance = math.dist(current, safe)
                    fraction = min(1.0, self.cartesian_speed * self._time_step / max(distance, 1e-12))
                    self._command_pose(tuple(current[i] + fraction * (safe[i] - current[i]) for i in range(3)))
                    horizontal_ready = math.dist(current[:2], moving[:2]) <= self.position_tolerance
                else:
                    self._command_pose(self._target_safe_position)
                    horizontal_ready = True
                self.open_gripper()
                self._gripper_open_wait_steps += 1
                if self._is_gripper_open() and self._at_orientation() and horizontal_ready:
                    self._gripper_opening = False
                    # The third segment starts at the measured TCP after the
                    # horizontal move, never at a stale cached pose.
                    self._prepare_virtual_contact()
                    self._begin_cartesian_segment(grasp_target)
                    self._set_state(ArmState.DESCEND)
                elif self._gripper_open_wait_steps > 480:
                    self._abort_task("gripper did not reach its open position")
            if self._timed_out():
                self._abort_task("could not complete Cartesian approach")
        elif self.state == ArmState.DESCEND:
            self.open_gripper()
            self._draw_tcp_diagnostic(grasp_target)
            self._report_contacts()
            arrived = (self._follow_moving_descent(grasp_target) if self._motion_tracking else
                       self._follow_cartesian_segment(soft_landing_z=grasp_target[2]))
            if arrived:
                self._draw_tcp_diagnostic(grasp_target, force=True)
                self._print_tcp_error(grasp_target)
                self._print_gripper_alignment()
                self._clear_contact_debug()
                self._tcp_reported = True
                self._grasp_hold_position = grasp_target
                # Budget measured finger travel and a short settling period.
                # Virtual mode still verifies real closure before attachment.
                opening = max((
                    abs(p.getJointState(self.gripper_id, j)[0])
                    for j in self.gripper_joint_indices
                ), default=0.0) if self.gripper_id is not None else 0.0
                self._grasp_close_steps = math.ceil(
                    (opening / self._task_gripper_close_speed + self.gripper_settle_seconds)
                    / self._time_step
                )
                self._grasp_timeout_steps = self._grasp_timeout_limit()
                self.close_gripper()
                self._path_waypoints = []
                self._path_index = 0
                self._set_state(ArmState.GRASP)
            elif self._timed_out():
                self._abort_task("could not reach vertical grasp height")
        elif self.state == ArmState.GRASP:
            if self._motion_tracking:
                self._grasp_hold_position = grasp_target
            self._command_pose(self._grasp_hold_position, soft_landing=True)
            self.close_gripper()
            closed = self.gripper_id is None or all(
                abs(p.getJointState(self.gripper_id, j)[0]) <= 0.002
                for j in self.gripper_joint_indices
            )
            present = self._grasp_object_present()
            # With a known simulation payload, a nearby/contacting object is
            # valid even if the fingers stop short of zero. A closed gripper
            # with no nearby payload is the classic "抓空" case.
            closure_ready = closed or present is True
            if self._state_steps >= self._grasp_close_steps and closure_ready:
                if self._task_object_id is not None and present is not True:
                    self._grasp_failure("payload absent at the fingers")
                    return self.state
                try:
                    # The simulated attachment is created only at the END
                    # of the slow close, immediately before lifting.
                    self._create_virtual_grasp_constraint()
                except (p.error, RuntimeError) as exc:
                    self._grasp_failure(f"virtual grasp failed: {exc}")
                    return self.state
                if self._motion_tracking:
                    self.pick_time_estimate = min(6.0, max(2.0, self._cycle_steps * self._time_step))
                    self._motion_tracking = False
                self._set_state(ArmState.LIFT)
            elif self._state_steps >= max(self._grasp_timeout_steps, 1):
                self._grasp_failure("gripper close timed out")
                return self.state
        elif self.state == ArmState.RETRY:
            # RETRY is deliberately a complete second approach: retreat
            # vertically, translate at safe height if the belt moved, wait for
            # a fully open gripper, then descend again through DESCEND/GRASP.
            self.open_gripper()
            if self._retry_phase == 0:
                if self._follow_cartesian_segment():
                    self._path_waypoints = []
                    self._path_index = 0
                    self._retry_phase = 1
            elif self._retry_phase == 1:
                if self._motion_tracking:
                    moving = self.predict_tracked_position()
                    self._target_pos = moving
                retry_safe = (
                    self._target_pos[0], self._target_pos[1], self._retry_safe_height
                )
                if not self._path_waypoints:
                    self._begin_cartesian_segment(retry_safe)
                if self._follow_cartesian_segment():
                    self._path_waypoints = []
                    self._path_index = 0
                    self._retry_phase = 2
                    self._retry_open_steps = 0
            else:
                self._retry_open_steps += 1
                if self._is_gripper_open() and (
                    self._retry_open_steps * self._time_step >= self.gripper_settle_seconds
                ):
                    if self._motion_tracking:
                        moving = self.predict_tracked_position()
                        self._target_pos = moving
                    retry_target = (
                        self._target_pos[0], self._target_pos[1],
                        self._target_pos[2] + self._task_grasp_offset_z,
                    )
                    self._prepare_virtual_contact()
                    self._begin_cartesian_segment(retry_target)
                    self._set_state(ArmState.DESCEND)
            if self._state_steps > min(self.state_timeout_steps, round(8.0 / self._time_step)):
                self._abort_grasp("grasp retry approach timed out")
        elif self.state == ArmState.ABORT:
            # Scheduler already receives last_error here; the arm remains
            # busy until recovery ends. Retrace to the initial lift point
            # before joint-space home, so open fingers do not sweep through
            # the side of a drop platform from a distant failed pick.
            self.open_gripper()
            if self._path_waypoints:
                if self._follow_cartesian_segment():
                    self._path_waypoints = []
                    self._path_index = 0
                    if not self._abort_returning_above_home:
                        self._abort_returning_above_home = True
                        home = self._initial_tcp_position
                        self._begin_cartesian_segment(
                            (home[0], home[1], home[2] + self.vertical_clearance)
                        )
                    else:
                        self._set_state(ArmState.RETURN_HOME)
            else:
                self._set_state(ArmState.RETURN_HOME)
            if self._state_steps > min(self.state_timeout_steps, round(8.0 / self._time_step)):
                if self.last_collision is not None or self.near_pickup_home:
                    self._block_collision_recovery("safe retreat timed out")
                else:
                    self._target_pos = None
                    self._drop_pos = None
                    self._set_state(ArmState.IDLE)
        elif self.state == ArmState.LIFT:
            # Clear BOTH support surfaces vertically before translating.
            # The payload remains attached and the fingers stay closed.
            if not self._path_waypoints:
                current = self.get_tcp_position()
                safe_z = max(
                    (self._target_safe_position or current)[2],
                    grasp_target[2] + self.approach_height,
                    drop_target[2] + self.approach_height,
                    self.shared_transit_height or 0.0,
                )
                self._begin_cartesian_segment((current[0], current[1], safe_z))
            if self._follow_cartesian_segment(soft_lift_z=grasp_target[2]):
                self._path_waypoints = []
                self._path_index = 0
                self._set_state(ArmState.MOVE_TO_DROP_ABOVE)
            elif self._timed_out():
                self._abort_task("could not lift to safe height")
        elif self.state == ArmState.MOVE_TO_DROP_ABOVE:
            if not self._path_waypoints:
                current = self.get_tcp_position()
                self._drop_safe_position = (drop_target[0], drop_target[1], current[2])
                if self.station_motion and self._transport_points is None:
                    # Move along the lane first, then exit on this station's side.
                    # This avoids cutting diagonally through the shoulder axis.
                    self._transport_points = [(drop_target[0], current[1], current[2]),
                                              self._drop_safe_position]
                    print(f"RobotArm {self.robot_id} carrying to own station {self._drop_safe_position}")
                elif self.shared_cell_motion and self._transport_points is None:
                    self._transport_points = [(.26, current[1], current[2]),
                                              (.26, drop_target[1], current[2]),
                                              self._drop_safe_position]
                    print(f"RobotArm {self.robot_id} carrying via corridor to {self._drop_safe_position}")
                if self.shared_cell_motion and self._transport_points:
                    self._begin_cartesian_segment(self._transport_points.pop(0))
                else:
                    self._begin_cartesian_segment(self._drop_safe_position)
            if self._follow_cartesian_segment():
                self._path_waypoints = []
                self._path_index = 0
                if not self._transport_points:
                    self._begin_cartesian_segment(drop_target)
                    self._set_state(ArmState.DESCEND_TO_DROP)
            elif self._timed_out():
                self._abort_task("could not complete Cartesian move to drop")
        elif self.state == ArmState.DESCEND_TO_DROP:
            # Keep the closed fingers and virtual attachment throughout the
            # transfer AND descent. Only a measured arrival permits release.
            if self._follow_cartesian_segment(soft_landing_z=drop_target[2]):
                actual = self.get_measured_tcp_position()
                print(
                    f"RobotArm {self.robot_id} drop TCP: target={drop_target}, "
                    f"actual={actual}, error={math.dist(actual, drop_target):.6f} m"
                )
                self._path_waypoints = []
                self._path_index = 0
                self._release_hold_steps = 0
                self._set_state(ArmState.RELEASE)
            elif self._timed_out():
                self._abort_task("could not descend vertically to drop point")
        elif self.state == ArmState.RELEASE:
            if self._state_steps == 1:
                self._remove_virtual_grasp_constraint(restore_collisions=False)
            self._command_pose(drop_target, soft_landing=True)
            self.open_gripper()
            self._release_hold_steps += 1
            if self._is_gripper_open() and (
                self._release_hold_steps * self._time_step >= self.gripper_settle_seconds
            ):
                self._set_state(ArmState.LIFT_AFTER_RELEASE)
            elif self._timed_out():
                self._abort_task("could not open gripper at release")
        elif self.state == ArmState.LIFT_AFTER_RELEASE:
            if not self._path_waypoints:
                current = self.get_tcp_position()
                safe_z = max(current[2], (self._drop_safe_position or current)[2])
                self._begin_cartesian_segment((current[0], current[1], safe_z))
            if self._follow_cartesian_segment(soft_lift_z=drop_target[2]):
                self._restore_virtual_collision_pairs()
                self._path_waypoints = []
                self._path_index = 0
                if self.workspace_gate and not self.near_pickup_home:
                    self.workspace_gate.release(self.robot_id)
                self._set_state(ArmState.RETURN_HOME)
            elif self._timed_out():
                self._abort_task("could not retreat above released payload")
        elif self.state == ArmState.RETURN_HOME:
            if self.shared_cell_motion and not self._tcp_return_complete:
                if self._return_points is None:
                    current = self.get_tcp_position()
                    home = self._initial_tcp_position
                    height = max(current[2], (self.shared_transit_height or home[2])
                                 if self.near_pickup_home else home[2] + self.vertical_clearance)
                    base_y = p.getBasePositionAndOrientation(self.robot_id)[0][1]
                    if self.station_motion:
                        # Empty return travel uses the overhead HOME plane.
                        # Carrying arms retain the lower transfer plane, so
                        # an empty wrist cannot park across a neighbour's
                        # pickup/loaded exit corridor and form a deadlock.
                        if self.last_error:
                            # A failed pick can be directly underneath the
                            # peer's overhead HOME. Exit horizontally at the
                            # low transfer plane before rising at our own HOME.
                            self._return_points = [(current[0], home[1], height),
                                                   (home[0], home[1], height), home]
                        else:
                            height = max(height, home[2])
                            self._return_points = [(current[0], current[1], height),
                                                   (current[0], home[1], height),
                                                   (home[0], home[1], height), home]
                    elif self.near_pickup_home:
                        # Cross the shoulder's inner dead zone on the belt side.
                        inner_y = math.copysign(.14, base_y)
                        self._return_points = [(.26, current[1], height),
                                               (.26, inner_y, height),
                                               (home[0], inner_y, height),
                                               (home[0], home[1], height), home]
                    else:
                        outer_y = base_y + math.copysign(.34, base_y)
                        self._return_points = [(.26, current[1], height),
                                               (.26, outer_y, height),
                                               (home[0], outer_y, height),
                                               (home[0], home[1], height), home]
                if not self._path_waypoints and self._return_points:
                    self._begin_cartesian_segment(self._return_points.pop(0))
                if self._path_waypoints and self._follow_cartesian_segment():
                    self._path_waypoints = []
                    self._path_index = 0
                if not self._path_waypoints and not self._return_points:
                    self._tcp_return_complete = True
                elif self._timed_out():
                    self._abort_task("could not return through shared-cell corridor")
                return self.state
            self._command_initial_joint_pose()
            if self._at_initial_joint_pose():
                if self.workspace_gate:
                    self.workspace_gate.release(self.robot_id)
                self._target_pos = None
                self._drop_pos = None
                self._clear_path_debug()
                self._set_state(ArmState.IDLE)
                self.last_cycle_sim_seconds = self._cycle_steps * self._time_step
                self.last_cycle_wall_seconds = perf_counter() - self._cycle_started_at
                print(
                    f"RobotArm {self.robot_id} cycle complete: "
                    f"simulation={self.last_cycle_sim_seconds:.3f}s, "
                    f"wall={self.last_cycle_wall_seconds:.3f}s; "
                    f"stages={ {key: round(value, 3) for key, value in self.stage_sim_seconds.items()} }"
                )
            elif self._timed_out():
                self._abort_task("could not return to initial pose")
        return self.state

    def _initial_pose_position(self) -> Tuple[float, float, float]:
        return self._initial_tcp_position

    def _command_initial_joint_pose(self) -> None:
        """Return through the recorded joint pose without an IK branch jump."""
        self.motion_profile = "NORMAL"
        for joint_index, target in zip(self.arm_joint_indices, self._initial_joint_positions):
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=float(target),
                positionGain=self.motor_position_gain,
                force=self.normal_joint_force,
                maxVelocity=self.normal_joint_max_velocity,
            )

    def _at_initial_joint_pose(self, tolerance: float = 0.02) -> bool:
        return all(
            abs(p.getJointState(self.robot_id, index)[0] - target) <= tolerance
            for index, target in zip(self.arm_joint_indices, self._initial_joint_positions)
        )

    def is_idle(self) -> bool:
        return (self.state == ArmState.IDLE and not self.traffic_yielding
                and self._traffic_yield_requested is None)

    def get_last_error(self) -> Optional[str]:
        return self.last_error

    def get_handoff_status(self) -> ArmHandoffStatus:
        return ArmHandoffStatus(
            eligible=bool(self.handoff_eligible),
            has_payload=self.held_object_id is not None,
            collision=self.last_collision is not None,
            recovery_blocked=bool(self._collision_recovery_blocked),
            recovering=self.state in (ArmState.ABORT, ArmState.RETURN_HOME),
        )


# Compatibility for legacy controller regression scripts.
RobotArm = SimRobotArm


def _as_position(position: Iterable[float]) -> Tuple[float, float, float]:
    values = tuple(float(value) for value in position)
    if len(values) != 3:
        raise ValueError("Position must contain exactly three values")
    return values


def _run_collision_guard_test() -> None:
    """Test the production guard with actual Bullet contacts, without a panel.

    Moving-wall detection runs physics normally. The small contact probes and
    relocated second arm use performCollisionDetection before the next control
    tick to test precise link filtering without applying a destructive impulse.
    Fixture poses are known inputs; no payload pose is queried.
    """
    from env_setup import SimulationEnvironment

    target, drop = (0.20, 0.0, 0.52), (0.15, -0.55, 0.80)

    def fixture(tiny_target=False):
        env = SimulationEnvironment(gui=False)
        attachment = env.get_arm_gripper(0)
        arm = RobotArm(
            env.arm_ids[0], gripper_id=attachment.body_id,
            gripper_joint_indices=attachment.joint_indices,
            tcp_offset_local=attachment.tcp_offset_local,
            display_name="Collision test arm",
        )
        arm.shared_cell_motion = arm.near_pickup_home = arm.station_motion = True
        arm.shared_transit_height = env.layout.transfer_heights[0]
        arm.open_gripper()
        if tiny_target:
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=(0.006,) * 3)
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                     basePosition=target)
        else:
            body = env.create_test_object(
                (target[0], target[1], env.conveyor_top_z + 0.05),
                shape="box", orientation=(0, 0, 0, 1),
            )
        for _ in range(120):
            env.step()
            arm.update()
        arm.assign_task(target, drop, object_id=body, grasp_offset_z=-0.015,
                        object_point_local=(0, 0, 0.05))
        return env, arm, body

    def reach_descent(env, arm):
        for _ in range(2500):
            env.step()
            arm.update()
            assert arm.last_error is None, arm.last_error
            if arm.state == ArmState.DESCEND:
                return
        raise AssertionError("fixture did not reach DESCEND")

    env, arm, _ = fixture()
    try:
        obstacle = None
        for _ in range(2500):
            env.step()
            contacts = [row for part in (arm.robot_id, arm.gripper_id)
                        for row in p.getContactPoints(bodyA=part)]
            arm.update()
            if arm.last_collision is not None:
                assert any(row[2] == obstacle for row in contacts), "no physical impact"
                assert arm.state == ArmState.ABORT, "collision was not intercepted on its tick"
                assert arm.last_collision["state"] == "MOVE_TO_TARGET_ABOVE"
                assert arm.last_collision["other_body"] == obstacle
                break
            if (obstacle is None and arm.state == ArmState.MOVE_TO_TARGET_ABOVE
                    and arm._approach_phase == 1):
                start, end = arm.current_motion_segment()
                delta = tuple(end[i] - start[i] for i in range(3))
                midpoint = tuple((start[i] + end[i]) / 2 for i in range(3))
                orientation = p.getQuaternionFromEuler((
                    0, -math.atan2(delta[2], math.hypot(delta[0], delta[1])),
                    math.atan2(delta[1], delta[0]),
                ))
                shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=(0.04, 0.16, 0.16))
                obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                             basePosition=midpoint, baseOrientation=orientation)
                assert not any(p.getClosestPoints(part, obstacle, distance=0.005)
                               for part in (arm.robot_id, arm.gripper_id)), (
                    "wall must start clear and be hit by real subsequent motion"
                )
        else:
            raise AssertionError("moving arm never detected the dynamic wall")

        try:
            arm.assign_task(target, drop)
        except RuntimeError:
            pass
        else:
            raise AssertionError("collision hold accepted a new task")
        # Exceed the existing eight-second ABORT deadline: persistent contact
        # must keep the arm unavailable rather than timing out into IDLE.
        for _ in range(round(8.5 / env.time_step)):
            env.step()
            arm.update()
            assert arm.state == ArmState.ABORT and arm._collision_waiting
        assert arm.collision_count == 1, "same persistent impact was reported repeatedly"
        p.removeBody(obstacle)
        recovery_states = set()
        for _ in range(2500):
            env.step()
            recovery_states.add(arm.update())
            if arm.is_idle():
                break
        assert arm.is_idle() and ArmState.RETURN_HOME in recovery_states
        assert not arm._collision_waiting and "collision detected" in arm.last_error
        assert arm._at_initial_joint_pose(), "recovery did not reach the home joint tolerance"
        # HOME intentionally accepts 0.02 rad joint error; allow the already
        # commanded motors to settle before asserting a stricter TCP distance.
        for _ in range(60):
            env.step()
            arm.update()
        assert math.dist(arm.get_tcp_position(), arm._initial_tcp_position) < 0.01
        print("PASS collision: real moving-wall impact, same-tick ABORT, persistent hold, clear then HOME")
    finally:
        env.close()

    env, arm, _ = fixture()
    try:
        reach_descent(env, arm)
        finger = p.getLinkState(arm.gripper_id, arm.gripper_joint_indices[0],
                                computeForwardKinematics=True)
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=(0.006,) * 3)
        obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                     basePosition=finger[4], baseOrientation=finger[5])
        p.performCollisionDetection()
        assert p.getContactPoints(bodyA=arm.gripper_id, bodyB=obstacle)
        arm.update()
        assert arm.state == ArmState.ABORT
        assert arm.last_collision["state"] == "DESCEND"
        assert arm.last_collision["other_body"] == obstacle
        print("PASS collision: DESCEND finger contact with a non-target object is rejected")
    finally:
        env.close()

    env, arm, body = fixture(tiny_target=True)
    try:
        reach_descent(env, arm)
        # Re-enable physical finger contacts for this geometry-only test.
        # The virtual demo otherwise disables them until after release.
        arm._restore_virtual_collision_pairs()
        finger = p.getLinkState(arm.gripper_id, arm.gripper_joint_indices[0],
                                computeForwardKinematics=True)
        p.resetBasePositionAndOrientation(body, finger[4], finger[5])
        p.performCollisionDetection()
        contacts = p.getContactPoints(bodyA=arm.gripper_id, bodyB=body)
        assert contacts and all(row[3] in arm.gripper_joint_indices for row in contacts)
        arm.update()
        assert arm.state == ArmState.DESCEND and arm.last_collision is None, (
            "normal finger-target contact was treated as an obstacle"
        )
        palm = p.getLinkState(arm.robot_id, arm.end_effector_link_index,
                              computeForwardKinematics=True)
        p.resetBasePositionAndOrientation(body, palm[4], palm[5])
        p.performCollisionDetection()
        assert any(row[3] == -1 for row in p.getContactPoints(bodyA=arm.gripper_id, bodyB=body))
        arm.update()
        assert arm.state == ArmState.ABORT and arm.last_collision["other_body"] == body
        assert arm.last_collision["source_body"] == arm.gripper_id
        assert arm.last_collision["source_link"] == -1, "palm impact was whitelisted"
        print("PASS collision: actual target-finger contact allowed; target-palm contact rejected")
    finally:
        env.close()

    env, arm, _ = fixture()
    try:
        reach_descent(env, arm)
        # Materialize the second UR5 at this robot's measured wrist solely to
        # create a deterministic inter-arm geometry contact before an impulse.
        wrist = p.getLinkState(arm.robot_id, arm.end_effector_link_index,
                               computeForwardKinematics=True)
        other_arm = env.arm_ids[1]
        p.resetBasePositionAndOrientation(other_arm, wrist[4],
                                          p.getQuaternionFromEuler((0, 0, math.pi)))
        p.performCollisionDetection()
        assert p.getContactPoints(bodyA=arm.robot_id, bodyB=other_arm)
        arm.update()
        assert arm.state == ArmState.ABORT and arm.last_collision["state"] == "DESCEND"
        assert arm.last_collision["other_body"] == other_arm
        print("PASS collision: another UR5 is detected without obstacle registration or a fault panel")
    finally:
        env.close()

    for stalled_stage in (ArmState.ABORT, ArmState.RETURN_HOME):
        env, arm, _ = fixture()
        try:
            reach_descent(env, arm)
            finger = p.getLinkState(arm.gripper_id, arm.gripper_joint_indices[0],
                                    computeForwardKinematics=True)
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=(0.006,) * 3)
            obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                         basePosition=finger[4], baseOrientation=finger[5])
            p.performCollisionDetection()
            arm.update()
            assert arm.state == ArmState.ABORT and arm.last_collision is not None
            p.removeBody(obstacle)
            # Wait for actual contact clearance, then advance only the recovery
            # deadline. This exercises the normal timeout branch without making
            # the safety test wait for an unreachable motor command for seconds.
            for _ in range(60):
                env.step()
                arm.update()
                if not arm._collision_waiting:
                    break
            assert not arm._collision_waiting and arm.state == ArmState.ABORT
            if stalled_stage == ArmState.ABORT:
                current = arm.get_measured_tcp_position()
                arm._begin_cartesian_segment((current[0], current[1], current[2] + 0.25))
                arm._state_steps = min(arm.state_timeout_steps, round(8.0 / env.time_step))
            else:
                assert not arm._at_initial_joint_pose(), "fixture is already at home"
                arm._set_state(ArmState.RETURN_HOME)
                arm._state_steps = arm.state_timeout_steps
            arm.update()
            assert arm._collision_recovery_blocked, f"{stalled_stage.name} timeout was not latched"
            assert arm.state == ArmState.ABORT and not arm.is_idle()
            assert "recovery" in arm.last_error.lower()
            held_joints = tuple(arm._collision_hold_joints)
            for _ in range(120):
                env.step()
                arm.update()
                assert arm.state == ArmState.ABORT and arm._collision_recovery_blocked
            current_joints = [p.getJointState(arm.robot_id, j)[0] for j in arm.arm_joint_indices]
            assert max(abs(a - b) for a, b in zip(current_joints, held_joints)) < 0.02, (
                "recovery timeout continued moving instead of holding the measured joints"
            )
            try:
                arm.assign_task(target, drop)
            except RuntimeError:
                pass
            else:
                raise AssertionError("blocked recovery accepted a new task")
            print(f"PASS collision: {stalled_stage.name} deadline latches recovery-blocked ABORT")
        finally:
            env.close()
    print("PASS: production collision guard, precise whitelist and bounded recovery")


def _run_grasp_failure_test() -> None:
    """Exercise GRASP -> RETRY -> ABORT without using an object pose oracle."""
    from env_setup import SimulationEnvironment

    env = SimulationEnvironment(gui=False)
    try:
        attachment = env.get_arm_gripper(0)
        arm = RobotArm(
            env.arm_ids[0],
            gripper_id=attachment.body_id,
            gripper_joint_indices=attachment.joint_indices,
            tcp_offset_local=attachment.tcp_offset_local,
            virtual_grasp=True,
        )
        # The controller is sent to the normal left-arm fixture point while
        # the valid object handle is deliberately placed 45 cm away. This
        # tests the visual/scheduler supplied handle without reading its pose.
        target = (-0.55, -0.30, env.conveyor_top_z + 0.10)
        object_id = env.create_test_object(
            (target[0] + 0.45, target[1], env.conveyor_top_z + 0.05),
            shape="box",
        )
        arm.assign_task(
            target, (-0.55, -0.65, 0.52), object_id=object_id,
            object_point_local=(0.0, 0.0, 0.05),
            object_long_axis_local=(1.0, 0.0, 0.0),
        )
        states = []
        for _ in range(8000):
            env.step()
            state = arm.update()
            if not states or states[-1] != state:
                states.append(state)
            if arm.is_idle():
                break
        assert arm.is_idle(), "failed grasp left the arm busy"
        assert arm.last_error, "ABORT did not report a failure"
        assert arm._grasp_retry_count == 1, "failure was not retried exactly once"
        assert ArmState.RETRY in states and ArmState.ABORT in states
        assert arm._virtual_grasp_constraint_id < 0
        print(
            "PASS: failed grasp retried once, entered ABORT, reported "
            f"{arm.last_error!r}, and returned IDLE"
        )
    finally:
        env.close()


if __name__ == "__main__":
    import argparse
    import numpy as np
    from env_setup import SimulationEnvironment

    parser = argparse.ArgumentParser(description="Two-arm sorting into blue/green drop zones")
    parser.add_argument("--headless", action="store_true", help="run a finite DIRECT test")
    parser.add_argument("--physical-grasp", action="store_true", help="disable virtual grasp")
    parser.add_argument(
        "--test-grasp-failure", action="store_true",
        help="run the GRASP timeout/empty-payload RETRY -> ABORT regression",
    )
    parser.add_argument(
        "--test-collision", action="store_true",
        help="test real contacts, emergency hold, recovery and the target-finger whitelist",
    )
    parser.add_argument("--steps", type=int, default=20000, help="headless step limit")
    parser.add_argument(
        "--target-euler-left", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"),
        help="left gripper WORLD Euler radians; default aligns with the box",
    )
    parser.add_argument(
        "--target-euler-right", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"),
        help="right gripper WORLD Euler radians; default aligns with the box",
    )
    parser.add_argument(
        "--object-yaw", type=float, default=0.0,
        help="known fixture box WORLD yaw in radians; default long edge is X",
    )
    args = parser.parse_args()
    if args.test_grasp_failure:
        _run_grasp_failure_test()
        raise SystemExit(0)
    if args.test_collision:
        _run_collision_guard_test()
        raise SystemExit(0)
    env = SimulationEnvironment(gui=not args.headless)
    arms, tasks, object_ids = [], [], []
    # Keep the ENTIRE box inside the belt, including a 90-degree test turn.
    # These are explicit fixture points for controller testing, not a visual
    # detection shortcut.  vision.py obtains its target from RGB/depth.
    local_offsets = ((0.40, 0.65, 0.0), (0.35, 0.65, 0.0))
    object_orientation = p.getQuaternionFromEuler((0.0, 0.0, args.object_yaw))
    target_eulers = (args.target_euler_left, args.target_euler_right)

    def assign_test_task(index):
        target, drop = tasks[index]
        arms[index].assign_task(
            target, drop, grasp_offset_z=-0.015, object_id=object_ids[index],
            object_orientation=object_orientation,
            object_point_local=(0.0, 0.0, 0.05),
            object_long_axis_local=(1.0, 0.0, 0.0),
        )

    def assign_failure_task(index):
        """Send a reachable fake point while retaining a real payload handle."""
        target, drop = tasks[index]
        # Move away from the fixture along X. The real object remains at its
        # known test location/drop area, so the gripper closes in empty space.
        offset_x = 0.35 if index == 0 else -0.35
        fake_target = (target[0] + offset_x, target[1], target[2])
        arms[index].assign_task(
            fake_target, drop, grasp_offset_z=-0.015, object_id=object_ids[index],
            object_orientation=object_orientation,
            object_point_local=(0.0, 0.0, 0.05),
            object_long_axis_local=(1.0, 0.0, 0.0),
        )
        print(
            f"Arm {index + 1}: Force Grab Failure -> fake target={fake_target}",
            flush=True,
        )

    try:
        for index, robot_id in enumerate(env.arm_ids):
            attachment = env.get_arm_gripper(index)
            arm = RobotArm(
                robot_id, gripper_id=attachment.body_id,
                gripper_joint_indices=attachment.joint_indices,
                tcp_offset_local=attachment.tcp_offset_local,
                virtual_grasp=not args.physical_grasp,
                target_euler=target_eulers[index],
                display_name=f"Arm {index + 1}",
                trail_color=(1.0, 0.0, 0.0) if index == 0 else (0.0, 0.35, 1.0),
            )
            # Robot base calibration is allowed; no object pose is read.
            base_position, base_orientation = p.getBasePositionAndOrientation(robot_id)
            target_xy = p.multiplyTransforms(
                base_position, base_orientation, local_offsets[index],
                (0.0, 0.0, 0.0, 1.0),
            )[0]
            zone = env.get_drop_zone(index)
            target = (target_xy[0], target_xy[1], env.conveyor_top_z + 0.10)
            # Both task points refer to the 10 cm box's top surface. The
            # controller applies the grasp Z offset consistently at release.
            drop = (*zone.surface_position[:2], zone.surface_position[2] + 0.10)
            assert math.dist(target[:2], drop[:2]) >= 0.25
            object_ids.append(env.create_test_object(
                (target[0], target[1], env.conveyor_top_z + 0.05),
                shape="box", orientation=object_orientation,
            ))
            arms.append(arm)
            tasks.append((target, drop))
            arm.open_gripper()
            assign_test_task(index)
            print(
                f"Arm {index + 1}: robot={robot_id}, gripper={attachment.body_id}, "
                f"base={base_position}, ee={arm.end_effector_link_index}, "
                f"target={target}, drop={drop}, zone={zone.name}, "
                f"virtual_grasp={arm.virtual_grasp}"
            )
            p.addUserDebugPoints([target], [[1.0, 0.0, 0.0]], pointSize=10.0)
            p.addUserDebugPoints([drop], [list(zone.color[:3])], pointSize=10.0)

        replay_controls = [] if args.headless else [
            p.addUserDebugParameter(f"Replay arm {index + 1} (move to 1)", 0, 1, 0)
            for index in range(len(arms))
        ]
        last_controls = [p.readUserDebugParameter(item) for item in replay_controls]
        failure_control = None if args.headless else p.addUserDebugParameter(
            "Force Grab Failure", 0, 1, 0
        )
        last_failure_control = (
            p.readUserDebugParameter(failure_control) if failure_control is not None else 0.0
        )
        previous_states = [None] * len(arms)
        visited_states = [set() for _ in arms]
        state_sequences = [[] for _ in arms]
        descent_profiles = [set() for _ in arms]
        lift_profiles = [set() for _ in arms]
        maximum_line_errors = [0.0 for _ in arms]
        # RETRY/ABORT are failure-only branches and are intentionally absent
        # from the successful pick/place sequence below.
        success_states = {
            ArmState.IDLE, ArmState.MOVE_TO_TARGET_ABOVE, ArmState.DESCEND,
            ArmState.GRASP, ArmState.LIFT, ArmState.MOVE_TO_DROP_ABOVE,
            ArmState.DESCEND_TO_DROP, ArmState.RELEASE,
            ArmState.LIFT_AFTER_RELEASE, ArmState.RETURN_HOME,
        }
        expected_sequence = [
            state for state in ArmState if state in success_states and state != ArmState.IDLE
        ] + [ArmState.IDLE]
        completed = False
        for step in range(args.steps if args.headless else 1000000):
            if not p.isConnected():
                break
            env.step()
            if failure_control is not None:
                failure_value = p.readUserDebugParameter(failure_control)
                if failure_value > 0.5 and last_failure_control <= 0.5:
                    idle_index = next(
                        (candidate for candidate, item in enumerate(arms) if item.is_idle()),
                        None,
                    )
                    if idle_index is None:
                        print("Force Grab Failure: both arms are busy", flush=True)
                    else:
                        assign_failure_task(idle_index)
                last_failure_control = failure_value
            for index, arm in enumerate(arms):
                if replay_controls:
                    value = p.readUserDebugParameter(replay_controls[index])
                    if value != last_controls[index] and value > 0.5:
                        if arm.is_idle():
                            # Respawn a fixture using known setup parameters;
                            # the controller never teleports a moving arm.
                            p.removeBody(object_ids[index])
                            target = tasks[index][0]
                            object_ids[index] = env.create_test_object(
                                (target[0], target[1], env.conveyor_top_z + 0.05),
                                shape="box", orientation=object_orientation,
                            )
                            assign_test_task(index)
                        else:
                            print(f"Arm {index + 1}: finish current task before replay")
                    last_controls[index] = value
                before_state = arm.state
                before_tcp = arm.get_tcp_position() if args.headless else None
                if args.headless and arm._path_waypoints and before_state in (
                    ArmState.DESCEND, ArmState.LIFT, ArmState.MOVE_TO_DROP_ABOVE,
                    ArmState.DESCEND_TO_DROP, ArmState.LIFT_AFTER_RELEASE,
                ):
                    # Verify the PHYSICAL TCP follows the line, not just that
                    # generated command points happen to be collinear.
                    start = np.asarray(arm._segment_start)
                    vector = np.asarray(arm._path_waypoints[-1]) - start
                    delta = np.asarray(arm.get_measured_tcp_position()) - start
                    projection = np.dot(delta, vector) / max(np.dot(vector, vector), 1e-12)
                    line_error = float(np.linalg.norm(delta - vector * projection))
                    maximum_line_errors[index] = max(maximum_line_errors[index], line_error)
                    assert line_error < 0.005, "TCP departed the straight path by more than 5 mm"
                state = arm.update()
                visited_states[index].add(state)
                if args.headless:
                    if before_state in (ArmState.DESCEND, ArmState.DESCEND_TO_DROP):
                        destination = tasks[index][0 if before_state == ArmState.DESCEND else 1]
                        distance_z = before_tcp[2] - (destination[2] - 0.015)
                        expected_profile = "SOFT_LANDING" if distance_z <= 0.03 else "NORMAL"
                        assert arm.motion_profile == expected_profile, "descent speed switched too early/late"
                        descent_profiles[index].add((before_state, arm.motion_profile))
                    elif before_state in (ArmState.LIFT, ArmState.LIFT_AFTER_RELEASE):
                        destination = tasks[index][0 if before_state == ArmState.LIFT else 1]
                        distance_z = before_tcp[2] - (destination[2] - 0.015)
                        expected_profile = "SOFT_LANDING" if distance_z <= 0.03 else "NORMAL"
                        assert arm.motion_profile == expected_profile, "lift speed switched too early/late"
                        lift_profiles[index].add((before_state, arm.motion_profile))
                    elif before_state in (ArmState.MOVE_TO_DROP_ABOVE, ArmState.RETURN_HOME):
                        assert arm.motion_profile == "NORMAL", "slow contact profile leaked into free space"
                if state != previous_states[index]:
                    print(f"step={step} Arm {index + 1}: {state.name}", flush=True)
                    previous_states[index] = state
                    state_sequences[index].append(state)
                    if state == ArmState.GRASP:
                        target = tasks[index][0]
                        actual = arm.get_measured_tcp_position()
                        p.addUserDebugPoints([actual], [[0.0, 1.0, 0.0]], pointSize=10.0)
                        p.addUserDebugLine(target, actual, [1.0, 1.0, 0.0], lineWidth=2.0)
                        if args.headless:
                            grasp = (target[0], target[1], target[2] - 0.015)
                            assert math.dist(actual, grasp) < 0.01
                            assert arm._virtual_grasp_constraint_id == -1
                            alignment = arm.get_gripper_alignment()
                            assert alignment["long_axis_dot"] < math.sin(math.radians(2.0)), (
                                "opening axis must be perpendicular to the box long edge; "
                                "adjust WORLD target_euler yaw"
                            )
                            assert alignment["down_alignment"] > math.cos(math.radians(2.0))
                    elif args.headless and state in (
                        ArmState.LIFT, ArmState.MOVE_TO_DROP_ABOVE,
                        ArmState.DESCEND_TO_DROP, ArmState.RELEASE,
                    ) and arm.virtual_grasp:
                        assert arm._virtual_grasp_constraint_id >= 0
                        if state == ArmState.RELEASE:
                            drop = tasks[index][1]
                            release = (drop[0], drop[1], drop[2] - 0.015)
                            actual = arm.get_measured_tcp_position()
                            assert math.dist(actual, release) < 0.01
                            assert math.dist(actual[:2], tasks[index][0][:2]) > 0.25, (
                                "payload was released at the pick location"
                            )
                    elif args.headless and state == ArmState.LIFT_AFTER_RELEASE:
                        assert arm._virtual_grasp_constraint_id == -1
                        drop = tasks[index][1]
                        release = (drop[0], drop[1], drop[2] - 0.015)
                        assert math.dist(arm.get_measured_tcp_position(), release) < 0.01
            if args.headless and all(arm.is_idle() for arm in arms):
                completed = True
                break

        if args.headless:
            assert completed, "arms did not complete within the step limit"
            for index, (arm, visited) in enumerate(zip(arms, visited_states)):
                assert arm.last_error is None, arm.last_error
                assert success_states <= visited, "a pick/place stage was skipped"
                assert state_sequences[index] == expected_sequence, "pick/place stages ran out of order"
                assert descent_profiles[index] == {
                    (stage, profile)
                    for stage in (ArmState.DESCEND, ArmState.DESCEND_TO_DROP)
                    for profile in ("NORMAL", "SOFT_LANDING")
                }, "both descents must include normal and final-3-cm profiles"
                assert lift_profiles[index] == {
                    (stage, profile)
                    for stage in (ArmState.LIFT, ArmState.LIFT_AFTER_RELEASE)
                    for profile in ("NORMAL", "SOFT_LANDING")
                }, "both lifts must include initial-3-cm and normal profiles"
                assert arm._virtual_grasp_constraint_id == -1, "grasp constraint leaked"
                assert not arm._virtual_collision_pairs, "collision filters leaked"
                if arm.virtual_grasp:
                    assert arm.last_cycle_sim_seconds <= 10.0, "demo cycle exceeded 10 simulation seconds"
                print(f"Arm {index + 1} maximum measured line deviation: {maximum_line_errors[index]:.6f} m")
            assert p.getNumConstraints() == len(env.gripper_attachments)
            # Verify landing from the CAMERA depth/segmentation buffers.  This
            # observes the payload independently without reading its body pose.
            _, depth_buffer, segmentation = env.camera._capture_buffers()
            config = env.camera.config
            view = p.computeViewMatrixFromYawPitchRoll(
                config.target, config.distance, config.yaw_deg,
                config.pitch_deg, config.roll_deg, upAxisIndex=2,
            )
            projection = p.computeProjectionMatrixFOV(
                config.fov_deg, config.width / config.height, config.near, config.far
            )
            inverse = np.linalg.inv(
                np.asarray(projection).reshape(4, 4, order="F")
                @ np.asarray(view).reshape(4, 4, order="F")
            )
            segmentation = np.asarray(segmentation).reshape(config.height, config.width)
            depth_buffer = np.asarray(depth_buffer).reshape(config.height, config.width)
            for index, object_id in enumerate(object_ids):
                rows, columns = np.where((segmentation & ((1 << 24) - 1)) == object_id)
                assert len(rows) > 10, f"object {object_id} disappeared from the camera"
                clip = np.stack((
                    2.0 * columns / config.width - 1.0,
                    1.0 - 2.0 * rows / config.height,
                    2.0 * depth_buffer[rows, columns] - 1.0,
                    np.ones(len(rows)),
                ))
                world = inverse @ clip
                world = (world[:3] / world[3]).T
                xy = np.median(world[:, :2], axis=0)
                top_z = float(np.quantile(world[:, 2], 0.95))
                drop = tasks[index][1]
                zone = env.get_drop_zone(index)
                error_xy = float(np.linalg.norm(xy - np.asarray(drop[:2])))
                print(
                    f"Arm {index + 1} camera landing in {zone.name}: XY error={error_xy:.6f} m, "
                    f"surface Z={top_z:.6f} m"
                )
                assert error_xy < 0.03, "payload missed the requested drop area"
                assert abs(top_z - drop[2]) < 0.02, "payload did not settle on the drop platform"
                assert p.getContactPoints(object_id, zone.body_id), "payload is not supported by its drop zone"
            print(
                f"PASS: both arms sorted into blue/green zones and returned in {step + 1} steps "
                f"({(step + 1) / 240.0:.2f} simulation seconds)"
            )
    except p.error:
        if p.isConnected():
            raise
    finally:
        env.close()
