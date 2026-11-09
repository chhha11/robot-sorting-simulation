"""PyBullet scene setup for the multi-arm vision sorting demo.

This module only owns simulation assets and the camera acquisition interface.
Object poses must be obtained by downstream vision code from camera images; this
file deliberately does not expose ground-truth object coordinates.
"""

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

try:  # Keep import errors readable when the environment is not installed yet.
    import numpy as np
    import pybullet as p
    import pybullet_data
except ImportError:  # pragma: no cover - exercised by the standalone test block
    np = None
    p = None
    pybullet_data = None


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    fov_deg: float = 60.0
    near: float = 0.05
    far: float = 10.0
    # Legacy field; actual pose is defined by target/distance/yaw/pitch/roll.
    position: Tuple[float, float, float] = (0.0, -0.15, 4.5)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.42)
    distance: float = 3.6
    # Oblique RGB-D view sees underneath the tools hovering over the belt.
    yaw_deg: float = 45.0
    pitch_deg: float = -65.0
    roll_deg: float = 0.0
    renderer: str = "tiny"


@dataclass(frozen=True)
class DropZoneConfig:
    """One arm's calibrated destination, independent of task allocation."""

    name: str
    arm_name: str
    label: str
    centre_xy: Tuple[float, float]


@dataclass(frozen=True)
class SceneLayout:
    """Known scene geometry, shared by generation and future route-A policies."""

    # Opposite sides, staggered along the travel direction; Arm 2 is mirrored.
    arm_bases: Tuple[Tuple[float, float, float], ...] = (
        (-0.20, -0.46, 0.85), (0.20, 0.46, 0.85))
    arm_base_yaws: Tuple[float, float] = (0.0, math.pi)
    # Both HOME projections are inside this shared pickup rectangle.
    overlap_bounds: Tuple[float, float, float, float] = (-0.40, 0.40, -0.07, 0.07)
    belt_center_xy: Tuple[float, float] = (0.0, 0.0)
    belt_length: float = 4.50
    belt_width: float = 0.52  # fits the original UR5 reach and fixed transfer corridor
    belt_top_z: float = 0.42
    drop_top_z: float = 0.70
    spawn_x: float = -0.40
    spawn_y_range: Tuple[float, float] = (-0.07, 0.07)
    drop_zone_specs: Tuple[DropZoneConfig, ...] = (
        DropZoneConfig("Arm1_Drop_Blue", "arm1", "Box_B", (0.15, -0.55)),
        DropZoneConfig("Arm1_Drop_Red", "arm1", "Box_A", (-0.55, -0.55)),
        DropZoneConfig("Arm2_Drop_Blue", "arm2", "Box_B", (-0.15, 0.55)),
        DropZoneConfig("Arm2_Drop_Red", "arm2", "Box_A", (0.55, 0.55)),
    )
    drop_half_size_xy: Tuple[float, float] = (0.14, 0.12)
    # Hover over opposite ends of the shared lane. Lower transfer planes pass
    # below the waiting tool; the controller checks actual geometric clearance.
    home_tcp_positions: Tuple[Tuple[float, float, float], ...] = (
        (-0.37, -0.03, 1.25), (0.37, 0.03, 1.25))
    transfer_heights: Tuple[float, float] = (0.95, 0.95)

    @property
    def belt_bounds(self) -> Tuple[float, float, float, float]:
        x, y = self.belt_center_xy
        return (x - self.belt_length / 2, x + self.belt_length / 2,
                y - self.belt_width / 2, y + self.belt_width / 2)

    @property
    def drop_centres(self) -> Tuple[Tuple[float, float], ...]:
        return tuple(spec.centre_xy for spec in self.drop_zone_specs)


@dataclass(frozen=True)
class CameraFrame:
    """Synchronized image buffers and their exact OpenGL calibration."""

    rgb: "np.ndarray"
    depth: "np.ndarray"  # metric camera Z, metres
    segmentation: "np.ndarray"  # simulation-only attachment/debug handle
    view_matrix: Tuple[float, ...]
    projection_matrix: Tuple[float, ...]
    pixel_offset: Tuple[float, float]


@dataclass(frozen=True)
class GripperAttachment:
    """Identifiers and TCP calibration for one attached two-finger gripper."""

    body_id: int
    constraint_id: int
    joint_indices: Tuple[int, int]
    tcp_offset_local: Tuple[float, float, float]


@dataclass(frozen=True)
class DropZone:
    """Known placement-surface calibration, independent of object detection."""

    name: str
    body_id: int
    surface_position: Tuple[float, float, float]
    half_size_xy: Tuple[float, float]
    color: Tuple[float, float, float, float]


class GlobalCamera:
    """Camera acquisition abstraction used by the vision module."""

    def __init__(self, config: CameraConfig, physics_client_id: int = 0):
        _require_pybullet()
        self.config = config
        self.physics_client_id = physics_client_id
        if config.renderer not in ("tiny", "opengl"):
            raise ValueError("camera renderer must be 'tiny' or 'opengl'")
        if config.width <= 0 or config.height <= 0 or not 0 < config.near < config.far:
            raise ValueError("invalid camera resolution or near/far planes")
        self.active_renderer = config.renderer
        self.view_matrix = tuple(p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=config.target, distance=config.distance,
            yaw=config.yaw_deg, pitch=config.pitch_deg, roll=config.roll_deg,
            upAxisIndex=2,
        ))
        self.projection_matrix = tuple(p.computeProjectionMatrixFOV(
            fov=config.fov_deg, aspect=config.width / config.height,
            nearVal=config.near, farVal=config.far,
        ))

    @property
    def pixel_offset(self) -> Tuple[float, float]:
        # TinyRenderer rasterizes at integer X and flips Y with height-1;
        # OpenGL samples pixel centres. Use matching viewport coordinates
        # when converting image array indices back to normalized device space.
        return (0.0, 1.0) if self.active_renderer == "tiny" else (0.5, 0.5)

    def capture_frame(self) -> CameraFrame:
        """Capture after the caller's physics step, using one render for RGB/depth."""
        rgba, depth_buffer, segmentation = self._capture_buffers()
        height, width = self.config.height, self.config.width
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()
        buffer = np.asarray(depth_buffer, dtype=np.float32).reshape(height, width)
        near, far = self.config.near, self.config.far
        depth = (far * near / (far - (far - near) * buffer)).copy()
        return CameraFrame(
            rgb, depth, np.asarray(segmentation, dtype=np.int32).reshape(height, width).copy(),
            self.view_matrix, self.projection_matrix, self.pixel_offset,
        )

    def capture_rgb(self) -> "np.ndarray":
        """Return one RGB uint8 image of shape ``(height, width, 3)``."""
        return self.capture_frame().rgb

    def capture_depth(self) -> "np.ndarray":
        """Return metric depth in metres from the same camera pose."""
        return self.capture_frame().depth

    def capture_rgbd(self) -> Tuple["np.ndarray", "np.ndarray"]:
        """Return ``(rgb, metric_depth)`` from one synchronized render."""
        frame = self.capture_frame()
        return frame.rgb, frame.depth

    def _capture_buffers(self):
        # TinyRenderer works in GUI and DIRECT without a GPU context. OpenGL
        # is optional; a blank render triggers a permanent CPU fallback.
        def render(renderer):
            _, _, rgba, depth, segmentation = p.getCameraImage(
                width=self.config.width, height=self.config.height,
                viewMatrix=self.view_matrix, projectionMatrix=self.projection_matrix,
                renderer=renderer, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=self.physics_client_id,
            )
            image = np.asarray(rgba, dtype=np.uint8).reshape(-1, 4)[:, :3]
            if np.ptp(image.astype(np.int16)) < 2 or not np.any(np.asarray(depth) < 0.9999):
                raise RuntimeError("camera returned a blank RGB/depth frame")
            return rgba, depth, segmentation

        if self.active_renderer == "opengl":
            try:
                return render(p.ER_BULLET_HARDWARE_OPENGL)
            except (p.error, RuntimeError):
                self.active_renderer = "tiny"
                print("Camera: blank/unavailable OpenGL render; switched to TinyRenderer")
        return render(p.ER_TINY_RENDERER)


class SimulationEnvironment:
    """Owns the PyBullet connection and static scene geometry."""

    def __init__(self, gui: bool = True, camera_config: Optional[CameraConfig] = None,
                 conveyor_speed: float = 0.0):
        _require_pybullet()
        self.gui = gui
        self._cleanup_callbacks = []
        self.layout = SceneLayout()
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("Unable to connect to PyBullet.")
        self.camera = GlobalCamera(camera_config or CameraConfig(), self.client_id)
        if gui:
            # These Bullet preview panels are not our camera display and can
            # stay white when their internal buffers are not refreshed.
            for flag in (
                p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
            ):
                p.configureDebugVisualizer(flag, 0, physicsClientId=self.client_id)
            p.resetDebugVisualizerCamera(4.6, 35, -45, (0, 0, 0.60))
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        # Keep contacts non-bouncy and let the per-body friction settings in
        # RobotArm.assign_task() provide a stable, low-impulse grasp.
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 240.0,
            numSubSteps=4,
            numSolverIterations=150,
            contactERP=0.10,
            frictionERP=0.20,
            restitutionVelocityThreshold=0.20,
        )
        self.plane_id = self._load_plane()
        self.conveyor_id = self._create_static_conveyor()
        self.conveyor_top_z = self.layout.belt_top_z
        self.arm_support_ids = tuple(
            self._create_grounded_support((x, y), z, (0.045, 0.045))
            for x, y, z in self.layout.arm_bases
        )
        self.arm_ids = self._load_two_ur5_arms()
        self.gripper_attachments = tuple(
            self._attach_two_finger_gripper(arm_id, ee_link_index=6)
            for arm_id in self.arm_ids
        )
        self.gripper_ids = tuple(item.body_id for item in self.gripper_attachments)
        self.drop_zones = self._create_drop_zones()
        self._mark_overlap_zone()
        self.time_step = 1.0 / 240.0
        self.simulation_step = 0
        self.conveyor_speed = float(conveyor_speed)
        if not math.isfinite(self.conveyor_speed) or self.conveyor_speed < 0:
            raise ValueError("conveyor_speed must be finite and nonnegative")
        self.stream_objects = {}
        self._box_collision_shape = None
        self._box_visual_shapes = {}
        self.spawned_count = 0
        self._transport_exclusions = set()
        self._infeed_rng = None
        self._infeed_color_rng = None
        self._previous_infeed_label = None
        self._spawn_limit = None
        self._next_spawn_step = 0
        self._spawn_interval = (4.0, 6.0)
        self._cluster_probability = 0.60
        self._cluster_size = (3, 3)
        self._infeed_colors = []
        if self.conveyor_speed:
            # Bullet has no moving-surface velocity API. Implement an ideal
            # belt velocity boundary on SUPPORTED payloads, not pose resets.
            # Zero surface friction avoids a second, stationary friction law
            # fighting that boundary. Object/drop-zone friction stays enabled.
            p.changeDynamics(self.conveyor_id, -1, lateralFriction=0.0)

    def enable_infeed(self, seed=7, interval=(4.0, 6.0), limit=None,
                      cluster_probability=0.60, cluster_size=(3, 3)):
        """Central infeed: 60% three-item bursts, 40% single-item batches.

        Burst items arrive at the normal safe spacing, never stacked at birth.
        The first colour is random; subsequent batches alternate colours so a
        short demo shows both classes. Setting cluster_probability=0 instead
        selects each item's colour independently with equal probability.
        Colour decisions have their own RNG, independent of position/timing.
        A finite limit may truncate the last batch. This configures future
        arrivals only; it does not remove objects or reset spawned_count.
        """
        low, high = map(float, interval)
        if (not math.isfinite(self.conveyor_speed) or self.conveyor_speed <= 0
                or not all(math.isfinite(v) for v in (low, high)) or not 0 < low <= high):
            raise ValueError("infeed requires a positive belt speed and valid interval")
        if low * self.conveyor_speed < 0.19:
            raise ValueError("spawn interval must leave at least 19 cm between boxes")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise ValueError("limit must be a positive integer or None")
        if not math.isfinite(cluster_probability) or not 0 <= cluster_probability <= 1:
            raise ValueError("cluster_probability must be in [0, 1]")
        if (len(cluster_size) != 2 or any(not isinstance(v, int) or isinstance(v, bool) for v in cluster_size)
                or not 3 <= cluster_size[0] <= cluster_size[1]):
            raise ValueError("cluster_size must be integer bounds with 3 <= MIN <= MAX")
        self._infeed_rng = random.Random(seed)
        self._infeed_color_rng = random.Random(seed)
        self._previous_infeed_label = None
        self._cluster_probability, self._cluster_size = float(cluster_probability), tuple(cluster_size)
        self._infeed_colors.clear()
        self._spawn_interval, self._spawn_limit = (low, high), limit
        self._next_spawn_step = self.simulation_step + 1

    def _next_infeed_label(self):
        if not self._infeed_colors:
            rng = self._infeed_color_rng
            if self._cluster_probability == 0 or self._previous_infeed_label is None:
                label = rng.choice(("Box_A", "Box_B"))
            else:
                label = "Box_B" if self._previous_infeed_label == "Box_A" else "Box_A"
            clustered = self._cluster_probability > 0 and rng.random() < self._cluster_probability
            count = rng.randint(*self._cluster_size) if clustered else 1
            self._previous_infeed_label = label
            self._infeed_colors = [label] * count
        return self._infeed_colors.pop()

    def set_transport_exclusions(self, body_ids):
        """Carried payloads are controlled by the gripper, even during lift-off."""
        self._transport_exclusions = {body for body in body_ids if body is not None}

    def _advance_infeed(self):
        if self._infeed_rng is None or self.simulation_step < self._next_spawn_step:
            return
        if self._spawn_limit is not None and self.spawned_count >= self._spawn_limit:
            return
        label = self._next_infeed_label()
        y = self._infeed_rng.uniform(*self.layout.spawn_y_range)
        color = (0.95, 0.08, 0.03, 1) if label == "Box_A" else (0.03, 0.18, 0.95, 1)
        # Release 3 cm above the belt; gravity settles the 10 cm-high box.
        body = self.create_test_object((self.layout.spawn_x, y, self.conveyor_top_z + 0.08),
                                       shape="box", color=color)
        self.spawned_count += 1
        # Spawn metadata is lifecycle bookkeeping, never a detection input.
        self.stream_objects[body] = (label, self.spawned_count)
        self._next_spawn_step = self.simulation_step + round(
            self._infeed_rng.uniform(*self._spawn_interval) / self.time_step
        )
        print(f"Infeed #{self.spawned_count}: {label}, t={self.simulation_step * self.time_step:.2f}s")

    def _drive_conveyor_contacts(self):
        if not self.conveyor_speed:
            return
        supported = {contact[2] for contact in p.getContactPoints(bodyA=self.conveyor_id)
                     if contact[8] <= 0.002}
        for body in (supported & self.stream_objects.keys()) - self._transport_exclusions:
            velocity, _ = p.getBaseVelocity(body)
            p.resetBaseVelocity(body, linearVelocity=(self.conveyor_speed, 0.0, velocity[2]),
                                angularVelocity=(0.0, 0.0, 0.0))

    def floor_payloads(self):
        """Lifecycle failure sensor; contacts never supply a robot target pose."""
        return {row[2] for row in p.getContactPoints(bodyA=self.plane_id)} & self.stream_objects.keys()

    def remove_stream_object(self, body):
        if body in self.stream_objects:
            p.removeBody(body)
            del self.stream_objects[body]
        self._transport_exclusions.discard(body)

    def get_arm_gripper(self, arm_index: int) -> GripperAttachment:
        """Return the gripper attachment belonging to one zero-based arm."""
        return self.gripper_attachments[arm_index]

    def get_drop_zone(self, zone_index: int) -> DropZone:
        """Indexed compatibility API: Arm1 Blue/Red, then Arm2 Blue/Red."""
        return self.drop_zones[zone_index]

    def get_arm_drop_zone(self, arm_name: str, label: str) -> DropZone:
        """Return an arm's own red or blue destination in world coordinates."""
        name = str(arm_name).lower().replace(" ", "").replace("_", "")
        labels = {"red": "Box_A", "blue": "Box_B", "box_a": "Box_A", "box_b": "Box_B"}
        key = (name, labels.get(str(label).lower(), label))
        try:
            return self.arm_drop_zones[key]
        except KeyError as exc:
            raise ValueError(f"Unknown arm/class destination: {arm_name!r}, {label!r}") from exc

    def _create_drop_zones(self) -> Tuple[DropZone, ...]:
        """Four grounded black platforms with visible red/blue perimeter bars."""
        zones = []
        half_size_xy = self.layout.drop_half_size_xy
        half_height = 0.05
        self.drop_support_ids = []
        self.drop_frame_ids = []
        self.arm_drop_zones = {}
        for spec in self.layout.drop_zone_specs:
            name, xy = spec.name, spec.centre_xy
            color = (0.95, 0.08, 0.03, 1.0) if spec.label == "Box_A" else (0.03, 0.18, 0.95, 1.0)
            surface = (xy[0], xy[1], self.layout.drop_top_z)
            # Grounded plinth under the calibrated tabletop, never a floating slab.
            self.drop_support_ids.append(self._create_grounded_support(
                xy, surface[2] - 2 * half_height, half_size_xy))
            half_extents = (*half_size_xy, half_height)
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_extents, rgbaColor=(0.025, 0.025, 0.03, 1)
            )
            body_id = p.createMultiBody(
                baseMass=0.0, baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=(surface[0], surface[1], surface[2] - half_height),
                baseOrientation=(0.0, 0.0, 0.0, 1.0),
            )
            p.changeDynamics(body_id, -1, lateralFriction=1.2, restitution=0.0)
            # Ordinary visual geometry renders in both GUI and TinyRenderer.
            # The thin non-colliding bars cannot snag a settling payload.
            hx, hy = half_size_xy
            edge, raised = 0.006, 0.003
            for offset, half in (
                ((0, -hy + edge, raised), (hx, edge, raised)),
                ((0, hy - edge, raised), (hx, edge, raised)),
                ((-hx + edge, 0, raised), (edge, hy, raised)),
                ((hx - edge, 0, raised), (edge, hy, raised)),
            ):
                frame_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
                self.drop_frame_ids.append(p.createMultiBody(
                    baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=frame_visual,
                    basePosition=tuple(surface[i] + offset[i] for i in range(3)),
                    baseOrientation=(0, 0, 0, 1)))
            p.addUserDebugText(
                name, (surface[0] - 0.17, surface[1], surface[2] + 0.015),
                textColorRGB=(1, 1, 1), textSize=0.8,
            )
            zone = DropZone(name, body_id, surface, half_size_xy, color)
            zones.append(zone)
            self.arm_drop_zones[(spec.arm_name, spec.label)] = zone
        return tuple(zones)

    def _mark_overlap_zone(self):
        """Non-colliding debug outline; does not alter RGB/depth detections."""
        xmin, xmax, ymin, ymax = self.layout.overlap_bounds
        z = self.conveyor_top_z + 0.003
        corners = ((xmin, ymin, z), (xmax, ymin, z), (xmax, ymax, z), (xmin, ymax, z))
        for first, second in zip(corners, corners[1:] + corners[:1]):
            p.addUserDebugLine(first, second, (1, .8, 0), lineWidth=2)
        self.overlap_label_position = ((xmin + xmax) / 2 - .18, ymax + .025, z + .01)
        p.addUserDebugText("OVERLAP / SHARED PICKUP", self.overlap_label_position,
                           textColorRGB=(1, .8, 0), textSize=0.7)

    def _create_grounded_support(self, xy, top_z, half_size_xy):
        """A fixed, horizontal plinth spanning ground z=0 through top_z."""
        half_extents = (*half_size_xy, top_z / 2)
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                     rgbaColor=(.28, .30, .32, 1))
        return p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
            basePosition=(*xy, top_z / 2), baseOrientation=(0, 0, 0, 1))

    def _load_plane(self) -> int:
        return p.loadURDF("plane.urdf", basePosition=(0.0, 0.0, 0.0))

    def _create_static_conveyor(self) -> int:
        # x: belt travel direction; y: belt width; z: belt top is 0.42 m.
        length, width, height = self.layout.belt_length, self.layout.belt_width, self.layout.belt_top_z
        center_z = height / 2  # bottom rests on z=0; top stays at the calibrated height
        visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=(length / 2, width / 2, height / 2),
            rgbaColor=(0.12, 0.12, 0.12, 1.0),
        )
        collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=(length / 2, width / 2, height / 2)
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=(*self.layout.belt_center_xy, center_z),
            baseOrientation=(0.0, 0.0, 0.0, 1.0),
        )

    def _attach_two_finger_gripper(
        self, arm_id: int, ee_link_index: int
    ) -> GripperAttachment:
        """Create a small two-prismatic-finger gripper and fix it to the UR5."""
        ee_state = p.getLinkState(arm_id, ee_link_index, computeForwardKinematics=True)
        ee_position, ee_orientation = ee_state[4], ee_state[5]
        base_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=(0.055, 0.055, 0.018)
        )
        base_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=(0.055, 0.055, 0.018),
            rgbaColor=(0.18, 0.18, 0.20, 1.0),
        )
        finger_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=(0.012, 0.018, 0.055)
        )
        finger_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=(0.012, 0.018, 0.055),
            rgbaColor=(0.75, 0.75, 0.78, 1.0),
        )
        gripper_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=ee_position,
            baseOrientation=ee_orientation,
            linkMasses=[0.01, 0.01],
            linkCollisionShapeIndices=[finger_collision, finger_collision],
            linkVisualShapeIndices=[finger_visual, finger_visual],
            linkPositions=[(0.0, 0.040, 0.070), (0.0, -0.040, 0.070)],
            linkOrientations=[(0.0, 0.0, 0.0, 1.0)] * 2,
            linkInertialFramePositions=[(0.0, 0.0, 0.0)] * 2,
            linkInertialFrameOrientations=[(0.0, 0.0, 0.0, 1.0)] * 2,
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_PRISMATIC, p.JOINT_PRISMATIC],
            linkJointAxis=[(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        )
        # Keep the simplified box fingers non-bouncy and sufficiently grippy
        # for slow contact with the test objects.
        for link_index in (-1, 0, 1):
            p.changeDynamics(
                gripper_id,
                link_index,
                lateralFriction=1.0,
                spinningFriction=0.02,
                rollingFriction=0.02,
                restitution=0.0,
            )
            # The gripper palm overlaps its mounting flange by design.  A
            # fixed constraint and collision response between these attached
            # bodies would fight each other and displace the real TCP.
            for arm_link in range(-1, p.getNumJoints(arm_id)):
                p.setCollisionFilterPair(
                    gripper_id, arm_id, link_index, arm_link, enableCollision=0
                )
            p.changeDynamics(
                gripper_id, link_index,
                contactStiffness=1500.0, contactDamping=100.0,
            )
        constraint_id = p.createConstraint(
            parentBodyUniqueId=arm_id,
            parentLinkIndex=ee_link_index,
            childBodyUniqueId=gripper_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0.0, 0.0, 0.0),
            parentFramePosition=(0.0, 0.0, 0.0),
            childFramePosition=(0.0, 0.0, 0.0),
            parentFrameOrientation=(0.0, 0.0, 0.0, 1.0),
            childFrameOrientation=(0.0, 0.0, 0.0, 1.0),
        )
        p.changeConstraint(constraint_id, maxForce=500.0, erp=0.8)
        # Calibrate TCP from the midpoint of the two finger tips.  The finger
        # links are 0.125 m from the gripper base at their tips (0.07 + 0.055).
        tip_positions = []
        for link_index in (0, 1):
            link_state = p.getLinkState(
                gripper_id, link_index, computeForwardKinematics=True
            )
            tip_positions.append(
                p.multiplyTransforms(
                    link_state[4], link_state[5], (0.0, 0.0, 0.055),
                    (0.0, 0.0, 0.0, 1.0),
                )[0]
            )
        tcp_world = tuple((tip_positions[0][i] + tip_positions[1][i]) / 2.0 for i in range(3))
        inv_ee = p.invertTransform(ee_position, ee_orientation)
        tcp_offset_local = p.multiplyTransforms(
            inv_ee[0], inv_ee[1], tcp_world, (0.0, 0.0, 0.0, 1.0)
        )[0]
        return GripperAttachment(
            body_id=gripper_id,
            constraint_id=constraint_id,
            joint_indices=(0, 1),
            tcp_offset_local=tuple(float(value) for value in tcp_offset_local),
        )

    def create_test_object(
        self,
        position: Tuple[float, float, float],
        shape: str = "cylinder",
        orientation: Optional[Tuple[float, float, float, float]] = None,
        color: Tuple[float, float, float, float] = (0.95, 0.12, 0.05, 1.0),
    ) -> int:
        """Create a stable, simple-collision test object on the conveyor.

        The default cylinder is rotated so its axis follows the belt's X
        direction.  This helper is intended for deterministic demos and
        smoke tests; real objects continue to enter through the camera/YOLO
        path and are passed to ``RobotArm.assign_task`` by their visual point.
        """
        shape_name = shape.strip().lower()
        if shape_name == "cylinder":
            radius, length = 0.05, 0.18
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=radius, height=length
            )
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=length,
                rgbaColor=color,
            )
            # PyBullet cylinders use local Z as their axis.  Rotate Z onto X
            # so the object lies along the conveyor travel direction.
            object_orientation = orientation or p.getQuaternionFromEuler(
                (0.0, math.pi / 2.0, 0.0)
            )
        elif shape_name == "box":
            half_extents = (0.08, 0.05, 0.05)
            # Reuse immutable geometry across an endless infeed. Removing a
            # payload body must not leave a new mesh/shape allocation per box.
            if self._box_collision_shape is None:
                self._box_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            key = tuple(color)
            if key not in self._box_visual_shapes:
                self._box_visual_shapes[key] = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color,
                )
            collision, visual = self._box_collision_shape, self._box_visual_shapes[key]
            object_orientation = orientation or (0.0, 0.0, 0.0, 1.0)
        else:
            raise ValueError("shape must be 'cylinder' or 'box'")

        object_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=tuple(float(value) for value in position),
            baseOrientation=object_orientation,
        )
        p.changeDynamics(
            object_id,
            -1,
            lateralFriction=1.2,
            spinningFriction=0.05,
            rollingFriction=0.05,
            restitution=0.0,
            contactStiffness=1500.0,
            contactDamping=100.0,
            linearDamping=0.04,
            angularDamping=0.04,
        )
        return int(object_id)

    def _find_ur5_urdf(self) -> str:
        candidates = []
        configured = os.environ.get("UR5_URDF_PATH")
        if configured:
            candidates.append(Path(configured))
        if pybullet_data is not None:
            data_root = Path(pybullet_data.getDataPath())
            candidates.extend((data_root / "ur5" / "ur5.urdf", data_root / "ur5.urdf"))
        candidates.append(Path(__file__).resolve().parent / "assets" / "ur5" / "ur5.urdf")
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "UR5 URDF not found. Set UR5_URDF_PATH or place the asset in "
            f"assets/ur5/ur5.urdf. Searched: {searched}"
        )

    def _load_two_ur5_arms(self) -> Tuple[int, int]:
        ur5_path = self._find_ur5_urdf()
        # Fixed bases on opposite sides of the belt. The exact IK/control
        # conventions remain the responsibility of robot_arm.py.
        arm_1 = p.loadURDF(
            ur5_path, basePosition=self.layout.arm_bases[0],
            baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, self.layout.arm_base_yaws[0])),
            useFixedBase=True,
        )
        arm_2 = p.loadURDF(
            ur5_path, basePosition=self.layout.arm_bases[1],
            baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, self.layout.arm_base_yaws[1])),
            useFixedBase=True,
        )
        # Seed from the validated outward elbow posture, then fold into the
        # nearby standby pose. This keeps IK on the same collision-free branch.
        for index, robot in enumerate((arm_1, arm_2)):
            joints = [j for j in range(p.getNumJoints(robot))
                      if p.getJointInfo(robot, j)[2] == p.JOINT_REVOLUTE]
            orientation = p.getQuaternionFromEuler((math.pi, 0, self.layout.arm_base_yaws[index]))
            base = self.layout.arm_bases[index]
            seed = (base[0] + (-.35 if index == 0 else .35),
                    base[1] + math.copysign(.30, base[1]), base[2] + .10)
            home = self.layout.home_tcp_positions[index]
            target = (home[0], home[1], home[2] + .125)  # downward tool: EE is 12.5 cm above TCP
            for pose in (seed, target):
                for _ in range(12):
                    solution = p.calculateInverseKinematics(robot, 6, pose, orientation,
                                                            maxNumIterations=1000, residualThreshold=1e-7)
                    for joint, value in zip(joints, solution):
                        lo, hi = p.getJointInfo(robot, joint)[8:10]
                        # Equivalent angles at load time leave travel in both directions;
                        # a wrist seeded near 2*pi otherwise stalls against its joint stop.
                        value = math.remainder(value, 2 * math.pi)
                        p.resetJointState(robot, joint, min(hi, max(lo, value)))
            measured = p.getLinkState(robot, 6, computeForwardKinematics=True)[4]
            if math.dist(measured, target) > .003:
                raise RuntimeError(f"Arm {index+1} parking pose is unreachable")
        return arm_1, arm_2

    def get_global_rgb(self) -> "np.ndarray":
        """Capture the global camera image through the camera interface."""
        return self.camera.capture_rgb()

    def step(self, simulation_steps: int = 1) -> None:
        """Advance discrete simulation steps; never blocks wall-clock time."""
        if simulation_steps < 1:
            raise ValueError("simulation_steps must be >= 1")
        for _ in range(simulation_steps):
            self.simulation_step += 1
            self._advance_infeed()
            self._drive_conveyor_contacts()
            p.stepSimulation()

    def register_cleanup(self, callback) -> None:
        """Release attached controller resources before disconnecting physics."""
        self._cleanup_callbacks.append(callback)

    def close(self) -> None:
        callbacks, self._cleanup_callbacks = self._cleanup_callbacks, []
        try:
            for callback in reversed(callbacks):
                callback()
        finally:
            if p is not None and p.isConnected(self.client_id):
                p.disconnect(self.client_id)


def _require_pybullet() -> None:
    if p is None or np is None or pybullet_data is None:
        raise ImportError("Install pybullet and numpy before using env_setup.py")


if __name__ == "__main__":
    try:
        env = SimulationEnvironment(gui=False)
        test_object = env.create_test_object(
            (0.0, 0.0, env.conveyor_top_z + 0.05), shape="cylinder"
        )
        mass, friction = p.getDynamicsInfo(test_object, -1)[:2]
        print(
            f"Test object OK: id={test_object}, mass={mass:.2f} kg, "
            f"lateralFriction={friction:.2f}"
        )
        env.step(1)
        for index, zone in enumerate(env.drop_zones):
            assert p.getDynamicsInfo(zone.body_id, -1)[0] == 0.0
            assert abs(zone.surface_position[1]) - zone.half_size_xy[1] > 0.20
            print(f"Drop zone: {zone.name}, surface={zone.surface_position}")
        image = env.get_global_rgb()
        depth = env.camera.capture_depth()
        assert image.std() > 10 and np.ptp(depth) > 0.1, "blank camera image"
        print(f"Environment OK: RGB image shape={image.shape}, dtype={image.dtype}")
        print(
            f"Camera OK: renderer={env.camera.active_renderer}, "
            f"depth range={depth.min():.3f}..{depth.max():.3f} m"
        )
        env.step(10)
        env.close()
        env = SimulationEnvironment(gui=False, conveyor_speed=0.05)
        env.enable_infeed(seed=7, limit=1)
        env.step(240)
        body, = env.stream_objects
        assert p.getContactPoints(body, env.conveyor_id)
        assert abs(p.getBaseVelocity(body)[0][0] - 0.05) < 0.001
        env.set_transport_exclusions([body])
        p.resetBaseVelocity(body, (0, 0, 0))
        env.step()
        assert abs(p.getBaseVelocity(body)[0][0]) < 0.001, "belt drove an excluded payload"
        env.remove_stream_object(body)
        assert not env.stream_objects
        fallen = env.create_test_object((0, 0, 0.05), shape="box")
        env.stream_objects[fallen] = ("Box_A", 2)
        env.step(120)
        assert fallen in env.floor_payloads(), "floor-contact failure sensor missed a fallen box"
        env.remove_stream_object(fallen)
        env.close()
        print("PASS: continuous belt, infeed limit, held-payload exclusion and floor failure sensor")
    except Exception as exc:
        print(f"Environment test could not run: {exc}")
        raise
