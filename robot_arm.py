"""Backend-independent arm contract and explicit hardware-driver extension point.

SimRobotArm (and the legacy RobotArm alias) load PyBullet only on demand.
Positions are calibrated world-frame TCP coordinates in metres. update() is
one non-blocking control tick; it never advances the application's clock.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Sequence, Tuple

Position = Tuple[float, float, float]


class ArmState(Enum):
    IDLE = auto()
    MOVE_TO_TARGET_ABOVE = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    MOVE_TO_DROP_ABOVE = auto()
    DESCEND_TO_DROP = auto()
    RELEASE = auto()
    LIFT_AFTER_RELEASE = auto()
    RETURN_HOME = auto()
    RETRY = auto()
    ABORT = auto()

    MOVING_TO_TARGET = MOVE_TO_TARGET_ABOVE
    DESCENDING = DESCEND
    GRASPING = GRASP
    LIFTING = LIFT
    MOVING_TO_DROP = MOVE_TO_DROP_ABOVE
    RELEASING = RELEASE
    RETURNING = RETURN_HOME


@dataclass(frozen=True)
class ArmHandoffStatus:
    """Driver feedback needed for a safe handoff, without physics handles.

    eligible must be latched only after a pre-grasp failure. has_payload uses
    grasp feedback; recovering means an active, safe abort/retraction phase.
    Drivers without that feedback conservatively keep eligible=False.
    """

    eligible: bool = False
    has_payload: bool = False
    collision: bool = False
    recovery_blocked: bool = False
    recovering: bool = False


class BaseRobotArm(ABC):
    """Common asynchronous command/feedback boundary for every arm driver.

    assign_task starts an accepted task and returns promptly; update polls or
    advances it once without waiting for task completion. A terminal failure
    stays available through get_last_error until the next accepted task.
    is_idle is true only when ready to accept work, and becomes true after a
    successful task only once its release and return motion are complete.
    Driver-specific **options are opaque to the scheduler.
    """

    def ensure_ready(self) -> None:
        """Validate an already initialized driver before starting dispatch."""

    @abstractmethod
    def assign_task(self, target_pos: Sequence[float], drop_pos: Sequence[float], **options) -> None:
        """Start a world-frame pick/place command; reject work while busy."""

    @abstractmethod
    def update(self) -> ArmState:
        """Advance/poll one tick without sleeping or waiting for a peer arm."""

    @abstractmethod
    def is_idle(self) -> bool:
        """Whether the driver is ready for another task."""

    @abstractmethod
    def get_last_error(self) -> Optional[str]:
        """Return the latched task error, or None after success/new acceptance."""

    @abstractmethod
    def get_tcp_position(self) -> Position:
        """Latest measured world-frame TCP in metres, never a command target."""

    @abstractmethod
    def cancel_task(self, reason: str) -> None:
        """Request safe asynchronous recovery and latch a failure reason."""

    @abstractmethod
    def estimate_pick_seconds(self, target_pos: Sequence[float]) -> float:
        """Estimate lead time for moving-target prediction."""

    @abstractmethod
    def update_tracking_observation(self, position: Sequence[float], velocity: Sequence[float]) -> None:
        """Refresh a running task with measured world-frame target motion."""

    @property
    def is_tracking_target(self) -> bool:
        return False

    def get_handoff_status(self) -> ArmHandoffStatus:
        return ArmHandoffStatus()

    def set_traffic_task(self, track_id, handoff_count=0) -> None:
        """Optional task metadata for a backend's local traffic controller."""


class RealRobotArm(BaseRobotArm):
    """Hardware driver scaffold. No connection, motor commands or fake feedback.

    Implement non-blocking Socket/ROS transport, calibrated TCP feedback and
    hardware grasp/recovery acknowledgements before enabling real dispatch.
    """

    def __init__(self, display_name="Real Arm"):
        self.display_name = display_name
        self._last_error = ("RealRobotArm hardware driver is not implemented: "
                            "connect a Socket/ROS transport and calibrated feedback first.")

    def ensure_ready(self) -> None:
        raise NotImplementedError(self._last_error)

    def assign_task(self, target_pos, drop_pos, **options) -> None:
        # TODO: validate world/base calibration; enqueue a pick/place command
        # with a command ID. Send via non-blocking Socket or ROS action/topic.
        # Only mark a task active after the hardware controller accepts it.
        self.ensure_ready()

    def update(self) -> ArmState:
        # TODO: poll Socket messages / a ROS callback queue without blocking.
        # Update measured TCP, gripper feedback, task state and latched errors.
        # A sent command alone must never be treated as a completed task.
        self.ensure_ready()

    def is_idle(self) -> bool:
        # A disconnected/unimplemented driver cannot accept scheduled work.
        return False

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    def get_tcp_position(self) -> Position:
        # TODO: return calibrated encoder/controller feedback, in metres.
        self.ensure_ready()

    def cancel_task(self, reason: str) -> None:
        # TODO: request a hardware safe stop/retract and await acknowledgement
        # in later update ticks. Retain reason until the next accepted task.
        self.ensure_ready()

    def estimate_pick_seconds(self, target_pos) -> float:
        # TODO: estimate from controller speed/trajectory feedback.
        self.ensure_ready()

    def update_tracking_observation(self, position, velocity) -> None:
        # TODO: enqueue a supported tracking update with its measurement time.
        self.ensure_ready()


def create_robot_arm(backend="sim", *args, **kwargs) -> BaseRobotArm:
    """Select a driver without loading simulation dependencies for real mode."""
    if backend == "real":
        return RealRobotArm(*args, **kwargs)
    if backend == "sim":
        from sim_robot_arm import SimRobotArm
        return SimRobotArm(*args, **kwargs)
    raise ValueError(f"Unknown robot backend {backend!r}; choose 'sim' or 'real'")


def get_arm_last_error(arm) -> Optional[str]:
    """Read the public driver API; support existing third-party duck adapters."""
    reader = getattr(arm, "get_last_error", None)
    return reader() if callable(reader) else getattr(arm, "last_error", None)


def get_arm_handoff_status(arm) -> ArmHandoffStatus:
    """Prefer public feedback, with an isolated legacy adapter compatibility path.

    New drivers implement get_handoff_status; only older duck-typed adapters
    need these historical fields. The scheduler never reads physics handles.
    """
    reader = getattr(arm, "get_handoff_status", None)
    if callable(reader):
        return reader()
    return ArmHandoffStatus(
        eligible=bool(getattr(arm, "handoff_eligible", False)),
        has_payload=getattr(arm, "held_object_id", None) is not None,
        collision=getattr(arm, "last_collision", None) is not None,
        recovery_blocked=bool(getattr(arm, "_collision_recovery_blocked", False)),
        recovering=getattr(getattr(arm, "state", None), "name", None) in {"ABORT", "RETURN_HOME"},
    )


def __getattr__(name):
    # Python's from-import also probes __path__; never load Bullet for that.
    public_sim = {"SimRobotArm", "RobotArm", "SharedWorkspaceGate"}
    legacy_helpers = {"_as_position", "_run_collision_guard_test", "_run_grasp_failure_test"}
    if name not in public_sim | legacy_helpers:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import sim_robot_arm
    value = getattr(sim_robot_arm, "SimRobotArm" if name == "RobotArm" else name)
    globals()[name] = value
    return value


__all__ = ["BaseRobotArm", "SimRobotArm", "RealRobotArm", "RobotArm", "ArmState",
           "ArmHandoffStatus", "SharedWorkspaceGate", "create_robot_arm"]


if __name__ == "__main__":
    # Keep the original controller-only regression command working.
    import runpy
    runpy.run_module("sim_robot_arm", run_name="__main__")
