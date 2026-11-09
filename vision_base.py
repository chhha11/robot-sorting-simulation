"""Shared vision contract, independent of detector and simulation APIs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Protocol, Sequence, Tuple, TypedDict

import numpy as np

Position = Tuple[float, float, float]
BoundingBox = Tuple[float, float, float, float]

# Keep the historical Box_A / Box_B semantic names used by class routing.
# pos_3d is a measured surface point in the ROBOT WORLD frame, in metres.
# bbox and pixel_center use RGB pixel coordinates; depth uses camera-Z metres.
# Extra tracking fields are shared by both implementations, not backend data.
VisionResult = TypedDict("VisionResult", {
    "class": str,
    "pos_3d": Position,
    "bbox": BoundingBox,
    "track_id": int,
    "pixel_center": Tuple[float, float],
    "confidence": float,
    "observed_at": float,
    "velocity": Position,
})


class RGBDCamera(Protocol):
    """Real SDK adapters return synchronized RGB and metric camera-Z depth."""

    def capture_rgbd(self) -> Tuple[np.ndarray, np.ndarray]: ...


class WorldCoordinateConverter(Protocol):
    """Implement with calibrated intrinsics and camera-to-robot extrinsics."""

    def pixel_to_world(
        self, pixel: Sequence[float], depth_m: float,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Position: ...


class BaseVision(ABC):
    @abstractmethod
    def detect(
        self, rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> List[VisionResult]:
        """Return measured world detections with stable visual track IDs.

        With no images, capture one synchronized pair through the camera
        adapter. With images, use that exact pair without capturing again.
        timestamp is seconds in the caller's clock domain and must increase.
        Empty/occluded frames return []; they never invent 3D observations.
        """
        raise NotImplementedError

    def forget_track(self, track_id: int) -> None:
        """Optional lifecycle hook when a sorted/missed payload is retired."""


if __name__ == "__main__":
    try:
        BaseVision()
    except TypeError:
        print("PASS: BaseVision requires a concrete detect() implementation")
    else:
        raise AssertionError("abstract vision was instantiated")

    class EmptyVision(BaseVision):
        def detect(self, rgb_image=None, depth_image=None, timestamp=None):
            return []

    vision: BaseVision = EmptyVision()
    assert vision.detect() == []
    vision.forget_track(1)
    print("PASS: polymorphic detection and optional retirement hook")
