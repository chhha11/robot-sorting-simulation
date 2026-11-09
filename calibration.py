"""Pixel/depth to robot-world calibration adapters (metres).

SimulationCalibration uses OpenGL projection/view matrices. On hardware inject
RGBDCalibration with measured intrinsics and a camera-to-world hand-eye pose.
Depth must be registered to RGB and expressed as camera-Z metres; rectify RGB
and depth with the camera SDK before using these adapters.
"""
from typing import Optional, Sequence, Tuple
import numpy as np

class SimulationCalibration:
    """OpenGL RGB-D unprojection adapter; no physics/object API dependency."""

    def __init__(self, camera):
        self.camera = camera
        self._view_matrix = np.asarray(camera.view_matrix, dtype=np.float64).reshape(4, 4, order="F")
        self._projection_matrix = np.asarray(camera.projection_matrix, dtype=np.float64).reshape(4, 4, order="F")
        self._inverse_camera_matrix = np.linalg.inv(self._projection_matrix @ self._view_matrix)

    def pixel_to_world(
        self,
        pixel: Sequence[float],
        depth_m: float,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, float, float]:
        """Unproject ``(u, v, depth_m)`` into a world-space point.

        ``depth_m`` is metric camera depth returned by ``capture_rgbd``. It is
        converted back to the OpenGL depth buffer before multiplying by the
        inverse projection-view matrix.
        """
        config = self.camera.config
        width, height = image_size or (config.width, config.height)
        u, v = float(pixel[0]), float(pixel[1])
        offset_u, offset_v = self.camera.pixel_offset
        u, v = u + offset_u, v + offset_v
        near, far = config.near, config.far
        if not near < depth_m < far:
            raise ValueError(f"Depth {depth_m:.4f} m is outside camera range")
        depth_buffer = (far - near * far / depth_m) / (far - near)
        # Match OpenGL viewport dimensions, not width-1/height-1.
        ndc_x = 2.0 * u / width - 1.0
        ndc_y = 1.0 - 2.0 * v / height
        ndc_z = 2.0 * depth_buffer - 1.0
        world_h = self._inverse_camera_matrix @ np.array(
            [ndc_x, ndc_y, ndc_z, 1.0], dtype=np.float64
        )
        if abs(world_h[3]) < 1e-9:
            raise ValueError("Homogeneous world coordinate has near-zero scale")
        world = world_h[:3] / world_h[3]
        return tuple(float(value) for value in world)


    def pixels_to_world(self, columns, rows, depths, image_size=None):
        """Vectorized registered RGB-D unprojection; returns 3xN world points."""
        config = self.camera.config
        width, height = image_size or (config.width, config.height)
        depths = np.asarray(depths, dtype=float)
        if not np.all(np.isfinite(depths) & (depths > config.near) & (depths < config.far)):
            raise ValueError("Depth samples outside calibrated camera range")
        du, dv = self.camera.pixel_offset
        buffer = (config.far - config.near * config.far / depths) / (config.far - config.near)
        clip = np.stack((2 * (np.asarray(columns) + du) / width - 1,
                         1 - 2 * (np.asarray(rows) + dv) / height,
                         2 * buffer - 1, np.ones(len(depths))))
        world = self._inverse_camera_matrix @ clip
        if np.any(np.abs(world[3]) < 1e-9):
            raise ValueError("Homogeneous world coordinate has near-zero scale")
        return world[:3] / world[3]


class RGBDCalibration:
    """Pinhole intrinsics K and rigid camera-to-world transform (4x4)."""

    def __init__(self, intrinsics, camera_to_world):
        k = np.asarray(intrinsics, dtype=float)
        pose = np.asarray(camera_to_world, dtype=float)
        if k.shape != (3, 3) or pose.shape != (4, 4):
            raise ValueError("Expected 3x3 intrinsics and 4x4 camera-to-world pose")
        if not np.all(np.isfinite(k)) or not np.all(np.isfinite(pose)):
            raise ValueError("Calibration matrices must be finite")
        if (not np.allclose(k[2], [0, 0, 1]) or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1)):
            raise ValueError("Expected pinhole intrinsics and a rigid camera-to-world pose")
        self.intrinsics = k.copy()
        self.camera_to_world = pose.copy()
        self._inverse_intrinsics = np.linalg.inv(k)

    def pixel_to_world(self, pixel, depth_m, image_size=None):
        u, v = map(float, pixel)
        if not np.all(np.isfinite([u, v, depth_m])) or depth_m <= 0:
            raise ValueError("Pixel and positive metric depth must be finite")
        if image_size is not None and not (0 <= u < image_size[0] and 0 <= v < image_size[1]):
            raise ValueError("Pixel outside calibrated image")
        camera_point = self._inverse_intrinsics @ np.array([u, v, 1.0]) * depth_m
        world = self.camera_to_world @ np.r_[camera_point, 1.0]
        return tuple(float(value) for value in world[:3])
