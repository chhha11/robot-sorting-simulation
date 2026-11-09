"""Interchangeable color and YOLO vision with one world-coordinate contract.

RGB/depth -> pixel detections -> calibrated world points -> visual track IDs.
No simulated object pose or segmentation is used for detection or tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vision_base import BaseVision, RGBDCamera, VisionResult, WorldCoordinateConverter
from calibration import SimulationCalibration


@dataclass(frozen=True)
class Detection:
    """Internal pixel observation; never exposed to the scheduler."""

    class_id: int
    label: str
    confidence: float
    box_xyxy: Tuple[float, float, float, float]
    sample_pixel: Optional[Tuple[float, float]] = None

    @property
    def pixel_center(self):
        if self.sample_pixel is not None:
            return self.sample_pixel
        x1, y1, x2, y2 = self.box_xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


class _RgbNoise:
    """Optional sensor/lighting perturbation, independent of detection backend.

    ``noise_std`` is an 8-bit RGB intensity standard deviation. Lighting jitter
    samples one contrast multiplier in 1 +/- jitter and one brightness offset
    in +/- 255*jitter for each exposure. Depth/calibration are never modified.
    A local generator makes experiments reproducible without seeding physics.
    """

    def __init__(self, noise_std=0.0, lighting_jitter=0.0, noise_seed=None):
        self.noise_std = float(noise_std)
        self.lighting_jitter = float(lighting_jitter)
        if not np.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be a finite nonnegative RGB intensity")
        if not np.isfinite(self.lighting_jitter) or not 0 <= self.lighting_jitter <= 0.5:
            raise ValueError("lighting_jitter must be a finite value in [0, 0.5]")
        self.noise_seed = noise_seed
        self.rng = np.random.default_rng(noise_seed)

    @property
    def settings(self):
        return {"noise_std": self.noise_std, "lighting_jitter": self.lighting_jitter,
                "noise_seed": self.noise_seed}

    def apply(self, rgb):
        if self.noise_std == 0 and self.lighting_jitter == 0:
            return rgb
        # Allocate only for an enabled experiment; never change camera storage.
        values = rgb.astype(np.float32)
        if self.lighting_jitter:
            jitter = self.lighting_jitter
            contrast = self.rng.uniform(1 - jitter, 1 + jitter)
            brightness = self.rng.uniform(-255 * jitter, 255 * jitter)
            values -= 127.5
            values *= contrast
            values += 127.5 + brightness
        if self.noise_std:
            noise = self.rng.standard_normal(values.shape, dtype=np.float32)
            noise *= self.noise_std
            values += noise
        np.clip(values, 0, 255, out=values)
        return np.rint(values).astype(np.uint8)


class _WorldTracker:
    """Shared visual association with calibrated conveyor motion compensation."""

    def __init__(self, velocity):
        self.velocity = tuple(float(value) for value in velocity)
        if len(self.velocity) != 3 or not np.all(np.isfinite(self.velocity)):
            raise ValueError("conveyor_velocity must be three finite values")
        self.retention_seconds = 12.0
        self.tracks = {}
        self.next_track_id = 1

    def update(self, observations, timestamp) -> List[VisionResult]:
        self.tracks = {key: value for key, value in self.tracks.items()
                       if timestamp - value[2] <= self.retention_seconds}
        predictions = {
            track_id: (label, np.asarray(position) + np.asarray(self.velocity) * (timestamp - seen))
            for track_id, (label, position, seen) in self.tracks.items()
        }
        matches = sorted(
            (float(np.linalg.norm(np.subtract(world, predicted))), index, track_id)
            for index, (detection, world) in enumerate(observations)
            for track_id, (old_label, predicted) in predictions.items()
            if old_label == detection.label
            and np.linalg.norm(np.subtract(world, predicted)) < 0.075
        )
        matched, used_tracks = {}, set()
        for _, index, track_id in matches:
            if index not in matched and track_id not in used_tracks:
                matched[index] = track_id
                used_tracks.add(track_id)
        result = []
        for index, (detection, world) in enumerate(observations):
            track_id = matched.get(index)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
            self.tracks[track_id] = (detection.label, world, timestamp)
            result.append({
                "class": detection.label, "pos_3d": tuple(world),
                "bbox": tuple(float(value) for value in detection.box_xyxy),
                "track_id": track_id, "pixel_center": detection.pixel_center,
                "confidence": detection.confidence, "observed_at": timestamp,
                "velocity": self.velocity,
            })
        return result


def _images(camera, rgb_image, depth_image):
    if rgb_image is None and depth_image is None:
        if camera is None:
            raise NotImplementedError("Connect a synchronized real RGB-D camera adapter first")
        rgb_image, depth_image = camera.capture_rgbd()
    elif rgb_image is None or depth_image is None:
        raise ValueError("RGB and depth must be supplied together from the same frame")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected an H x W x 3 RGB image")
    if depth_image.ndim != 2 or depth_image.shape != rgb_image.shape[:2]:
        raise ValueError("Expected registered H x W metric depth matching RGB")
    return rgb_image, depth_image


def _depth_at(pixel, depth_image, window=2):
    u, v = (int(round(value)) for value in pixel)
    height, width = depth_image.shape
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError("Detection center is outside the image")
    samples = depth_image[max(0, v-window):v+window+1, max(0, u-window):u+window+1]
    valid = samples[np.isfinite(samples) & (samples > 0)]
    if not valid.size:
        raise ValueError("No valid metric depth at detection center")
    return float(np.median(valid))


def _inside(world, bounds):
    if len(world) != 3 or not np.all(np.isfinite(world)):
        return False
    if bounds is None:
        return True
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x, y, z = world
    return xmin < x < xmax and ymin < y < ymax and zmin < z < zmax


def _simulation_pickup_bounds(environment):
    """Calibrated belt ROI, shared by color, mock YOLO and neural adapters.

    Only the measured tops of conveyor payloads are eligible for tracking.
    Dedicated placement platforms and their colored outlines sit outside this
    XY rectangle; the height band also rejects the bare belt and held objects.
    No simulated body identity or payload position is used here.
    """
    belt_z = environment.conveyor_top_z
    return (*environment.layout.belt_bounds, belt_z + 0.06, belt_z + 0.15)


class SimColorVision(BaseVision):
    """Simulation RGB color detection, depth unprojection and moving tracking."""

    def __init__(self, environment, conveyor_velocity: Sequence[float] = (0, 0, 0),
                 *, noise_std=0.0, lighting_jitter=0.0, noise_seed=None):
        self.environment = environment
        self.camera = environment.camera
        self._geometry = SimulationCalibration(self.camera)
        self._inverse_camera_matrix = self._geometry._inverse_camera_matrix
        self._tracker = _WorldTracker(conveyor_velocity)
        self._rgb_noise = _RgbNoise(noise_std, lighting_jitter, noise_seed)
        self.last_rgb_image = None
        self.pickup_bounds = _simulation_pickup_bounds(environment)

    def get_camera_data(self):
        """Capture one synchronized RGB / metric-depth pair after physics steps."""
        return self.camera.capture_rgbd()

    @property
    def noise_settings(self):
        return self._rgb_noise.settings

    @property
    def view_matrix(self):
        return self._geometry._view_matrix.copy()

    @property
    def projection_matrix(self):
        return self._geometry._projection_matrix.copy()

    def pixel_to_world(self, pixel, depth_m, image_size=None):
        return self._geometry.pixel_to_world(pixel, depth_m, image_size)

    def detect(self, rgb_image=None, depth_image=None, timestamp=None) -> List[VisionResult]:
        rgb_image, depth_image = _images(self.camera, rgb_image, depth_image)
        rgb_image = self._rgb_noise.apply(rgb_image)
        self.last_rgb_image = rgb_image
        timestamp = float(timestamp if timestamp is not None else
                          self.environment.simulation_step * self.environment.time_step)
        observations = []
        for detection in self._top_surface_detections(rgb_image, depth_image):
            try:
                world = self.detection_to_world(detection, depth_image)
            except ValueError:
                continue
            # Exclude colored drop zones, carried payloads and occluded slivers.
            if _inside(world, self.pickup_bounds):
                observations.append((detection, world))
        return self._tracker.update(observations, timestamp)

    def forget_track(self, track_id):
        self._tracker.tracks.pop(track_id, None)

    @staticmethod
    def _color_masks(rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        return (
            ("Box_A", 0, cv2.inRange(hsv, (0, 100, 70), (12, 255, 255))
             | cv2.inRange(hsv, (168, 100, 70), (180, 255, 255))),
            ("Box_B", 1, cv2.inRange(hsv, (95, 100, 70), (130, 255, 255))),
        )

    def detect_colors(self, rgb_image: np.ndarray) -> List[Detection]:
        """Classify red Box_A and blue Box_B using RGB pixels only."""
        detections = []
        for label, class_id, mask in self._color_masks(rgb_image):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 30:
                    continue
                moments = cv2.moments(contour)
                u, v = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
                x, y, width, height = cv2.boundingRect(contour)
                detections.append(Detection(
                    class_id, label, 1.0, (x, y, x + width, y + height), (u, v)
                ))
        return sorted(detections, key=lambda item: (item.class_id, item.pixel_center[0]))

    def _top_surface_detections(self, rgb, depth):
        """Separate adjacent same-color boxes using their measured top faces."""
        candidates = []
        for detection in self.detect_colors(rgb):
            candidates.extend(self._top_surface_centers(detection, rgb, depth))
        # An occluding wrist can split one color contour into several pieces
        # with overlapping bounding rectangles. Refining each rectangle can
        # rediscover the same box top. Keep its most complete observation before
        # tracking, using the same 7.5 cm spatial scale as visual association.
        # Distinct 16 x 10 cm boxes remain farther apart than this radius.
        candidates.sort(key=lambda item: ((item.box_xyxy[2] - item.box_xyxy[0])
                                           * (item.box_xyxy[3] - item.box_xyxy[1])), reverse=True)
        accepted = []
        for detection in candidates:
            try:
                world = self.detection_to_world(detection, depth)
            except ValueError:
                continue
            if any(detection.label == previous.label
                   and np.linalg.norm(np.subtract(world, position)) < .075
                   for previous, position in accepted):
                continue
            accepted.append((detection, world))
        yield from sorted((detection for detection, _ in accepted),
                          key=lambda item: (item.class_id, item.pixel_center[0]))

    def _top_surface_centers(self, detection, rgb, depth):
        """Split color contours whose side faces touch in an oblique image.

        The calibrated belt ROI removes colored platforms before choosing the
        top height. Top faces then form separate image components even when
        two boxes' side faces bridge their original RGB contour.
        """
        x1, y1, x2, y2 = map(int, detection.box_xyxy)
        crop = rgb[y1:y2, x1:x2]
        mask = next(mask for label, _, mask in self._color_masks(crop) if label == detection.label)
        rows, columns = np.where(mask > 0)
        rows, columns = rows + y1, columns + x1
        samples = depth[rows, columns]
        near, far = self.environment.camera.config.near, self.environment.camera.config.far
        valid = np.isfinite(samples) & (samples > near) & (samples < far)
        rows, columns, samples = rows[valid], columns[valid], samples[valid]
        if len(rows) < 10:
            return
        height, width = depth.shape
        points = self._geometry.pixels_to_world(columns, rows, samples, (width, height))
        xmin, xmax, ymin, ymax, zmin, zmax = self.pickup_bounds
        inside = ((points[0] > xmin) & (points[0] < xmax)
                  & (points[1] > ymin) & (points[1] < ymax)
                  & (points[2] > zmin) & (points[2] < zmax))
        rows, columns, points = rows[inside], columns[inside], points[:, inside]
        if len(rows) < 10:
            return
        z = points[2]
        top = z >= np.quantile(z, 0.90) - 0.003
        top_mask = np.zeros(mask.shape, dtype=np.uint8)
        top_mask[rows[top] - y1, columns[top] - x1] = 1
        count, components = cv2.connectedComponents(top_mask, connectivity=8)
        point_components = components[rows - y1, columns - x1]
        for component in range(1, count):
            selected = point_components == component
            if np.count_nonzero(selected) < 10:
                continue
            # The demo's 16 x 10 cm boxes can leave colored slivers behind a
            # wrist. Require enough of each top face to localize its centre;
            # the motion tracker bridges frames rejected for occlusion.
            span = np.ptp(points[:2, selected], axis=1)
            if span[0] < 0.11 or span[1] < 0.06:
                continue
            u, v = float(np.median(columns[selected])), float(np.median(rows[selected]))
            if components[int(round(v)) - y1, int(round(u)) - x1] != component:
                continue
            bounds = (float(columns[selected].min()), float(rows[selected].min()),
                      float(columns[selected].max() + 1), float(rows[selected].max() + 1))
            yield Detection(detection.class_id, detection.label, detection.confidence,
                            bounds, (u, v))

    def detection_to_world(self, detection, depth_image, depth_window=2):
        height, width = depth_image.shape
        return self.pixel_to_world(detection.pixel_center,
                                   _depth_at(detection.pixel_center, depth_image, depth_window),
                                   image_size=(width, height))


class YoloVision(SimColorVision):
    """Simulation YOLO adapter: pretrained model loaded, color boxes emitted.

    COCO does not contain our red/blue box classes. This explicit mock backend
    emits numeric Nx6 rows [x1,y1,x2,y2,conf,class_id], with class 0=Box_A,
    1=Box_B. It never claims these rows came from neural inference. Use
    RealYoloVision (CLI yolo-real) with trained weights for neural detections.
    """

    def __init__(self, environment, conveyor_velocity=(0, 0, 0),
                 model_path="yolov8n.pt", model=None, **noise_options):
        super().__init__(environment, conveyor_velocity, **noise_options)
        if model is None:
            from ultralytics import YOLO
            model = YOLO(model_path)
        self.model = model
        self.last_yolo_output = np.empty((0, 6), dtype=np.float64)

    def mock_yolo_output(self, rgb, depth):
        rows = []
        for candidate in self._top_surface_detections(rgb, depth):
            # Emit a top-face-centred box so its literal box centre is the
            # measured grasp pixel. Avoid the perspective bias of visible sides.
            u, v = candidate.pixel_center
            x1, y1, x2, y2 = candidate.box_xyxy
            rx, ry = min(u-x1, x2-u), min(v-y1, y2-v)
            if rx > 0 and ry > 0:
                rows.append([u-rx, v-ry, u+rx, v+ry,
                             candidate.confidence, candidate.class_id])
        return np.asarray(rows, dtype=np.float64).reshape(-1, 6)

    def detect(self, rgb_image=None, depth_image=None, timestamp=None):
        rgb, depth = _images(self.camera, rgb_image, depth_image)
        rgb = self._rgb_noise.apply(rgb)
        self.last_rgb_image = rgb
        self.last_yolo_output = self.mock_yolo_output(rgb, depth)
        timestamp = float(timestamp if timestamp is not None else
                          self.environment.simulation_step * self.environment.time_step)
        observations = []
        for x1, y1, x2, y2, confidence, class_id in self.last_yolo_output:
            detection = Detection(int(class_id), ("Box_A", "Box_B")[int(class_id)],
                                  float(confidence), (x1, y1, x2, y2))
            try:
                world = self.detection_to_world(detection, depth)
            except ValueError:
                continue
            if _inside(world, self.pickup_bounds):
                observations.append((detection, world))
        return self._tracker.update(observations, timestamp)


class RealYoloVision(BaseVision):
    """YOLO framework: inject real camera SDK and hand-eye calibration adapters.

    Without a converter, pixel_to_world() intentionally raises instead of
    returning invented robot coordinates. Ultralytics/weights load only when
    inference is actually requested, so the color path needs neither of them.
    """

    def __init__(
        self, camera: Optional[RGBDCamera] = None,
        coordinate_converter: Optional[WorldCoordinateConverter] = None,
        model_path="yolov8n.pt", confidence=0.25, class_map=None,
        conveyor_velocity=(0, 0, 0), pickup_bounds=None, model=None,
        *, noise_std=0.0, lighting_jitter=0.0, noise_seed=None,
    ):
        self.camera = camera
        self.coordinate_converter = coordinate_converter
        self.model_path, self.confidence = model_path, confidence
        # Map trained model labels to stable scheduler classes, e.g.
        # {"red_box": "Box_A", "blue_box": "Box_B"}.
        self.class_map = dict(class_map or {})
        self.pickup_bounds = pickup_bounds
        self.model = model
        self._tracker = _WorldTracker(conveyor_velocity)
        self._rgb_noise = _RgbNoise(noise_std, lighting_jitter, noise_seed)
        self.last_rgb_image = None

    @property
    def noise_settings(self):
        return self._rgb_noise.settings

    def get_camera_data(self):
        # TODO: inject a real SDK adapter returning aligned RGB and metric depth.
        return _images(self.camera, None, None)

    def pixel_to_world(self, pixel, depth_m, image_size=None):
        # TODO: real intrinsics + distortion correction + hand-eye calibration.
        if self.coordinate_converter is None:
            raise NotImplementedError("Supply real pixel-to-world / hand-eye calibration")
        return self.coordinate_converter.pixel_to_world(pixel, depth_m, image_size)

    def detect(self, rgb_image=None, depth_image=None, timestamp=None) -> List[VisionResult]:
        rgb_image, depth_image = _images(self.camera, rgb_image, depth_image)
        rgb_image = self._rgb_noise.apply(rgb_image)
        self.last_rgb_image = rgb_image
        if self.coordinate_converter is None:
            raise NotImplementedError("Supply real pixel-to-world / hand-eye calibration")
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        timestamp = float(timestamp if timestamp is not None else monotonic())
        # Ultralytics numpy input expects BGR; the common camera contract is RGB.
        results = self.model.predict(source=cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
                                     conf=self.confidence, verbose=False)
        observations = []
        height, width = depth_image.shape
        for result in results:
            if result.boxes is None:
                continue
            for box, confidence, class_id in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                label = result.names[int(class_id)]
                detection = Detection(int(class_id), self.class_map.get(label, label),
                                      float(confidence), tuple(float(value) for value in box))
                try:
                    depth_m = _depth_at(detection.pixel_center, depth_image)
                    world = self.pixel_to_world(detection.pixel_center, depth_m, (width, height))
                except ValueError:
                    continue  # Invalid/occluded depth is not a grasp candidate.
                if _inside(world, self.pickup_bounds):
                    observations.append((detection, tuple(float(value) for value in world)))
        return self._tracker.update(observations, timestamp)

    def forget_track(self, track_id):
        self._tracker.tracks.pop(track_id, None)


def create_vision(environment, detector="color", model_path="yolov8n.pt",
                  conveyor_velocity=(0, 0, 0), *, noise_std=0.0,
                  lighting_jitter=0.0, noise_seed=None) -> BaseVision:
    """Composition boundary; main/scheduler only consume BaseVision outputs."""
    if detector == "color":
        return SimColorVision(environment, conveyor_velocity=conveyor_velocity,
                              noise_std=noise_std, lighting_jitter=lighting_jitter,
                              noise_seed=noise_seed)
    if detector == "yolo":
        return YoloVision(environment, conveyor_velocity, model_path=model_path,
                          noise_std=noise_std, lighting_jitter=lighting_jitter,
                          noise_seed=noise_seed)
    if detector == "yolo-real":
        # Exercise the real inference framework with SIMULATED camera geometry.
        # On hardware replace BOTH adapters; COCO weights do not know Box_A/B.
        return RealYoloVision(
            camera=environment.camera, coordinate_converter=SimulationCalibration(environment.camera),
            model_path=model_path, conveyor_velocity=conveyor_velocity,
            class_map={"red": "Box_A", "red_box": "Box_A", "blue": "Box_B", "blue_box": "Box_B"},
            pickup_bounds=_simulation_pickup_bounds(environment),
            noise_std=noise_std, lighting_jitter=lighting_jitter, noise_seed=noise_seed,
        )
    raise ValueError("detector must be 'color', 'yolo' or 'yolo-real'")


if __name__ == "__main__":
    from types import SimpleNamespace
    from env_setup import SimulationEnvironment

    def check_contract(items):
        for item in items:
            assert set(item) == set(VisionResult.__required_keys__)
            assert len(item["pos_3d"]) == 3 and np.all(np.isfinite(item["pos_3d"]))
            assert len(item["bbox"]) == 4 and isinstance(item["track_id"], int)
        return items

    env = SimulationEnvironment(gui=False)
    try:
        fixtures = (("Box_A", (-0.11, -0.06), (0.95, 0.08, 0.03, 1.0)),
                    ("Box_B", (0.11, 0.06), (0.03, 0.18, 0.95, 1.0)))
        for _, xy, color in fixtures:
            env.create_test_object((*xy, env.conveyor_top_z + 0.05), shape="box", color=color)
        pipeline: BaseVision = SimColorVision(env)
        env.step()
        rgb, depth = pipeline.get_camera_data()
        objects = check_contract(pipeline.detect(rgb, depth))
        assert {item["class"] for item in objects} == {"Box_A", "Box_B"}
        assert len(objects) == 2, "colored platform was detected as a pickup object"
        for item in objects:
            expected = next(xy for label, xy, _ in fixtures if label == item["class"])
            assert np.linalg.norm(np.subtract(item["pos_3d"][:2], expected)) < 0.018
            assert abs(item["pos_3d"][2] - 0.52) < 0.01
            print(f"{item['class']} visual ID={item['track_id']}, world={item['pos_3d']}")
        assert not pipeline.detect(np.zeros_like(rgb), depth)
        # No-argument acquisition obeys the same contract as a supplied frame.
        again = check_contract(pipeline.detect())
        assert [item["track_id"] for item in again] == [item["track_id"] for item in objects]
        print("PASS: uniform color output, depth unprojection, platform rejection and stable IDs")

        # Hold one REAL camera exposure fixed to isolate image-noise error from
        # physics, motion and occlusion. Reference points are measured by the
        # clean vision pipeline, never queried from simulated object poses.
        clean_rgb, clean_depth = rgb.copy(), depth.copy()
        reference = {item["class"]: np.asarray(item["pos_3d"]) for item in objects}
        noisy = SimColorVision(env, noise_std=12.0, lighting_jitter=0.04, noise_seed=2026)
        duplicate = SimColorVision(env, noise_std=12.0, lighting_jitter=0.04, noise_seed=2026)
        duplicate.detect(rgb, depth, timestamp=0)
        errors, matched, false_positives, track_ids = [], 0, 0, {}
        exposures = 80
        for exposure in range(exposures):
            measured = check_contract(noisy.detect(rgb, depth, timestamp=exposure / 20))
            if exposure == 0:
                assert np.array_equal(noisy.last_rgb_image, duplicate.last_rgb_image), (
                    "the same noise seed must reproduce the image"
                )
                assert not np.array_equal(rgb, noisy.last_rgb_image), "noise was not applied"
            seen = set()
            for item in measured:
                label = item["class"]
                if label not in reference or label in seen:
                    false_positives += 1
                    continue
                seen.add(label)
                error = float(np.linalg.norm(np.asarray(item["pos_3d"]) - reference[label]))
                errors.append(error)
                matched += 1
                track_ids.setdefault(label, set()).add(item["track_id"])
        assert np.array_equal(rgb, clean_rgb) and np.array_equal(depth, clean_depth), (
            "noise injection modified the caller's synchronized RGB-D exposure"
        )
        recall = matched / (exposures * len(reference))
        rms = float(np.sqrt(np.mean(np.square(errors))))
        maximum = max(errors)
        print(f"NOISE BENCHMARK: {exposures} fixed RGB-D exposures, RGB sigma=12/255, "
              f"lighting=+/-4%, seed=2026; recall={recall:.1%}, false positives={false_positives}, "
              f"3D error vs clean image RMS={rms * 1000:.3f} mm, max={maximum * 1000:.3f} mm")
        assert recall >= 0.99 and false_positives == 0, "noise destabilized color detection"
        assert maximum < 0.005, "RGB noise exceeded the 5 mm positioning tolerance"
        assert all(len(ids) == 1 for ids in track_ids.values()), "noise split a visual identity"
        print("PASS: seeded noise, unchanged RGB-D input, robust 3D localization and track IDs")
    finally:
        env.close()

    # Check the perturbation itself as well as its downstream effects. A flat
    # mid-gray image avoids clipping so measured sigma should match the input.
    flat = np.full((256, 256, 3), 128, np.uint8)
    perturbed = _RgbNoise(noise_std=12, noise_seed=7).apply(flat)
    residual = perturbed.astype(np.float32) - flat
    assert abs(float(residual.mean())) < 0.15 and abs(float(residual.std()) - 12) < 0.15
    assert np.array_equal(_RgbNoise().apply(flat), flat), "disabled noise changed RGB"
    for options in ({"noise_std": -1}, {"noise_std": float("nan")},
                    {"lighting_jitter": 0.51}, {"lighting_jitter": float("inf")}):
        try:
            _RgbNoise(**options)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid noise settings accepted")

    moving = SimulationEnvironment(gui=False, conveyor_speed=0.05)
    try:
        moving.enable_infeed(seed=7, limit=1)
        tracker = SimColorVision(moving, conveyor_velocity=(0.05, 0, 0))
        moving.step(120)
        first, = tracker.detect()
        moving.step(480)
        rgb, depth = tracker.get_camera_data()
        assert not tracker.detect(np.zeros_like(rgb), depth)
        moving.step(240)
        last, = tracker.detect()
        assert last["track_id"] == first["track_id"], "moving target lost its ID through occlusion"
        displacement = last["pos_3d"][0] - first["pos_3d"][0]
        assert abs(displacement - 0.05 * 3.0) < 0.015
        tracker.forget_track(last["track_id"])
        assert not tracker._tracker.tracks
        print(f"PASS: moving identity across 3s occlusion; displacement={displacement:.4f}m")
    finally:
        moving.close()

    # Exercise the real framework without hardware, weights or network access.
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)
        def cpu(self):
            return self
        def numpy(self):
            return self.values

    class FakeYolo:
        def predict(self, source, **options):
            assert tuple(source[0, 0]) == (30, 20, 10), "YOLO must receive BGR"
            boxes = SimpleNamespace(xyxy=Tensor([[1, 1, 5, 5]]),
                                    conf=Tensor([0.9]), cls=Tensor([0]))
            return [SimpleNamespace(boxes=boxes, names={0: "red_box"})]

    class FakeCamera:
        def __init__(self):
            self.calls = 0
        def capture_rgbd(self):
            self.calls += 1
            return np.full((8, 8, 3), (10, 20, 30), np.uint8), np.full((8, 8), 0.5)

    class FakeCalibration:
        def pixel_to_world(self, pixel, depth_m, image_size=None):
            assert pixel == (3, 3) and image_size == (8, 8)
            return (0.1, 0.2, depth_m)

    camera = FakeCamera()
    real: BaseVision = RealYoloVision(camera, FakeCalibration(), model=FakeYolo(),
                                     class_map={"red_box": "Box_A"})
    item, = check_contract(real.detect(timestamp=0))
    assert item["class"] == "Box_A" and item["pos_3d"] == (0.1, 0.2, 0.5)
    rgb, depth = camera.capture_rgbd()
    captures = camera.calls
    again, = real.detect(rgb, depth, timestamp=1)
    assert camera.calls == captures, "supplied frame caused an extra camera exposure"
    assert again["track_id"] == item["track_id"]
    assert not real.detect(rgb, np.zeros_like(depth), timestamp=2)
    real.forget_track(item["track_id"])
    for args in ((rgb, None), (rgb, depth[:4])):
        try:
            real.detect(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid RGB-D pair was accepted")
    try:
        RealYoloVision().pixel_to_world((1, 1), 0.5)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("missing real calibration returned fake coordinates")
    print("PASS: YOLO framework contract, camera/calibration injection, invalid depth and explicit placeholder")
