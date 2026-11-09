"""Isolated, reproducible camera stress fixtures for the visual comparison demo.

This module never runs a scheduler or changes the production sorting scene.
``capture`` returns one common RGB-D exposure for both image-based detectors.
``truth`` is a separate *simulation evaluation only* API: its segmentation
information must not be supplied to a detector advertised as real inference.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pybullet as p

from env_setup import CameraConfig, CameraFrame, SimulationEnvironment


class VisualStressScene:
    """Close-up belt fixtures with dense touching/stacked cubes and occluders.

    Scenarios ``dim`` and ``glare`` use the crowded geometry; their photometric
    perturbation belongs to the comparison runner so both backends see the
    exact same altered pixels. ``occluded`` adds real opaque overhead bars.
    All fixtures have nonintersecting collision geometry. Their fixed poses
    represent settled boxes, including two correctly supported stacked boxes.
    """

    SCENARIOS = ("clean", "crowded", "occluded", "dim", "glare")
    MIN_VISIBLE_PIXELS = 20
    MIN_VISIBLE_FRACTION = 0.15

    def __init__(self, seed: int = 7, width: int = 800, height: int = 520):
        self.seed = int(seed)
        self.width, self.height = int(width), int(height)
        self._rng = np.random.default_rng(self.seed)
        camera = CameraConfig(
            width=self.width, height=self.height, fov_deg=48.0,
            near=0.05, far=8.0, target=(0.0, 0.0, 0.45), distance=1.05,
            yaw_deg=24.0, pitch_deg=-59.0, renderer="tiny",
        )
        self.env = SimulationEnvironment(gui=False, camera_config=camera)
        # Only this independent DIRECT client's assets are removed. Keeping
        # the physical belt and floor gives a clear close-up without a robot
        # obscuring the comparison. Production env_setup.py is unchanged.
        retained = {self.env.plane_id, self.env.conveyor_id}
        bodies = [p.getBodyUniqueId(i, physicsClientId=self.env.client_id)
                  for i in range(p.getNumBodies(physicsClientId=self.env.client_id))]
        for body in bodies:
            if body not in retained:
                p.removeBody(body, physicsClientId=self.env.client_id)
        self.fixtures: Dict[int, str] = {}
        self.occluder_ids: List[int] = []
        self.distractor_ids: List[int] = []
        self._scenario = None
        self._cached_frame = None
        self._reference_pixels: Dict[int, int] = {}
        self._reference_bboxes: Dict[int, tuple] = {}
        self._colors = {}
        self._closed = False
        self._configure("clean")

    def _box(self, centre, half, color, *, collision=True):
        client = self.env.client_id
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                     rgbaColor=color, physicsClientId=client)
        shape = (p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                        physicsClientId=client) if collision else -1)
        body = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=shape, baseVisualShapeIndex=visual,
            basePosition=centre, baseOrientation=(0, 0, 0, 1),
            physicsClientId=client,
        )
        self._colors[body] = color
        return body

    def _configure(self, scenario):
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown visual test scenario {scenario!r}; use {self.SCENARIOS}")
        if self._scenario == scenario:
            return
        self._cached_frame = None
        client = self.env.client_id
        if self._scenario is not None and self._scenario != "clean" and scenario != "clean":
            # Keep the SAME object identities and geometry through lighting
            # and occlusion stages. Only the overhead bars enter/leave view.
            if scenario != "occluded":
                for body in self.occluder_ids:
                    p.removeBody(body, physicsClientId=client)
                    self._colors.pop(body, None)
                self.occluder_ids.clear()
            elif not self.occluder_ids:
                self._add_occluders()
            self._scenario = scenario
            return
        for body in list(self.fixtures) + self.occluder_ids + self.distractor_ids:
            p.removeBody(body, physicsClientId=client)
        self.fixtures.clear()
        self.occluder_ids.clear()
        self.distractor_ids.clear()
        self._colors.clear()
        self._reference_pixels.clear()
        self._reference_bboxes.clear()
        self._scenario = scenario
        belt = self.env.conveyor_top_z
        half = (0.050, 0.050, 0.050)
        if scenario == "clean":
            # Eight separated cubes, equal class counts, no imposed overlap.
            items = [(x, y, 0, "Box_A" if (i + j) % 2 == 0 else "Box_B")
                     for j, y in enumerate((-0.115, 0.115))
                     for i, x in enumerate((-0.36, -0.12, 0.12, 0.36))]
        else:
            # Ten touching boxes form two rows. Same-colour neighbours have
            # genuinely adjoining image regions; two further boxes sit ON
            # lower boxes, rather than interpenetrating at creation time.
            row_labels = (("Box_A", "Box_A", "Box_B", "Box_B", "Box_A"),
                          ("Box_B", "Box_B", "Box_A", "Box_A", "Box_B"))
            items = [(x, y, 0, row_labels[j][i])
                     for j, y in enumerate((-0.050, 0.050))
                     for i, x in enumerate((-0.20, -0.10, 0.0, 0.10, 0.20))]
            items += [(-0.10, 0.050, 1, "Box_A"),
                      (0.10, -0.050, 1, "Box_B")]
        # Seed changes which class starts the scene, while preserving balance.
        swap = bool(self.seed % 2 == 0)
        for x, y, layer, label in items:
            if swap:
                label = "Box_B" if label == "Box_A" else "Box_A"
            color = ((0.95, 0.075, 0.035, 1) if label == "Box_A"
                     else (0.035, 0.18, 0.95, 1))
            body = self._box((x, y, belt + 0.05 + layer * 0.10), half, color)
            self.fixtures[body] = label
        # Cache isolated silhouettes before adding occluders. This lets the
        # evaluator distinguish tiny/fully hidden surfaces from visible ones.
        self._build_visibility_reference()
        if scenario == "occluded":
            self._add_occluders()

    def _add_occluders(self):
        belt = self.env.conveyor_top_z
        for x, y, z, extents in (
                (-0.16, -0.035, belt + 0.245, (0.022, 0.18, 0.014)),
                (0.155, 0.015, belt + 0.245, (0.027, 0.18, 0.014))):
            self.occluder_ids.append(self._box(
                (x, y, z), extents, (0.46, 0.48, 0.50, 1)))

    def _build_visibility_reference(self):
        """Render one silhouette at a time for evaluation visibility ratios.

        Bodies are moved out of view only during this setup operation; they
        are restored before any detector sees a camera exposure. Doing this
        once per scene avoids assuming that a touching/stacked object's
        currently visible fragment represents its full projected silhouette.
        """
        client = self.env.client_id
        poses = {body: p.getBasePositionAndOrientation(body, physicsClientId=client)
                 for body in self.fixtures}
        for body in self.fixtures:
            p.resetBasePositionAndOrientation(body, (0, 0, -20), (0, 0, 0, 1),
                                              physicsClientId=client)
        for body, (position, orientation) in poses.items():
            p.resetBasePositionAndOrientation(body, position, orientation,
                                              physicsClientId=client)
            frame = self.env.camera.capture_frame()
            ys, xs = np.where(self._body_mask(frame.segmentation) == body)
            self._reference_pixels[body] = int(xs.size)
            if xs.size:
                self._reference_bboxes[body] = (int(xs.min()), int(ys.min()),
                                               int(xs.max()) + 1, int(ys.max()) + 1)
            p.resetBasePositionAndOrientation(body, (0, 0, -20), (0, 0, 0, 1),
                                              physicsClientId=client)
        for body, (position, orientation) in poses.items():
            p.resetBasePositionAndOrientation(body, position, orientation,
                                              physicsClientId=client)

    @staticmethod
    def _body_mask(segmentation):
        """Decode Bullet's body + ((link+1) << 24) segmentation convention."""
        segmentation = np.asarray(segmentation, dtype=np.int64)
        return np.where(segmentation < 0, -1, segmentation & ((1 << 24) - 1))

    def capture(self, timestamp: float = 0.0, scenario: str = "clean") -> CameraFrame:
        """One unaltered RGB-D frame; timestamp is accepted for runner parity.

        Geometry is intentionally stationary, so the raw camera buffers are
        cached read-only until the scenario changes. The runner can apply a
        changing exposure/noise field to this same exposure for both backends
        without rerendering the identical geometry every video frame.
        """
        if self._closed:
            raise RuntimeError("VisualStressScene has been closed")
        if not math.isfinite(float(timestamp)):
            raise ValueError("timestamp must be finite")
        self._configure(scenario)
        if self._cached_frame is None:
            self._cached_frame = self.env.camera.capture_frame()
            for buffer in (self._cached_frame.rgb, self._cached_frame.depth,
                           self._cached_frame.segmentation):
                buffer.setflags(write=False)
        return self._cached_frame

    def truth(self, frame: CameraFrame):
        """Simulation-only visible-object annotations; NEVER detector input.

        Returns every fixture, including fully occluded ones (bbox=None,
        eligible=False). ``bbox`` covers only the visible segmentation and
        uses exclusive x2/y2. Eligibility requires >=20 visible pixels and
        >=15% of an isolated silhouette. Opaque gray bars are not objects.
        """
        body_mask = self._body_mask(frame.segmentation)
        result = []
        for body, label in self.fixtures.items():
            ys, xs = np.where(body_mask == body)
            pixels = int(xs.size)
            fraction = min(1.0, pixels / max(1, self._reference_pixels.get(body, 0)))
            bbox = ((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                    if pixels else None)
            result.append({
                "body_id": body, "label": label,
                "class_id": 0 if label == "Box_A" else 1,
                "bbox": bbox, "visible_pixels": pixels,
                "visible_fraction": fraction,
                "eligible": bool(pixels >= self.MIN_VISIBLE_PIXELS
                                 and fraction >= self.MIN_VISIBLE_FRACTION),
            })
        return result

    def close(self):
        if not self._closed:
            self.env.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
