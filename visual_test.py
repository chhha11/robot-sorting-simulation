"""Same-frame detector stress demo; ideal simulated boxes are never model scores.

python main.py --visual-test
python visual_test.py --headless --visual-duration 10 --save-dir outputs/visual_test
python main.py --visual-test --visual-yolo real --model runs/detect/train/weights/best.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from vision import SimColorVision
from visual_test_metrics import score_detections


SCENARIOS = ("clean", "crowded", "occluded", "dim", "glare")
SCENARIO_NAMES = {
    "clean": "CLEAN / separated boxes",
    "crowded": "CROWDED / touching + stacked boxes",
    "occluded": "OCCLUDED / gray blockers + crowding",
    "dim": "DIM / low exposure + sensor noise",
    "glare": "GLARE / overexposure + washed-out color",
}
WINDOW_NAME = "Vision Stress Test | Color Threshold vs YOLO"


def add_visual_arguments(parser):
    parser.add_argument("--visual-test", action="store_true", help="same-frame split-screen vision stress demo")
    parser.add_argument("--visual-yolo", choices=("mock", "real"), default="mock",
                        help="mock: explicitly labeled ideal simulation overlay; real: trained red/blue model")
    parser.add_argument("--visual-scenario", choices=("cycle", *SCENARIOS), default="cycle")
    parser.add_argument("--visual-duration", type=float, default=0,
                        help="seconds of recorded scenario time; 0 loops in GUI / 30s headless")
    parser.add_argument("--visual-period", type=float, default=5, help="seconds per scenario in auto cycle")
    parser.add_argument("--visual-fps", type=int, default=15, help="display/video frames per second")
    parser.add_argument("--visual-record", metavar="FILE.mp4", help="optional split-screen video output")
    parser.add_argument("--visual-width", type=int, default=800, help="camera width per panel")
    parser.add_argument("--visual-height", type=int, default=520, help="camera height per panel")
    parser.add_argument("--visual-iou", type=float, default=.5, help="same-class IoU threshold for one-to-one scoring")
    parser.add_argument("--visual-confidence", type=float, default=.25, help="real YOLO confidence threshold")


def validate_options(args):
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if not np.isfinite(args.visual_duration) or args.visual_duration < 0:
        raise ValueError("--visual-duration must be finite and nonnegative")
    if not np.isfinite(args.visual_period) or args.visual_period <= 0:
        raise ValueError("--visual-period must be finite and positive")
    if not 1 <= args.visual_fps <= 60:
        raise ValueError("--visual-fps must be between 1 and 60")
    if not 480 <= args.visual_width <= 1600 or not 320 <= args.visual_height <= 1000:
        raise ValueError("visual resolution must be 480..1600 wide and 320..1000 high per panel")
    if not 0 < args.visual_iou <= 1 or not 0 < args.visual_confidence <= 1:
        raise ValueError("IoU and confidence must be in (0, 1]")
    if args.visual_record and Path(args.visual_record).suffix.lower() != ".mp4":
        raise ValueError("--visual-record must end in .mp4")
    if args.visual_record and args.visual_height % 2:
        raise ValueError("--visual-height must be even when recording MP4")


def perturb_rgb(rgb, scenario, timestamp, seed=7, frame_index=0):
    """Apply a single deterministic exposure to BOTH detector inputs.

    Physics geometry/occluders are in the rendered frame. Exposure, white veil
    and sensor noise below model common camera degradations, not a second
    clean image secretly supplied to the right-hand detector.
    """
    values = rgb.astype(np.float32)
    rng = np.random.default_rng(np.random.SeedSequence([seed, frame_index]))
    settings = dict(exposure=1.0, noise_std=0.0, white_veil=0.0)
    if scenario == "dim":
        exposure = .16 + .045 * np.sin(timestamp * 1.7)
        values *= exposure
        values += rng.normal(0, 3.0, values.shape).astype(np.float32)
        settings.update(exposure=float(exposure), noise_std=3.0)
    elif scenario == "glare":
        height, width = rgb.shape[:2]
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        center = width * (.50 + .12 * np.sin(timestamp * 1.2))
        veil = .75 * np.exp(-((xx-center)**2/(.48*width)**2
                              + (yy-.52*height)**2/(.65*height)**2))
        values = values * (1-veil[..., None]) + 255*veil[..., None]
        settings.update(white_veil=.75)
    return np.clip(np.rint(values), 0, 255).astype(np.uint8), settings


def truth_rows(items):
    return np.asarray([(*item["bbox"], 1.0, item["class_id"])
                       for item in items if item["eligible"]], dtype=float).reshape(-1, 6)


class ColorThresholdDetector:
    """Call the project's original RGB HSV+contour baseline unchanged."""

    def __init__(self, environment):
        self.backend = SimColorVision(environment)

    def predict(self, rgb):
        detections = self.backend.detect_colors(rgb)
        return np.asarray([(*item.box_xyxy, item.confidence, item.class_id)
                           for item in detections], dtype=float).reshape(-1, 6)


class RealYoloDetector:
    """Neural RGB-only inference, with explicit red/blue class names.

    Receives no simulator body IDs, segmentation, depth or ground truth.
    The generic COCO weights cannot provide the requested class comparison.
    """

    def __init__(self, weights, confidence=.25):
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"YOLO weights not found: {path}")
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        aliases = {"box_a": 0, "red": 0, "red_box": 0, "red block": 0, "red_block": 0,
                   "box_b": 1, "blue": 1, "blue_box": 1, "blue block": 1, "blue_block": 1}
        names = self.model.names
        pairs = names.items() if isinstance(names, dict) else enumerate(names)
        self.class_map = {int(index): aliases[str(name).strip().lower()] for index, name in pairs
                          if str(name).strip().lower() in aliases}
        if set(self.class_map.values()) != {0, 1}:
            raise ValueError("This model has no red/blue box classes. COCO yolov8n.pt cannot benchmark "
                             "these blocks. Use trained Box_A/Box_B (or red/blue) weights, "
                             "or --visual-yolo mock for the labeled simulation demo.")
        self.confidence = confidence
        self.metadata = dict(weights=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             classes={str(key): value for key, value in self.class_map.items()})

    def predict(self, rgb):
        results = self.model.predict(source=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                     conf=self.confidence, verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for row in result.boxes.data.cpu().numpy():
                class_id = self.class_map.get(int(row[5]))
                if class_id is not None:
                    detections.append((*map(float, row[:5]), class_id))
        return np.asarray(detections, dtype=float).reshape(-1, 6)


def _text(image, text, point, color=(220, 226, 235), scale=.52, thickness=1):
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_boxes(image, detections):
    for x1, y1, x2, y2, confidence, class_id in detections:
        color = (70, 95, 255) if int(class_id) == 0 else (255, 170, 65)
        left, top = int(x1), int(y1)
        cv2.rectangle(image, (left, top), (int(x2), int(y2)), color, 2)
        label = f"{'Red' if int(class_id) == 0 else 'Blue'} {confidence:.2f}"
        _text(image, label, (left, max(16, top-5)), color, scale=.43)


def compose_panel(rgb, left_boxes, right_boxes, truth_count, left_score, right_score,
                  scenario, timestamp, mode, exposure, elapsed_ms):
    height, width = rgb.shape[:2]
    header, footer, gap = 156, 96, 6
    canvas = np.full((height+header+footer, width*2+gap, 3), (20, 24, 32), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    left, right = bgr.copy(), bgr.copy()
    _draw_boxes(left, left_boxes)
    _draw_boxes(right, right_boxes)
    canvas[header:header+height, :width] = left
    canvas[header:header+height, width+gap:] = right
    _text(canvas, f"STRESS LAB   {SCENARIO_NAMES[scenario]}   t={timestamp:.1f}s", (20, 28), scale=.65, thickness=2)
    for offset, title, boxes in ((0, "Color Threshold", left_boxes), (width+gap, "YOLO", right_boxes)):
        _text(canvas, f"{title}: Found {len(boxes)} objects", (offset+18, 65),
              (240, 240, 240), scale=.76, thickness=2)
    _text(canvas, "Original OpenCV HSV + contours", (18, 91))
    if mode == "mock":
        cv2.rectangle(canvas, (width+gap+8, 76), (width*2+gap-8, 109), (35, 55, 110), -1)
        _text(canvas, "SIMULATED / IDEAL BOXES - NOT NEURAL INFERENCE", (width+gap+18, 98),
              (100, 220, 255), scale=.52, thickness=2)
        _text(canvas, "Model accuracy: N/A (ground-truth overlay)", (width+gap+18, 130), (100, 220, 255))
    else:
        _text(canvas, f"Real trained YOLO | inference {elapsed_ms:.1f} ms", (width+gap+18, 91))
        _text(canvas, f"Frame P {right_score['precision']:.0%} | R {right_score['recall']:.0%}"
              f" | FP {right_score['fp']} | FN {right_score['fn']}", (width+gap+18, 130))
    _text(canvas, f"Frame P {left_score['precision']:.0%} | R {left_score['recall']:.0%}"
          f" | FP {left_score['fp']} | FN {left_score['fn']}", (18, 130))
    bottom = header+height
    _text(canvas, f"Same degraded RGB on both sides | eligible visible boxes: {truth_count}"
          f" | exposure: {exposure['exposure']:.2f}", (18, bottom+24), scale=.54)
    disclaimer = ("DEMO ONLY: right boxes use simulation annotations; this is NOT a YOLO accuracy benchmark."
                  if mode == "mock" else "Frame detection metrics: class-aware one-to-one IoU matching; no simulator input to either detector.")
    _text(canvas, disclaimer, (18, bottom+49), (100, 220, 255) if mode == "mock" else (170, 220, 170), scale=.52)
    _text(canvas, "1 Clean   2 Crowded   3 Occluded   4 Dim   5 Glare   A Auto   SPACE Pause   S Screenshot   Q/ESC Exit",
          (18, bottom+76), scale=.51)
    return canvas


def run_visual_test(args):
    validate_options(args)
    # Reject unsuitable real weights before constructing any simulation/UI.
    neural = (RealYoloDetector(args.model, args.visual_confidence)
              if args.visual_yolo == "real" else None)
    from visual_test_scene import VisualStressScene
    output = Path(args.save_dir or "outputs/visual_test").resolve()
    output.mkdir(parents=True, exist_ok=True)
    duration = args.visual_duration or (30 if args.headless else 0)
    max_frames = max(1, round(duration*args.visual_fps)) if duration else None
    report = dict(created_at=datetime.now().isoformat(timespec="seconds"), mode=args.visual_yolo,
                  right_is_neural_inference=neural is not None,
                  mock_uses_simulation_annotations=neural is None,
                  model=neural.metadata if neural is not None else None,
                  seed=args.seed, fps=args.visual_fps, iou_threshold=args.visual_iou,
                  baseline="Unmodified SimColorVision.detect_colors: HSV thresholds + contours (RGB only)",
                  inputs="Identical perturbed RGB bytes for both real detectors. Simulator labels only evaluate/ideal mock.",
                  metric="Frame-wise counts, class-aware one-to-one IoU; not unique-object sorting success.",
                  ground_truth="Visible segmentation boxes, >=20 pixels and >=15% of same-view reference visibility; fully hidden excluded.",
                  annotation_caveat="Visible boxes can differ from full-object training boxes under occlusion; predictions on ineligible fragments still count as false positives.",
                  frames=[], per_scenario={})
    totals = defaultdict(lambda: {"frames": 0, "truth_instances": 0, "color_tp": 0, "color_fp": 0,
                                  "color_fn": 0, "yolo_tp": 0, "yolo_fp": 0, "yolo_fn": 0})
    scene, video = None, None
    selected, paused, frame_index = args.visual_scenario, False, 0
    snapshots, last_panel = set(), None
    print(f"[VisualTest] mode={args.visual_yolo}; same-frame comparison; output={output}", flush=True)
    if neural is None:
        print("[VisualTest] SIMULATED ideal boxes come from scene annotations, not YOLO inference. "
              "No YOLO model accuracy is reported.", flush=True)
    try:
        scene = VisualStressScene(seed=args.seed, width=args.visual_width, height=args.visual_height)
        color = ColorThresholdDetector(scene.env)
        if not args.headless:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            # Leave room for Windows DPI scaling; the saved video stays full size.
            window_width = min(args.visual_width*2+6, 1100)
            window_height = round(window_width*(args.visual_height+252)/(args.visual_width*2+6))
            cv2.resizeWindow(WINDOW_NAME, window_width, window_height)
        while max_frames is None or frame_index < max_frames:
            started = time.perf_counter()
            timestamp = frame_index/args.visual_fps
            scenario = (SCENARIOS[int(timestamp/args.visual_period) % len(SCENARIOS)]
                        if selected == "cycle" else selected)
            frame = scene.capture(timestamp, scenario)
            truths = scene.truth(frame)
            expected = truth_rows(truths)
            rgb, exposure = perturb_rgb(frame.rgb, scenario, timestamp, args.seed, frame_index)
            left_boxes = color.predict(rgb)
            infer_start = time.perf_counter()
            # The ideal mock is deliberately isolated from the neural API.
            right_boxes = expected.copy() if neural is None else neural.predict(rgb)
            elapsed_ms = (time.perf_counter()-infer_start)*1000
            left_score = score_detections(left_boxes, expected, args.visual_iou)
            right_score = (score_detections(right_boxes, expected, args.visual_iou)
                           if neural is not None else None)
            panel = compose_panel(rgb, left_boxes, right_boxes, len(expected), left_score,
                                  right_score, scenario, timestamp, args.visual_yolo, exposure, elapsed_ms)
            last_panel = panel
            if scenario not in snapshots:
                cv2.imwrite(str(output/f"{scenario}.png"), panel)
                snapshots.add(scenario)
                print(f"[VisualTest] {scenario}: Color Found {len(left_boxes)}, "
                      f"{'YOLO' if neural else 'MOCK'} Found {len(right_boxes)}, "
                      f"visible={len(expected)}, FP={left_score['fp']}, FN={left_score['fn']}", flush=True)
            if args.visual_record:
                if video is None:
                    destination = Path(args.visual_record).resolve()
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    video = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                                            args.visual_fps, (panel.shape[1], panel.shape[0]))
                    if not video.isOpened():
                        raise RuntimeError(f"Cannot create video: {destination}")
                video.write(panel)
            entry = dict(frame=frame_index, time=timestamp, scenario=scenario, visible_truth=len(expected),
                         spawned=len(truths), color_count=len(left_boxes), right_count=len(right_boxes),
                         color_metrics=left_score, yolo_metrics=right_score, exposure=exposure)
            report["frames"].append(entry)
            row = totals[scenario]
            row["frames"] += 1
            row["truth_instances"] += len(expected)
            for name, score in (("color", left_score), ("yolo", right_score)):
                if score is not None:
                    for field in ("tp", "fp", "fn"):
                        row[f"{name}_{field}"] += score[field]
            if not args.headless:
                cv2.imshow(WINDOW_NAME, panel)
                key = cv2.waitKey(max(1, round((1/args.visual_fps-(time.perf_counter()-started))*1000))) & 255
                if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("s"):
                    cv2.imwrite(str(output/f"capture_{frame_index:06d}.png"), panel)
                if key == ord(" "):
                    paused = True
                while paused:
                    key = cv2.waitKey(50) & 255
                    if key == ord("s"):
                        cv2.imwrite(str(output/f"capture_{frame_index:06d}.png"), panel)
                    if key in (ord(" "), ord("q"), 27):
                        paused = False
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        key = ord("q")
                        paused = False
                if key in (ord("q"), 27):
                    break
                if ord("1") <= key <= ord("5"):
                    selected = SCENARIOS[key-ord("1")]
                elif key == ord("a"):
                    selected = "cycle"
            frame_index += 1
    finally:
        if video is not None:
            video.release()
        if last_panel is not None:
            cv2.imwrite(str(output/"last_frame.png"), last_panel)
        if scene is not None:
            scene.close()
        if not args.headless:
            cv2.destroyAllWindows()
        for scenario, row in totals.items():
            result = dict(frames=row["frames"], truth_instances=row["truth_instances"])
            for name in ("color", "yolo"):
                if name == "yolo" and neural is None:
                    result[name] = None
                    continue
                tp, fp, fn = (row[f"{name}_{field}"] for field in ("tp", "fp", "fn"))
                result[name] = dict(tp=tp, fp=fp, fn=fn, precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn))
            report["per_scenario"][scenario] = result
        (output/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[VisualTest] finished {len(report['frames'])} frames; metrics={output/'metrics.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_visual_arguments(parser)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--save-dir")
    args = parser.parse_args()
    try:
        run_visual_test(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
