"""Continuous visual sorting: camera -> motion prediction -> two arm FSMs.

GUI:      python main.py
Robot backend: python main.py --robot sim (default); --robot real is a driver scaffold
Visual comparison: python main.py --visual-test
Self-test: python main.py --headless --objects 6
Static regression: python main.py --static --headless
Single-handoff regression: python main.py --static --headless --test-fault grasp
GUI fault panel: four sliders; raise to 1, return to 0 to rearm.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import cv2
import numpy as np

from robot_arm import BaseRobotArm, create_robot_arm
from scheduler import ArmRoute, Scheduler, ConveyorScheduler, TaskStatus
from vision_base import BaseVision
from vision import create_vision


def load_simulation_dependencies():
    """Import Bullet-specific scene/control helpers only on the sim path.

    Importing the application or selecting the real driver scaffold does not
    require PyBullet. Existing simulation helper functions remain importable.
    """
    global p, CameraConfig, SimulationEnvironment, SharedWorkspaceGate, FaultInjectionPanel
    import pybullet as p
    from env_setup import CameraConfig, SimulationEnvironment
    from robot_arm import SharedWorkspaceGate
    from fault_injection import FaultInjectionPanel


def require_robot_backend(robot):
    """Fail before scene creation when a selected hardware driver is unready."""
    if robot == "real":
        driver = create_robot_arm("real", display_name="Arm 1")
        driver.ensure_ready()
    elif robot != "sim":
        raise ValueError(f"Unknown robot backend: {robot!r}")


class ParallelMotionMonitor:
    """Measure physical motion, rather than counting assigned/waiting FSMs.

    Bullet advances both robots once in env.step(). Sampling the measured TCPs
    after that shared step therefore compares motion over exactly the same
    simulation interval; controller calls themselves never advance physics.
    """

    def __init__(self, arms, time_step, minimum_speed=.005):
        self.arms = arms
        self.time_step = float(time_step)
        self.minimum_speed = float(minimum_speed)
        self.previous = [np.asarray(arm.get_measured_tcp_position()) for arm in arms]
        self.active_steps = 0
        self.longest_run_steps = 0
        self.current_run_steps = 0
        self.intervals = []
        self._start_time = None
        self.central_zone_violation_steps = 0
        self.central_zone_max_occupancy = 0

    def sample(self, step):
        positions = [np.asarray(arm.get_measured_tcp_position()) for arm in self.arms]
        speeds = [float(np.linalg.norm(point - previous) / self.time_step)
                  for point, previous in zip(positions, self.previous)]
        self.previous = positions
        gate = getattr(self.arms[0], "workspace_gate", None) if self.arms else None
        if gate is not None and hasattr(gate, "in_spatial_zone"):
            occupancy = sum(bool(gate.in_spatial_zone(point)) for point in positions)
            self.central_zone_max_occupancy = max(self.central_zone_max_occupancy, occupancy)
            if occupancy > 1:
                self.central_zone_violation_steps += 1
        moving = [not arm.is_idle() and not arm_wait_reason(arm)
                  and speed > self.minimum_speed
                  for arm, speed in zip(self.arms, speeds)]
        parallel = len(moving) > 1 and all(moving)
        time = step * self.time_step
        if parallel:
            self.active_steps += 1
            self.current_run_steps += 1
            self.longest_run_steps = max(self.longest_run_steps, self.current_run_steps)
            if self._start_time is None:
                self._start_time = time - self.time_step
        elif self._start_time is not None:
            self.intervals.append((self._start_time, time - self.time_step))
            self._start_time = None
            self.current_run_steps = 0
        return parallel

    def summary(self):
        intervals = list(self.intervals)
        if self._start_time is not None:
            intervals.append((self._start_time,
                              self._start_time + self.current_run_steps * self.time_step))
        gate = getattr(self.arms[0], "workspace_gate", None) if self.arms else None
        clearance = getattr(gate, "minimum_observed_clearance", math.inf)
        traffic_events = getattr(gate, "traffic_events", ())
        return dict(actual_parallel_seconds=self.active_steps * self.time_step,
                    longest_parallel_seconds=self.longest_run_steps * self.time_step,
                    minimum_tcp_speed_m_s=self.minimum_speed,
                    minimum_observed_clearance_m=clearance if math.isfinite(clearance) else None,
                    geometry_wait_events=len(getattr(gate, "wait_events", ())),
                    traffic_event_counts=dict(Counter(event["event"] for event in traffic_events)),
                    central_zone_violation_steps=self.central_zone_violation_steps,
                    central_zone_max_occupancy=self.central_zone_max_occupancy,
                    central_zone=dict(center=getattr(gate, "center", None),
                                      radius_m=getattr(gate, "radius", None),
                                      ceiling_m=getattr(gate, "ceiling", None)),
                    collision_count_by_arm={f"arm{i+1}": int(getattr(arm, "collision_count", 0))
                                            for i, arm in enumerate(self.arms)},
                    intervals=intervals)


def arm_wait_reason(arm):
    if getattr(arm, "traffic_yielding", False):
        return "priority_yield"
    if getattr(arm, "_workspace_waiting", False):
        return "central_token"
    if getattr(arm, "_proximity_waiting", False):
        return "physical_clearance"
    if getattr(arm, "_collision_waiting", False):
        return "collision_recovery"
    return ""


def arm_activity_label(arm):
    """Expose traffic holds/retreats separately from the preserved task FSM."""
    if getattr(arm, "traffic_yielding", False):
        return "YIELDING"
    if arm_wait_reason(arm):
        return "WAITING"
    if arm.is_idle():
        return "IDLE"
    return arm.state.name if arm.state.name in ("ABORT", "RETRY") else "MOVING"


def update_arms(arms, step, time_step, previous_states, debug_motion=False):
    """Advance every independent FSM once per tick, including while a peer waits."""
    refresh_vision = False
    for index, arm in enumerate(arms):
        state = arm.update()
        waiting = arm_wait_reason(arm)
        observation = (state, waiting)
        if debug_motion and step % 240 == 0:
            print(f"[Motion] t={step * time_step:.3f}s Arm {index+1}: {state.name} "
                  f"phase={arm._approach_phase} tcp={arm.get_tcp_position()} "
                  f"segment={arm.current_motion_segment()}", flush=True)
        if observation != previous_states[index]:
            if previous_states[index] is not None and arm.is_idle():
                refresh_vision = True
            label = arm_activity_label(arm)
            print(f"[State] t={step * time_step:.3f}s step={step} "
                  f"Arm {index+1}: {label} phase={state.name}"
                  f"{f' wait={waiting}' if waiting else ''}", flush=True)
            previous_states[index] = observation
    return refresh_vision


def report_parallel_motion(monitor, args, directory=None):
    result = monitor.summary()
    print(f"[Parallel] actual simultaneous TCP motion="
          f"{result['actual_parallel_seconds']:.3f}s; longest continuous overlap="
          f"{result['longest_parallel_seconds']:.3f}s; "
          f"TCP speed threshold={result['minimum_tcp_speed_m_s']:.3f}m/s", flush=True)
    if directory is not None:
        (directory / "parallel_motion.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        gate = getattr(monitor.arms[0], "workspace_gate", None) if monitor.arms else None
        (directory / "traffic_events.json").write_text(
            json.dumps(getattr(gate, "traffic_events", []), indent=2), encoding="utf-8")
    if getattr(args, "require_parallel_motion", False):
        assert result['longest_parallel_seconds'] >= .25, (
            "load test had no continuous 0.25s of measured simultaneous TCP motion")
    assert result['central_zone_violation_steps'] == 0, (
        "two measured TCPs entered the controlled central pickup volume simultaneously")
    return result


def spawn_test_batch(env):
    """Known setup coordinates are used ONLY for spawning, never for dispatch."""
    bodies = {}
    for label, xy, color in (
        ("Box_A", (-0.33, 0.0), (0.95, 0.08, 0.03, 1.0)),
        ("Box_A", (-0.11, 0.0), (0.95, 0.08, 0.03, 1.0)),
        ("Box_B", (0.11, 0.0), (0.03, 0.18, 0.95, 1.0)),
        ("Box_B", (0.33, 0.0), (0.03, 0.18, 0.95, 1.0)),
    ):
        body = env.create_test_object(
            (*xy, env.conveyor_top_z + 0.05), shape="box", color=color,
            orientation=(0, 0, 0, 1),
        )
        bodies[body] = label
    return bodies


def create_controllers(env, moving=False, fault_panel=None, robot="sim"):
    require_robot_backend(robot)
    load_simulation_dependencies()
    arms: list[BaseRobotArm] = []
    x_min, x_max, y_min, y_max = env.layout.overlap_bounds
    workspace_gate = SharedWorkspaceGate(center=((x_min+x_max)/2, (y_min+y_max)/2))
    print(f"[Traffic] central pickup cylinder: center={workspace_gate.center}, "
          f"radius={workspace_gate.radius:.2f}m, ceiling={workspace_gate.ceiling:.2f}m; "
          "one pickup token; overhead/private-station motion remains independent", flush=True)
    env.register_cleanup(workspace_gate.close)
    for index, robot_id in enumerate(env.arm_ids):
        attachment = env.get_arm_gripper(index)
        arm = create_robot_arm(
            robot,
            robot_id, gripper_id=attachment.body_id,
            gripper_joint_indices=attachment.joint_indices,
            tcp_offset_local=attachment.tcp_offset_local,
            display_name=f"Arm {index + 1}",
            trail_color=(1.0, 0.0, 0.0) if index == 0 else (0.0, 0.35, 1.0),
        )
        arm.max_tracking_age = 8.0  # taller shared cell can occlude a moving target longer
        arm.shared_cell_motion = True
        arm.near_pickup_home = True
        arm.station_motion = True
        arm.shared_transit_height = env.layout.transfer_heights[index]
        arm.workspace_gate = workspace_gate
        workspace_gate.names[robot_id] = arm.display_name
        workspace_gate.register(arm, env._find_ur5_urdf())
        arm.open_gripper()
        print(f"[Home] Arm {index + 1} TCP={arm.get_tcp_position()}, "
              f"transfer_z={arm.shared_transit_height:.3f}m", flush=True)
        if fault_panel is not None:
            arm = fault_panel.wrap(arm)
        arms.append(arm)
    routes = {f"arm{i+1}": ArmRoute(f"arm{i+1}", arm) for i, arm in enumerate(arms)}
    destinations = {}
    for arm_name in routes:
        destinations[arm_name] = {}
        for label in ("Box_A", "Box_B"):
            zone = env.get_arm_drop_zone(arm_name, label)
            x, y, z = zone.surface_position
            destinations[arm_name][label] = ((x, y, z+.10),)
    options = dict(arm_drop_targets=destinations, handoff_enabled=True,
                   pickup_bounds=env.layout.overlap_bounds, random_seed=7)
    return arms, (ConveyorScheduler(routes, velocity=(env.conveyor_speed, 0, 0),
                                    time_step=env.time_step,
                                    conveyor_end_x=env.layout.belt_bounds[1],
                                    **options) if moving else Scheduler(routes, **options))


def simulation_attachment_id(frame, pixel, permitted_bodies):
    """Simulation-only body HANDLE for virtual attachment, sampled AFTER vision.

    RGB determines class; RGB+depth determines all grasp coordinates and visual
    track IDs. Segmentation supplies only the body handle required by Bullet's
    createConstraint API. A real gripper adapter omits this function/options.
    """
    u, v = (int(round(value)) for value in pixel)
    height, width = frame.segmentation.shape
    if not (0 <= u < width and 0 <= v < height):
        return None
    encoded = int(frame.segmentation[v, u])
    body_id = encoded & ((1 << 24) - 1)
    return body_id if encoded >= 0 and body_id in permitted_bodies else None


def depth_preview(depth, far):
    """Map METRIC depth into a visible color image, rather than displaying metres as white."""
    valid = np.isfinite(depth) & (depth > 0) & (depth < far - 0.001)
    gray = np.zeros(depth.shape, np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], (1, 99))
        gray[valid] = np.clip(
            (high - depth[valid]) * 255.0 / max(float(high - low), 0.001), 0, 255
        ).astype(np.uint8)
    preview = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def show_camera(frame, objects, scheduler, arms, far, detected_rgb=None):
    rgb_display = cv2.cvtColor(frame.rgb if detected_rgb is None else detected_rgb, cv2.COLOR_RGB2BGR)
    for item in objects:
        x1, y1, x2, y2 = map(int, item["bbox"])
        color = (0, 0, 255) if item["class"] == "Box_A" else (255, 100, 0)
        cv2.rectangle(rgb_display, (x1, y1), (x2, y2), color, 2)
        short_class = "A" if item["class"] == "Box_A" else "B"
        cv2.putText(
            rgb_display, f"{short_class}#{item['track_id']}",
            (x1, max(y1 - 5, 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1,
            cv2.LINE_AA,
        )
    cv2.rectangle(rgb_display, (0, 0), (rgb_display.shape[1], 47), (25, 25, 25), -1)
    for index, arm in enumerate(arms):
        text = f"Arm {index + 1}: {arm_activity_label(arm)} sorted={scheduler.completed_count} missed={getattr(scheduler, 'missed_count', 0)}"
        cv2.putText(rgb_display, text, (8, 18 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Global Camera RGB", rgb_display)
    cv2.imshow("Global Camera Depth (metres)", depth_preview(frame.depth, far))


def vision_noise_options(args):
    """Camera perturbation configuration; scheduling still sees BaseVision."""
    options = dict(noise_std=getattr(args, "noise_std", 0.0),
                   lighting_jitter=getattr(args, "lighting_jitter", 0.0),
                   noise_seed=getattr(args, "noise_seed", 2026))
    if options["noise_std"] or options["lighting_jitter"]:
        print(f"RGB noise: sigma={options['noise_std']}/255, "
              f"lighting=+/-{options['lighting_jitter']:.1%}, seed={options['noise_seed']}; "
              "depth unchanged", flush=True)
    return options


def verify_results(env, scheduler, dispatched_bodies, fixtures, parallel_steps, expected_failed=0,
                   deliveries=None):
    load_simulation_dependencies()
    failed = [task for task in scheduler.assigned_tasks.values() if task.status == TaskStatus.FAILED]
    assert len(failed) == expected_failed, "unexpected static failure count"
    assert scheduler.completed_count + len(failed) == len(fixtures), "some objects were not handled"
    assert len(scheduler.assigned_tasks) == len(fixtures), "a visual object was dispatched twice"
    assert set(dispatched_bodies) == set(fixtures), "a fixture was missed or assigned twice"
    frame = env.camera.capture_frame()
    assert frame.rgb.std() > 10 and np.ptp(frame.depth) > 0.1, "camera is blank"
    config = env.camera.config
    inverse = np.linalg.inv(
        np.asarray(frame.projection_matrix).reshape(4, 4, order="F")
        @ np.asarray(frame.view_matrix).reshape(4, 4, order="F")
    )
    for body, track_id in dispatched_bodies.items():
        task = scheduler.assigned_tasks[track_id]
        assert task.label == fixtures[body], "wrong detected class"
        if task.status == TaskStatus.FAILED:
            assert task.track_id in scheduler.failed_track_ids
            assert body not in {p.getBodyUniqueId(i) for i in range(p.getNumBodies())}, (
                "failed fixture was not discarded"
            )
            continue
        if deliveries is not None:
            assert track_id in deliveries, "completed task has no verified station receipt"
            receipt = deliveries[track_id]
            assert receipt['arm'] == task.arm_name and receipt['label'] == task.label
            assert receipt['error'] < .035, "missed dedicated drop zone"
            continue
        zone = env.get_arm_drop_zone(task.arm_name, task.label)
        assert p.getContactPoints(body, zone.body_id), "object did not land on its platform"
        # Independent verification from camera pixels/depth, never a body-pose read.
        rows, cols = np.where((frame.segmentation & ((1 << 24) - 1)) == body)
        assert len(rows) > 10, "placed object disappeared from camera"
        metric = frame.depth[rows, cols]
        buffer = (config.far - config.near * config.far / metric) / (config.far - config.near)
        offset_u, offset_v = frame.pixel_offset
        clip = np.stack((2 * (cols + offset_u) / config.width - 1,
                         1 - 2 * (rows + offset_v) / config.height,
                         2 * buffer - 1, np.ones(len(rows))))
        points = inverse @ clip
        points = (points[:3] / points[3]).T
        xy = np.median(points[:, :2], axis=0)
        error = float(np.linalg.norm(xy - np.asarray(task.drop_pos[:2])))
        top = float(np.quantile(points[:, 2], 0.95))
        assert error < 0.035 and abs(top - task.drop_pos[2]) < 0.02, "missed drop slot"
        print(f"Verified {task.label} #{track_id} in {zone.name}: camera XY error={error:.4f} m")
    assert p.getNumConstraints() == len(env.gripper_attachments), "virtual grasp constraint leaked"
    return frame


def run_static_demo(args):
    robot = getattr(args, "robot", "sim")
    require_robot_backend(robot)
    load_simulation_dependencies()
    env = SimulationEnvironment(
        gui=not args.headless, camera_config=CameraConfig(renderer=args.renderer)
    )
    step, parallel_steps = 0, 0
    frame = None
    faults = None
    try:
        fixtures = spawn_test_batch(env)
        available_fixtures = dict(fixtures)
        body_by_track = {}
        def remove_fault_target(body):
            if body in available_fixtures:
                p.removeBody(body)
                available_fixtures.pop(body)

        test_fault = getattr(args, "test_fault", None)
        if not args.headless or test_fault:
            faults = FaultInjectionPanel(env, remove_fault_target, gui=not args.headless)
        if test_fault:
            faults.request(test_fault)
        arms, scheduler = create_controllers(env, fault_panel=faults, robot=robot)
        motion_monitor = ParallelMotionMonitor(arms, env.time_step)
        # The factory selects the implementation; the loop only uses BaseVision.
        vision: BaseVision = create_vision(env, detector=args.detector, model_path=args.model,
                                          **vision_noise_options(args))
        dispatched_bodies = {}
        deliveries = {}
        previous_states = [None] * len(arms)
        refresh_vision = False
        if not args.headless:
            for name in ("Global Camera RGB", "Global Camera Depth (metres)"):
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(name, 640, 480)
            cv2.moveWindow("Global Camera RGB", 10, 30)
            cv2.moveWindow("Global Camera Depth (metres)", 660, 30)
        print("Route A: nearest IDLE arm; each arm uses its own RED / BLUE station")
        print("Camera: stepSimulation -> synchronized RGB/depth -> vision; Q/Esc to quit")
        limit = args.steps if args.steps is not None else (40000 if args.headless else None)
        while p.isConnected() and (limit is None or step < limit):
            # This order is deliberate: render the updated scene, then make
            # all control decisions from that same frame. No sleep/wait loops.
            env.step()
            if faults is not None:
                faults.tick(env.simulation_step)
            camera_updated = refresh_vision or step % args.camera_interval == 0
            if camera_updated:
                refresh_vision = False
                frame = env.camera.capture_frame()
                objects = vision.detect(frame.rgb, frame.depth, env.simulation_step * env.time_step)
                candidates = []
                if step >= 120:  # Let new fixtures settle; all arm FSMs still update.
                    for item in objects:
                        body = simulation_attachment_id(frame, item["pixel_center"], available_fixtures)
                        if body is None:
                            continue
                        candidates.append(dict(item, arm_options={
                             "object_id": body, "grasp_offset_z": -0.015,
                             "object_orientation": (0, 0, 0, 1),
                             "object_point_local": (0, 0, 0.05),
                             "object_long_axis_local": (1, 0, 0)},
                        ))
                    by_track = {item["track_id"]: item for item in candidates}
                    for task in scheduler.update(candidates, step):
                        body = by_track[task.track_id]["arm_options"]["object_id"]
                        if body in dispatched_bodies:
                            assert dispatched_bodies[body] == task.track_id and task.handoff_count == 1, (
                                "only one handoff of the same visual ID may repeat an assignment")
                        dispatched_bodies[body] = task.track_id
                        body_by_track[task.track_id] = body
                        p.addUserDebugPoints([task.target_pos], [[1, 0, 0]], pointSize=10, lifeTime=1.0)
                if not args.headless:
                    show_camera(frame, objects, scheduler, arms, env.camera.config.far,
                                getattr(vision, "last_rgb_image", None))
            refresh_vision |= update_arms(arms, env.simulation_step, env.time_step, previous_states,
                                          getattr(args, 'debug_motion', False))
            motion_monitor.sample(env.simulation_step)
            parallel_steps += int(all(not arm.is_idle() for arm in arms))
            scheduler.tick(step)
            for task in scheduler.drain_finished():
                if task.status == TaskStatus.COMPLETED:
                    body = body_by_track[task.track_id]
                    zone = env.get_arm_drop_zone(task.arm_name, task.label)
                    receipt_frame = env.camera.capture_frame()
                    assert p.getContactPoints(body, zone.body_id), "payload did not reach its own station"
                    error = camera_delivery_error(receipt_frame, body, task.drop_pos, env.camera.config)
                    assert error < .035, "dedicated destination placement error"
                    deliveries[task.track_id] = dict(arm=task.arm_name, label=task.label,
                                                     zone=zone.name, error=error)
                    print(f"Verified ID_{task.track_id} in {zone.name}: camera XY error={error:.4f} m")
                    remove_fault_target(body)
                    vision.forget_track(task.track_id)
                    scheduler.release_destination(task.track_id)
                elif task.status == TaskStatus.FAILED:
                    body = body_by_track.pop(task.track_id, None)
                    # Delete before forgetting the visual identity. An already
                    # deleted target is harmless; never remove a recycled ID.
                    remove_fault_target(body)
                    vision.forget_track(task.track_id)
                    print(f"DISCARDED: {task.label} #{task.track_id}, {task.arm_name}, "
                          f"FAILED / never reassign: {task.error}", flush=True)
            step += 1
            if (args.headless and scheduler.completed_count + len(scheduler.failed_track_ids) == len(fixtures)
                    and all(arm.is_idle() for arm in arms)):
                break
            if not args.headless and camera_updated:
                # Pump native window events; this is not a motion wait.
                # Windows waitKey(1) can consume an OS scheduling quantum.
                # Service camera windows per frame, never 240 times/second.
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in (
                    "Global Camera RGB", "Global Camera Depth (metres)"
                )):
                    break
        if args.headless:
            frame = verify_results(env, scheduler, dispatched_bodies, fixtures, parallel_steps,
                                   expected_failed=int(test_fault in ("lost", "collision")), deliveries=deliveries)
            if test_fault in ("grasp", "unreachable"):
                assert any(task.handoff_count == 1 and task.status == TaskStatus.COMPLETED
                           for task in scheduler.assigned_tasks.values()), "fault was not recovered by handoff"
            print(
                f"PASS: {scheduler.completed_count} objects sorted exactly once; "
                f"discarded={len(scheduler.failed_track_ids)}; "
                f"both_assigned_steps={parallel_steps}; total={step} steps "
                f"({step / 240.0:.2f} simulation seconds); renderer={env.camera.active_renderer}"
            )
        if args.save_dir and frame is not None:
            directory = Path(args.save_dir).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(directory / "rgb.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(directory / "depth.png"), depth_preview(frame.depth, env.camera.config.far))
            np.save(directory / "depth_metres.npy", frame.depth)
            print(f"Saved RGB/depth: {directory}")
        report_parallel_motion(motion_monitor, args,
                               Path(args.save_dir).resolve() if args.save_dir else None)
    except p.error:
        if p.isConnected():
            raise
    finally:
        if faults is not None:
            faults.close()
        if not args.headless:
            cv2.destroyAllWindows()
        env.close()


def camera_delivery_error(frame, body, drop, config):
    """Validate the received box using depth pixels, never a physics pose."""
    rows, cols = np.where((frame.segmentation & ((1 << 24) - 1)) == body)
    if len(rows) < 10:
        return float("inf")
    depth = frame.depth[rows, cols]
    buffer = (config.far - config.near * config.far / depth) / (config.far - config.near)
    du, dv = frame.pixel_offset
    clip = np.stack((2 * (cols + du) / config.width - 1, 1 - 2 * (rows + dv) / config.height,
                     2 * buffer - 1, np.ones(len(rows))))
    inverse = np.linalg.inv(np.asarray(frame.projection_matrix).reshape(4, 4, order="F")
                            @ np.asarray(frame.view_matrix).reshape(4, 4, order="F"))
    world = inverse @ clip
    points = (world[:3] / world[3]).T
    measured = (*np.median(points[:, :2], axis=0), np.quantile(points[:, 2], 0.95))
    return float(np.linalg.norm(np.subtract(measured, drop)))


def run_conveyor_demo(args):
    robot = getattr(args, "robot", "sim")
    require_robot_backend(robot)
    load_simulation_dependencies()
    # Four camera observations per simulated second, with a calibrated smaller
    # image to keep the dynamic loop responsive on CPU TinyRenderer.
    env = SimulationEnvironment(gui=not args.headless,
                                camera_config=CameraConfig(width=480, height=360, renderer=args.renderer),
                                conveyor_speed=args.belt_speed)
    object_limit = args.objects if args.objects is not None else (6 if args.headless else None)
    env.enable_infeed(seed=args.seed, interval=tuple(args.spawn_interval), limit=object_limit)
    track_by_body, body_by_track = {}, {}
    removed_by_track = {}

    def remove_fault_target(body):
        # Preserve lifecycle metadata independently of Bullet's recycled body
        # handles until Scheduler acknowledges the failed task after recovery.
        track = track_by_body.pop(body, None)
        if track is not None:
            removed_by_track[track] = env.stream_objects.get(body)
            body_by_track.pop(track, None)
        env.remove_stream_object(body)

    test_fault = getattr(args, "test_fault", None)
    faults = (FaultInjectionPanel(env, remove_fault_target, gui=not args.headless)
              if not args.headless or test_fault else None)
    if test_fault:
        faults.request(test_fault)
    arms, scheduler = create_controllers(env, moving=True, fault_panel=faults, robot=robot)
    motion_monitor = ParallelMotionMonitor(arms, env.time_step)
    vision: BaseVision = create_vision(env, detector=args.detector, model_path=args.model,
                                      conveyor_velocity=(env.conveyor_speed, 0, 0),
                                      **vision_noise_options(args))
    dispatched_serials, retired_serials = set(), set()
    dispatch_attempts = 0
    completed_by_arm = {"arm1": 0, "arm2": 0}
    previous_states = [None, None]
    refresh_vision = False
    parallel_steps, suppressed = 0, False
    max_visible = 0
    logged_visual_ids = set()
    frame = None
    steps = args.steps or round(((object_limit or 6) * args.spawn_interval[1]
                                 + 3 / args.belt_speed + 25) / env.time_step)
    if not args.headless and args.steps is None:
        steps = None
    try:
        if not args.headless:
            for name in ("Global Camera RGB", "Global Camera Depth (metres)"):
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(name, 640, 480)
        print(f"Continuous +X belt: {env.conveyor_speed:.3f} m/s; nearest-IDLE dispatch; own RED / BLUE destinations")
        while p.isConnected() and (steps is None or env.simulation_step < steps):
            env.set_transport_exclusions(arm.held_object_id for arm in arms)
            env.step()
            step = env.simulation_step
            if faults is not None:
                faults.tick(step)
            if refresh_vision or step % args.camera_interval == 0:
                refresh_vision = False
                frame = env.camera.capture_frame()
                objects = vision.detect(frame.rgb, frame.depth, step * env.time_step)
                candidates = []
                for item in objects:
                    if item["track_id"] not in logged_visual_ids:
                        logged_visual_ids.add(item["track_id"])
                        print(f"[Vision] ID_{item['track_id']} {item['class']} world={item['pos_3d']} "
                              f"bbox={item['bbox']}", flush=True)
                    body = simulation_attachment_id(frame, item["pixel_center"], env.stream_objects)
                    if body is None:
                        continue
                    # This is a test assertion, not a body-ID tracking shortcut.
                    # Visual association alone must preserve a moving object's ID.
                    if body in track_by_body:
                        assert track_by_body[body] == item["track_id"], (
                            f"moving visual identity changed: old={track_by_body[body]}, "
                            f"new={item['track_id']}, observed={item['pos_3d']}, step={step}"
                        )
                    track_by_body[body], body_by_track[item["track_id"]] = item["track_id"], body
                    if args.test_miss and not suppressed and item["class"] == "Box_B":
                        scheduler.suppressed_tracks.add(item["track_id"])
                        suppressed = True
                    candidates.append(dict(item, arm_options={
                        "object_id": body, "grasp_offset_z": -0.015, "object_orientation": (0, 0, 0, 1),
                        "object_point_local": (0, 0, 0.05), "object_long_axis_local": (1, 0, 0)}))
                max_visible = max(max_visible, len(candidates))
                for task in scheduler.update(candidates, step):
                    body = body_by_track[task.track_id]
                    serial = env.stream_objects[body][1]
                    if object_limit is not None:
                        if task.handoff_count == 0:
                            assert serial not in dispatched_serials, "payload dispatched more than once"
                        else:
                            assert task.handoff_count == 1 and serial in dispatched_serials, (
                                "handoff must reuse the existing payload and visual ID")
                        dispatched_serials.add(serial)
                    dispatch_attempts += 1
                    p.addUserDebugPoints([task.target_pos], [[1, 0, 0]], pointSize=10, lifeTime=0.5)
                if not args.headless:
                    show_camera(frame, objects, scheduler, arms, env.camera.config.far,
                                getattr(vision, "last_rgb_image", None))
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                    if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in (
                        "Global Camera RGB", "Global Camera Depth (metres)"
                    )):
                        break
            refresh_vision |= update_arms(arms, step, env.time_step, previous_states,
                                          getattr(args, 'debug_motion', False))
            motion_monitor.sample(step)
            parallel_steps += int(all(not arm.is_idle() for arm in arms))
            scheduler.tick(step)
            for body in env.floor_payloads():
                track = track_by_body.get(body)
                if track is not None:
                    scheduler.fail(track, step, "物体掉落到地面")
                else:
                    label, serial = env.stream_objects[body]
                    print(f"漏抓: {label} serial={serial}, 未检测物体落地", flush=True)
                    scheduler.missed_count += 1
                    if object_limit is not None:
                        retired_serials.add(serial)
                    env.remove_stream_object(body)
            for task in scheduler.drain_finished():
                body = body_by_track.get(task.track_id)
                zone = env.get_arm_drop_zone(task.arm_name, task.label) if task.arm_name in scheduler.arms else None
                delivered = (task.status == TaskStatus.COMPLETED and body in env.stream_objects
                             and zone is not None and bool(p.getContactPoints(body, zone.body_id)))
                if delivered:
                    # By RETURN_HOME the release has settled; the latest frame
                    # is at most one camera period old and is sufficient here.
                    error = camera_delivery_error(frame, body, task.drop_pos, env.camera.config)
                    delivered = error < 0.035
                    print(f"Delivery #{task.track_id}: camera error={error:.4f} m")
                scheduler.retire(task, delivered)
                if delivered:
                    completed_by_arm[task.arm_name] += 1
                metadata = (env.stream_objects.get(body) if body is not None else None)
                if metadata is None:
                    metadata = removed_by_track.pop(task.track_id, None)
                if metadata is not None:
                    label, serial = metadata
                    if delivered:
                        assert label == task.label, "wrong class route"
                    if object_limit is not None:
                        assert serial not in retired_serials
                        retired_serials.add(serial)
                    # An output collector removes confirmed deliveries so the
                    # four private drop platforms can accept an endless stream.
                    if body in env.stream_objects:
                        env.remove_stream_object(body)
                    track_by_body.pop(body, None)
                body_by_track.pop(task.track_id, None)
                vision.forget_track(task.track_id)
            if object_limit and env.spawned_count >= object_limit and not env.stream_objects and all(a.is_idle() for a in arms):
                break
        if args.headless:
            assert env.spawned_count == object_limit and len(retired_serials) == object_limit, "stream did not drain"
            assert not scheduler.active_tasks and not scheduler.pending_tasks and not scheduler.assigned_tasks
            assert p.getNumConstraints() == len(env.gripper_attachments), "grasp constraint leaked"
            assert scheduler.completed_count + scheduler.missed_count == object_limit
            assert scheduler.missed_count == (int(args.test_miss) + int(test_fault in ("lost", "collision"))), "unexpected missed count"
            if test_fault:
                if test_fault in ("grasp", "unreachable"):
                    assert any(task.handoff_count == 1 and task.status == TaskStatus.COMPLETED
                               for task in scheduler.history), "fault was not recovered by handoff"
                else:
                    failures = [task for task in scheduler.history if task.status == TaskStatus.FAILED]
                    assert any((task.error or "").startswith("[FAULT TEST]") for task in failures)
                if test_fault == "collision":
                    assert any(event[:2] == ("collision", "contact") for event in faults.events)
            # With slow arrivals, the nearest arm can legitimately finish
            # before the next box appears. Require participation only in an
            # explicitly requested load-sharing regression.
            if getattr(args, "require_both_arms", False):
                assert all(completed_by_arm.values()), "load test did not exercise both arms"
            print(f"PASS moving belt: spawned={env.spawned_count}, sorted={scheduler.completed_count}, "
                  f"missed={scheduler.missed_count}, both_assigned_steps={parallel_steps}, "
                  f"simulation={env.simulation_step * env.time_step:.2f}s; "
                  f"max_simultaneous_detections={max_visible}; unique_dispatches={len(dispatched_serials)}; "
                  f"dispatch_attempts={dispatch_attempts}; completed_by_arm={completed_by_arm}", flush=True)
        if args.save_dir:
            frame = env.camera.capture_frame()
            directory = Path(args.save_dir).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(directory / "rgb.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(directory / "depth.png"), depth_preview(frame.depth, env.camera.config.far))
        report_parallel_motion(motion_monitor, args,
                               Path(args.save_dir).resolve() if args.save_dir else None)
    except p.error:
        if p.isConnected():
            raise
    finally:
        if faults is not None:
            faults.close()
        if not args.headless:
            cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    from visual_test import add_visual_arguments
    add_visual_arguments(parser)
    parser.add_argument("--robot", choices=("sim", "real"), default="sim",
                        help="robot driver: sim uses PyBullet; real is an unimplemented Socket/ROS scaffold")
    parser.add_argument("--debug-motion", action="store_true", help="print motion diagnostics once per simulation second")
    parser.add_argument("--headless", action="store_true", help="finite automatic integration test")
    parser.add_argument("--require-both-arms", action="store_true",
                        help="moving headless load test: also require at least one delivery by each arm")
    parser.add_argument("--require-parallel-motion", action="store_true",
                        help="require >=0.25s continuous measured simultaneous TCP motion")
    parser.add_argument("--static", action="store_true", help="run the previous stationary four-object regression")
    parser.add_argument("--belt-speed", type=float, default=0.05, help="continuous +X belt speed, m/s")
    parser.add_argument("--objects", type=int, help="finite infeed count; GUI default is continuous")
    # .30-.45m belt spacing keeps neighboring targets separate while supplying
    # work quickly enough to exercise both independently moving stations.
    parser.add_argument("--spawn-interval", type=float, nargs=2, default=(6.0, 9.0), metavar=("MIN", "MAX"))
    parser.add_argument("--seed", type=int, default=7, help="reproducible random infeed")
    parser.add_argument("--test-miss", action="store_true", help="let the first blue box run off the belt to verify cleanup")
    parser.add_argument("--test-fault", choices=("grasp", "lost", "unreachable", "collision"),
                        help="queue one panel fault for a finite static/conveyor regression (also works headless)")
    parser.add_argument("--steps", type=int, help="maximum physics steps; GUI default is unlimited")
    parser.add_argument(
        "--camera-interval", type=int,
        help="ticks per camera frame (default: moving 60 / static 120)",
    )
    parser.add_argument("--renderer", choices=("tiny", "opengl"), default="tiny")
    parser.add_argument("--detector", choices=("color", "yolo", "yolo-real"), default="yolo",
                        help="yolo: explicit color-based Nx6 mock; yolo-real: neural inference; color: baseline")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights with Box_A/Box_B classes")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="RGB Gaussian sigma in 0..255 intensity units; e.g. 12 or 24 (default off)")
    parser.add_argument("--lighting-jitter", type=float, default=0.0,
                        help="brightness/contrast random fraction; e.g. 0.04 or 0.08 (default off)")
    parser.add_argument("--noise-seed", type=int, default=2026, help="reproducible RGB perturbation seed")
    parser.add_argument("--save-dir", help="optional output folder for RGB and depth files")
    args = parser.parse_args()
    if args.visual_test:
        from visual_test import run_visual_test
        try:
            run_visual_test(args)
        except ValueError as error:
            parser.error(str(error))
        sys.exit(0)
    try:
        require_robot_backend(args.robot)
    except (NotImplementedError, ValueError) as error:
        parser.error(str(error))
    if args.test_fault and args.test_miss:
        parser.error("--test-fault cannot be combined with --test-miss")
    if (not np.isfinite(args.noise_std) or args.noise_std < 0 or
            not np.isfinite(args.lighting_jitter) or not 0 <= args.lighting_jitter <= 0.5):
        parser.error("noise sigma must be finite/nonnegative; lighting jitter must be in [0, 0.5]")
    if args.camera_interval is None:
        args.camera_interval = 120 if args.static else 60
    if args.camera_interval < 1 or (args.steps is not None and args.steps < 1):
        parser.error("camera interval and step limit must be positive")
    if not np.isfinite(args.belt_speed) or args.belt_speed <= 0 or (args.objects is not None and args.objects < 1):
        parser.error("belt speed and object count must be positive")
    if not all(np.isfinite(args.spawn_interval)) or not 0 < args.spawn_interval[0] <= args.spawn_interval[1]:
        parser.error("spawn interval must be finite and 0 < MIN <= MAX")
    if not args.static and args.spawn_interval[0] * args.belt_speed < 0.19:
        parser.error("spawn interval is too short: leave at least 19 cm between boxes")
    run_static_demo(args) if args.static else run_conveyor_demo(args)
