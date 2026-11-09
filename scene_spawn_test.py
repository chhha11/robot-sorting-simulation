"""Generate central colour bursts and verify shared IK reach; no scheduler.

python scene_spawn_test.py --objects 24
python scene_spawn_test.py --gui --objects 12
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import pybullet as p

from env_setup import CameraConfig, SimulationEnvironment


def check_reachability(env):
    """Independent static IK/FK checks, including TCP offset and joint limits.

    Does not execute a motion trajectory or test simultaneous arm movement.
    All joint states are restored before the generation-only simulation runs.
    """
    xmin, xmax, ymin, ymax = env.layout.overlap_bounds
    results = []
    for arm_index, robot in enumerate(env.arm_ids):
        joints = [j for j in range(p.getNumJoints(robot))
                  if p.getJointInfo(robot, j)[2] == p.JOINT_REVOLUTE]
        saved = [p.getJointState(robot, j)[0] for j in joints]
        lower = [p.getJointInfo(robot, j)[8] for j in joints]
        upper = [p.getJointInfo(robot, j)[9] for j in joints]
        orientation = p.getQuaternionFromEuler((math.pi, 0, arm_index * math.pi))
        attachment = env.get_arm_gripper(arm_index)
        offset = p.multiplyTransforms((0, 0, 0), orientation, attachment.tcp_offset_local,
                                      (0, 0, 0, 1))[0]
        transit_height = env.layout.arm_bases[arm_index][2] + .17
        points = [(float(x), float(y), z, 'overlap')
                  for x in np.linspace(xmin, xmax, 5)
                  for y in np.linspace(ymin, ymax, 3)
                  for z in (env.conveyor_top_z + .085, max(env.conveyor_top_z + .335, transit_height))]
        for zone in env.drop_zones:
            drop = zone.surface_position
            points += [(drop[0]+dx, drop[1], z, zone.name)
                       for dx in (-.09, .09)
                       for z in (drop[2] + .10, max(drop[2] + .35, transit_height))]
        errors, angles, margins = [], [], []
        try:
            for x, y, z, kind in points:
                target = np.array([x, y, z]) - offset
                # Multiple IK seeds distinguish a local solver branch failure
                # from an unreachable pose; validate every result with FK.
                rng = np.random.default_rng(2026)
                for attempt in range(12):
                    if attempt:
                        for joint, value in zip(joints, rng.uniform(-math.pi, math.pi, 6)):
                            p.resetJointState(robot, joint, value)
                    for _ in range(4):
                        solution = p.calculateInverseKinematics(
                            robot, 6, targetPosition=target, targetOrientation=orientation,
                            maxNumIterations=1000, residualThreshold=1e-7)
                        solution = [math.remainder(value, 2*math.pi) for value in solution]
                        for joint, value in zip(joints, solution):
                            p.resetJointState(robot, joint, value)
                    state = p.getLinkState(robot, 6, computeForwardKinematics=True)
                    error = float(np.linalg.norm(np.asarray(state[4]) - target))
                    angle = 2 * math.acos(min(1, abs(float(np.dot(state[5], orientation)))))
                    margin = min(min(value-lo, hi-value)
                                 for value, lo, hi in zip(solution, lower, upper))
                    if (error < .005 and angle < math.radians(1) and margin > math.radians(1)
                            and not any(c[8] < -.001 for c in p.getClosestPoints(robot, env.conveyor_id, 0))):
                        break
                assert error < .005, (arm_index, kind, (x, y, z), error)
                assert angle < math.radians(1), (arm_index, kind, 'orientation', angle)
                assert margin > math.radians(1), (arm_index, kind, 'joint limit', margin)
                assert not any(c[8] < -.001 for c in p.getClosestPoints(robot, env.conveyor_id, 0)), (
                    arm_index, kind, 'arm intersects belt')
                errors.append(error)
                angles.append(math.degrees(angle))
                margins.append(math.degrees(margin))
        finally:
            for joint, value in zip(joints, saved):
                p.resetJointState(robot, joint, value)
        results.append({'arm': arm_index + 1, 'tested_poses': len(points),
                        'max_position_error_mm': max(errors) * 1000,
                        'max_orientation_error_deg': max(angles),
                        'min_joint_limit_margin_deg': min(margins)})
    return results


def check_colour_distribution(env):
    """Sample batch decisions separately from the finite physics scene."""
    env.enable_infeed(seed=2026, interval=(4, 4))
    burst_count, lengths, labels, previous = 0, [], [], None
    for _ in range(10000):
        first = env._next_infeed_label()
        length = 1 + len(env._infeed_colors)
        batch = [first] + [env._next_infeed_label() for _ in range(length - 1)]
        assert len(set(batch)) == 1
        assert length in (1, 3)
        assert first != previous, 'consecutive batches must switch colour'
        previous = first
        burst_count += length > 1
        lengths.append(length)
        labels.extend(batch)
    fraction = burst_count / len(lengths)
    assert .58 < fraction < .62, fraction
    assert .47 < labels.count('Box_A') / len(labels) < .53
    # Explicit extremes exercise both branches and deterministic replay.
    sequences = []
    for _ in range(2):
        env.enable_infeed(seed=7, interval=(4, 4), cluster_probability=1, cluster_size=(3, 3))
        sequence = [env._next_infeed_label() for _ in range(30)]
        assert all(len(set(sequence[i:i+3])) == 1 for i in range(0, 30, 3))
        sequences.append(sequence)
    assert sequences[0] == sequences[1]
    env.enable_infeed(seed=7, interval=(4, 4), cluster_probability=0)
    for _ in range(100):
        env._next_infeed_label()
        assert not env._infeed_colors
    return {'sampled_batches': len(lengths), 'same_colour_burst_fraction': fraction,
            'red_fraction_of_objects': labels.count('Box_A') / len(labels)}


def scene_image(env):
    frame = env.camera.capture_frame()
    image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    xmin, xmax, ymin, ymax = env.layout.overlap_bounds
    matrix = (np.asarray(frame.projection_matrix).reshape(4, 4, order='F')
              @ np.asarray(frame.view_matrix).reshape(4, 4, order='F'))
    def pixel(x, y, z):
        q = matrix @ [x, y, z, 1]
        q = q[:3] / q[3]
        return (round((q[0]+1)/2 * image.shape[1]), round((1-q[1])/2 * image.shape[0]))
    polygon = np.array([pixel(x, y, env.conveyor_top_z + .004)
                        for x, y in ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))])
    cv2.polylines(image, [polygon], True, (0, 220, 255), 2)
    cv2.putText(image, 'Shared pickup: 45 x 16 cm', (18, 28), cv2.FONT_HERSHEY_SIMPLEX,
                .65, (0, 220, 255), 2)
    return image


def run(args):
    directory = Path(args.save_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    env = SimulationEnvironment(gui=args.gui, conveyor_speed=.05,
                                camera_config=CameraConfig(width=960, height=720))
    try:
        reach = check_reachability(env)
        distribution = check_colour_distribution(env)
        env.enable_infeed(seed=args.seed, interval=(4, 4), limit=args.objects)
        records, settled, births, peak = {}, set(), {}, 0
        image = None
        maximum_steps = (args.objects - 1) * 960 + 480
        for _ in range(maximum_steps):
            env.step()
            for body, (label, serial) in list(env.stream_objects.items()):
                pos = p.getBasePositionAndOrientation(body)[0]
                if serial not in records:
                    assert abs(pos[0] - env.layout.spawn_x) < .001
                    assert env.layout.spawn_y_range[0] <= pos[1] <= env.layout.spawn_y_range[1]
                    assert not any(c[8] < -.001 for other in env.stream_objects if other != body
                                   for c in p.getClosestPoints(body, other, 0)), 'birth overlap'
                    records[serial] = {'label': label, 'spawn_position': pos,
                                       'spawn_time': env.simulation_step * env.time_step}
                    births[serial] = env.simulation_step
                if serial not in settled and env.simulation_step - births[serial] >= 120:
                    assert p.getContactPoints(body, env.conveyor_id), (serial, 'not on belt')
                    assert abs(pos[2] - (env.conveyor_top_z + .05)) < .003
                    assert abs(pos[1] - records[serial]['spawn_position'][1]) < .003
                    assert abs(p.getBaseVelocity(body)[0][0] - env.conveyor_speed) < .001
                    assert env.layout.overlap_bounds[0] <= pos[0] <= env.layout.overlap_bounds[1]
                    records[serial]['settled_position'] = pos
                    settled.add(serial)
                if pos[0] > 1.25:
                    env.remove_stream_object(body)
            assert not env.floor_payloads(), 'payload fell off belt before collector'
            if env.simulation_step % 60 == 0:
                xmin, xmax, ymin, ymax = env.layout.overlap_bounds
                visible = sum(xmin <= (pos := p.getBasePositionAndOrientation(body)[0])[0] <= xmax
                              and ymin <= pos[1] <= ymax and serial in settled
                              for body, (_, serial) in env.stream_objects.items())
                if visible > peak:
                    peak = visible
                    image = scene_image(env)
            if args.gui:
                time.sleep(env.time_step)
        assert len(records) == len(settled) == args.objects
        assert peak >= min(3, args.objects), peak
        report = {'arm_bases': env.layout.arm_bases, 'overlap_bounds': env.layout.overlap_bounds,
                  'drop_centres': env.layout.drop_centres, 'reachability': reach,
                  'colour_distribution': distribution, 'generated': len(records),
                  'settled_on_belt': len(settled), 'max_simultaneous_in_overlap': peak,
                  'seed': args.seed, 'records': records}
        (directory / 'scene_spawn_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        assert cv2.imwrite(str(directory / 'scene_overlap.png'), image)
        print('PASS: generation only; ' + json.dumps({k: v for k, v in report.items() if k != 'records'}))
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--objects', type=int, default=24)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save-dir', default='outputs/central_scene')
    args = parser.parse_args()
    if args.objects < 1:
        parser.error('--objects must be positive')
    run(args)
