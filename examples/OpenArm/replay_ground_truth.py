#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Replay a recorded episode's observation.state directly onto the robot,
bypassing GR00T inference entirely. Use this to isolate whether visually
wrong robot behavior comes from the model's predictions or from the
joint-mapping/publish pipeline shared with ros2_gr00t_client.py (same
STATE_JOINT_GROUPS/COMMAND_JOINT_GROUPS/hand-threshold/TRAJECTORY_COMMAND_TOPICS
constants, imported directly from it) -- if replaying ground-truth data still
looks wrong, the bug is in the mapping/publish code, not the model.

--mode sim (default): publishes a flat sensor_msgs/JointState to /joint_command
(Isaac Sim), all 4 modality groups (left_arm/right_arm/left_hand/right_hand),
for the 16-dim sim dataset (e.g. openarm_revo2_hand_config.py).

--mode real: publishes trajectory_msgs/JointTrajectory to
TRAJECTORY_COMMAND_TOPICS, arms only (left_arm/right_arm, no hand -- the real
dataset has no hand state at all, see openarm_revo2_arms_only_config.py).
Interpolates between consecutive recorded rows and streams at --command-hz,
same reasoning as ros2_gr00t_client.py's --command-hz: the real arm
controllers run interpolation_method: none and expect a high-frequency
command stream, not sparse waypoints at the dataset's native --fps.

Runs under ROS2's system Python (rclpy), same as ros2_gr00t_client.py.

Usage:
    # sim, 16-dim (arm+hand)
    python examples/OpenArm/replay_ground_truth.py --mode sim \
        --episode-path /home/ws/data/sim/lerobot_v2_data_filtered/20260803/000/data/chunk-000/episode_000000.parquet \
        --fps 20

    # real, 14-dim (arms only)
    python examples/OpenArm/replay_ground_truth.py --mode real \
        --episode-path /home/ws/data/real/lerobot_v2_data_filtered/20260731/001/data/chunk-000/episode_000000.parquet \
        --fps 20
"""

import argparse
import time

from builtin_interfaces.msg import Duration
import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ros2_gr00t_client import (
    COMMAND_JOINT_GROUPS,
    HandDebouncer,
    JOINT_COMMAND_TOPIC,
    STATE_JOINT_GROUPS,
    TRAJECTORY_COMMAND_TOPICS,
)

SIM_STATE_KEYS = ["left_arm", "right_arm", "left_hand", "right_hand"]
REAL_STATE_KEYS = ["left_arm", "right_arm"]


def split_state_row(row: np.ndarray, state_keys: list[str]) -> dict[str, np.ndarray]:
    """Split a flat observation.state row into named groups, in the same
    column order as meta/modality.json."""
    values = {}
    idx = 0
    for key in state_keys:
        width = len(STATE_JOINT_GROUPS[key])
        values[key] = row[idx : idx + width]
        idx += width
    return values


def build_command_msg(
    values: dict[str, np.ndarray],
    stamp,
    hand_debouncers: dict[str, HandDebouncer],
) -> JointState:
    """--mode sim: flat JointState for /joint_command."""
    msg = JointState()
    msg.header.stamp = stamp
    msg.name = [n for group in COMMAND_JOINT_GROUPS.values() for n in group]
    positions = []
    for key in COMMAND_JOINT_GROUPS:
        if key in ("left_hand", "right_hand"):
            scalar = float(values[key][0])
            positions.append(hand_debouncers[key].update(scalar))
        else:
            positions.append(np.asarray(values[key], dtype=np.float64))
    msg.position = np.concatenate(positions).tolist()
    return msg


def stream_real_segment(
    pubs: dict[str, object],
    node: Node,
    start: dict[str, np.ndarray],
    end: dict[str, np.ndarray],
    duration_s: float,
    command_hz: float,
) -> None:
    """--mode real: linearly interpolate start -> end and stream single-point
    JointTrajectory commands at command_hz -- same streaming pattern as
    ros2_gr00t_client.py's publish_command_real / state_machine.py's
    _stream_segment (interpolation_method: none needs a high-frequency
    command stream, not sparse waypoints)."""
    n_substeps = max(1, round(duration_s * command_hz))
    dt = 1.0 / command_hz
    time_from_start = Duration(sec=0, nanosec=int(dt * 1e9))
    for sub in range(1, n_substeps + 1):
        loop_start = time.perf_counter()
        alpha = sub / n_substeps
        stamp = node.get_clock().now().to_msg()
        for arm_key, pub in pubs.items():
            blended = (1.0 - alpha) * start[arm_key] + alpha * end[arm_key]
            msg = JointTrajectory()
            msg.header.stamp = stamp
            msg.joint_names = STATE_JOINT_GROUPS[arm_key]
            point = JointTrajectoryPoint()
            point.positions = blended.tolist()
            point.time_from_start = time_from_start
            msg.points = [point]
            pub.publish(msg)
        elapsed = time.perf_counter() - loop_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def run_sim(node: Node, states: np.ndarray, fps: float, loop: bool) -> None:
    pub = node.create_publisher(JointState, JOINT_COMMAND_TOPIC, 10)
    print(f"Publishing to: {JOINT_COMMAND_TOPIC}")
    dt = 1.0 / fps
    hand_debouncers = {"left_hand": HandDebouncer(), "right_hand": HandDebouncer()}

    while True:
        for i, row in enumerate(states):
            values = split_state_row(row, SIM_STATE_KEYS)
            msg = build_command_msg(values, node.get_clock().now().to_msg(), hand_debouncers)
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.0)
            if i % int(max(fps, 1)) == 0:
                print(f"frame {i}/{len(states)}")
            time.sleep(dt)
        if not loop:
            break
        print("Looping...")
        # Reset debounce state for a fresh loop pass.
        hand_debouncers = {"left_hand": HandDebouncer(), "right_hand": HandDebouncer()}


def run_real(node: Node, states: np.ndarray, fps: float, command_hz: float, loop: bool) -> None:
    pubs = {
        key: node.create_publisher(JointTrajectory, topic, 10)
        for key, topic in TRAJECTORY_COMMAND_TOPICS.items()
    }
    for key, topic in TRAJECTORY_COMMAND_TOPICS.items():
        print(f"Publishing '{key}' commands to: {topic}")
    time.sleep(0.5)  # let publishers connect before the first command

    rows = [split_state_row(row, REAL_STATE_KEYS) for row in states]
    segment_duration = 1.0 / fps  # one recorded frame's worth of real time, per segment

    while True:
        for i in range(len(rows) - 1):
            stream_real_segment(pubs, node, rows[i], rows[i + 1], segment_duration, command_hz)
            if i % int(max(fps, 1)) == 0:
                print(f"frame {i}/{len(rows)}")
        if not loop:
            break
        print("Looping...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded observation.state directly onto the robot")
    parser.add_argument("--mode", choices=["sim", "real"], default="sim", help="See module docstring")
    parser.add_argument("--episode-path", type=str, required=True, help="Path to episode_XXXXXX.parquet")
    parser.add_argument("--fps", type=float, default=20.0, help="Playback rate (match the dataset's fps)")
    parser.add_argument(
        "--command-hz", type=float, default=250.0,
        help="--mode real only: publish rate for the interpolated command stream "
        "between recorded rows (see ros2_gr00t_client.py's --command-hz).",
    )
    parser.add_argument("--loop", action="store_true", help="Loop the episode instead of playing once")
    args = parser.parse_args()

    df = pd.read_parquet(args.episode_path)
    print(f"Loaded {len(df)} frames from {args.episode_path}")
    if "task_index" in df.columns:
        print(f"task_index values in this episode: {sorted(df['task_index'].unique().tolist())}")

    states = np.stack(df["observation.state"].to_numpy())
    expected_dim = 16 if args.mode == "sim" else 14
    if states.shape[1] != expected_dim:
        raise ValueError(
            f"--mode {args.mode} expects {expected_dim}-dim observation.state, "
            f"got {states.shape[1]} from {args.episode_path}. Wrong dataset for this mode?"
        )

    rclpy.init()
    node = Node("openarm_ground_truth_replay")
    try:
        if args.mode == "sim":
            run_sim(node, states, args.fps, args.loop)
        else:
            run_real(node, states, args.fps, args.command_hz, args.loop)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
