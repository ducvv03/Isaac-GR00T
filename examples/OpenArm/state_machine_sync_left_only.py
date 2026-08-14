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
Synchronous state machine for the LEFT-ARM-ONLY checkpoint (see
examples/openarm_revo2_left_only_config.py / gr00t_openarm_revo2_left_only_lora)
-- same HOME -> PREPARE -> single-task control loop as state_machine_sync.py,
but for a fundamentally different embodiment: state/action are 9-dim
(left_arm(7) + left_hand(2): thumb_metacarpal, index_proximal), no right_arm
at all, only "head"/"left" cameras (no "right"), single task ("Pick brown box
and place it into blue bin.", no LIFT/PLACE toggle -- this dataset has no
task_index switches, see the modality config's docstring).

--start-mode {prepare,direct} (default prepare) chooses whether to run the
HOME -> PREPARE move before starting inference, or skip it and start
inference directly from wherever the left arm currently is. The prepare move
itself (move_to_prepare_left_only()) only ever publishes to left_arm -- even
though it reads prepare.csv's (possibly bimanual) waypoints via
state_machine.py's load_prepare_waypoints(), any right_arm columns in that
file are simply not used, so the right arm (if the physical robot has one)
is never commanded and stays wherever it currently is.

This file does NOT modify ros2_gr00t_client.py or state_machine.py, and
reuses state_machine.py's wait_for_enter / check_quit_nonblocking /
load_prepare_waypoints / DEFAULT_PREPARE_CSV as-is.
Reuses ros2_gr00t_client.py's generic, embodiment-agnostic pieces
(CAMERA_TOPICS, STATE_JOINT_GROUPS["left_arm"], CONTROLLER_STATE_TOPICS,
TRAJECTORY_COMMAND_TOPICS, HAND_CONTROLLER_STATE_TOPICS,
HAND_TRAJECTORY_COMMAND_TOPICS, NAMES_L_HAND, HAND_CLOSED, decode_image,
CAMERAS_ROTATED_180, request_action_chunk, blend_action_dicts) but defines
its own lean single-arm ROS2 node (LeftArmGr00tClientNode) instead of
OpenArmGr00tClientNode -- that class is bimanual (always subscribes/expects
right_arm feedback, and its hand state/action is a single index_proximal
scalar), neither of which fits this checkpoint.

Hand mapping (this checkpoint's action space is only 2 values per hand step,
not the single index_proximal scalar the other real-hand checkpoints use):
  - thumb_metacarpal: published exactly as the model predicts it.
  - index_proximal:   published exactly as the model predicts it.
  - middle_proximal, ring_proximal, pinky_proximal: each COPY index_proximal's
    predicted value directly (no proportional scaling, unlike
    compute_hand_finger_positions() in ros2_gr00t_client.py).
  - thumb_proximal: not part of this checkpoint's action space at all -- held
    fixed at HAND_CLOSED[1] (see compute_left_hand_finger_positions()).

Prerequisite: a GR00T policy server already running on a checkpoint matching
this schema (e.g. gr00t_openarm_revo2_left_only_lora_10000_trt).

Usage:
    cd examples/OpenArm && source .venv-ros/bin/activate
    python3 state_machine_sync_left_only.py --host <server-host>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import threading
import time
from typing import Callable

from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from ros2_gr00t_client import (
    CAMERA_TOPICS,
    CAMERAS_ROTATED_180,
    CONTROLLER_STATE_TOPICS,
    HAND_CLOSED,
    HAND_CONTROLLER_STATE_TOPICS,
    HAND_TRAJECTORY_COMMAND_TOPICS,
    NAMES_L_HAND,
    STATE_JOINT_GROUPS,
    TRAJECTORY_COMMAND_TOPICS,
    blend_action_dicts,
    decode_image,
    request_action_chunk,
)
from sensor_msgs.msg import Image
from server_client import PolicyClient
from state_machine import (
    DEFAULT_PREPARE_CSV,
    check_quit_nonblocking,
    load_prepare_waypoints,
    wait_for_enter,
)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PICK_PLACE_TASK = "Pick brown box and place it into blue bin."

# Real state feedback for this checkpoint's 2-dim left_hand: named joints read
# from the Revo2 hand controller_state, in the same order this checkpoint's
# state/action expects (thumb_metacarpal, index_proximal) -- see
# examples/openarm_revo2_left_only_config.py / dataset info.json feature names.
LEFT_HAND_STATE_JOINTS = ["left_thumb_metacarpal_joint", "left_index_proximal_joint"]


def compute_left_hand_finger_positions(thumb_metacarpal: float, index_proximal: float) -> np.ndarray:
    """Expand this checkpoint's 2 predicted hand values into all 6 real Revo2
    finger-joint targets, in HAND_FINGERS order (see module docstring for the
    exact mapping). thumb_proximal isn't predicted by this checkpoint at all,
    so it's held fixed at HAND_CLOSED[1]."""
    return np.array(
        [
            thumb_metacarpal,
            HAND_CLOSED[1],
            index_proximal,
            index_proximal,
            index_proximal,
            index_proximal,
        ],
        dtype=np.float64,
    )


def _stream_left_arm_segment(
    pub, node: Node, start: np.ndarray, end: np.ndarray, duration_s: float, command_hz: float
) -> bool:
    """Same streaming pattern as state_machine.py's _stream_segment, trimmed
    to a single left_arm publisher (used by move_to_prepare_left_only() --
    the right arm, if the physical robot has one, is never published to and
    stays wherever it currently is). Returns False (stopping mid-segment) if
    'q' was pressed; True if the segment completed."""
    n_substeps = max(1, round(duration_s * command_hz))
    dt = 1.0 / command_hz
    time_from_start = Duration(sec=0, nanosec=int(dt * 1e9))
    for sub in range(1, n_substeps + 1):
        if check_quit_nonblocking():
            return False
        loop_start = time.perf_counter()
        alpha = sub / n_substeps
        blended = (1.0 - alpha) * start + alpha * end
        msg = JointTrajectory()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.joint_names = STATE_JOINT_GROUPS["left_arm"]
        point = JointTrajectoryPoint()
        point.positions = blended.tolist()
        point.time_from_start = time_from_start
        msg.points = [point]
        pub.publish(msg)
        elapsed = time.perf_counter() - loop_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    return True


def move_to_prepare_left_only(csv_path: Path, duration_s: float, command_hz: float) -> None:
    """STATE: HOME -> PREPARE, left arm only. Unlike state_machine.py's
    move_to_prepare (which drives both arms), this publishes ONLY to
    left_arm -- the right arm, if the physical robot has one, is never
    commanded and stays exactly wherever it currently is. Reuses
    state_machine.py's load_prepare_waypoints() (reads prepare.csv's
    left_arm columns; any right_arm columns in that file are read but simply
    not used here). Assumes the left arm is at its zero/home pose when this
    starts. Stops early (holding position) if 'q' is pressed."""
    waypoints = load_prepare_waypoints(csv_path)
    if not waypoints:
        raise RuntimeError(f"No waypoints parsed from {csv_path}")
    logger.info(
        "STATE HOME -> PREPARE (left arm only): %d waypoints from %s over %.1fs "
        "(press 'q' to stop early)",
        len(waypoints),
        csv_path,
        duration_s,
    )

    home = np.zeros(7, dtype=np.float64)
    sequence = [home, *(w["left_arm"] for w in waypoints)]
    segment_duration = duration_s / (len(sequence) - 1)

    rclpy.init()
    node = Node("openarm_left_only_prepare_move")
    pub = node.create_publisher(JointTrajectory, TRAJECTORY_COMMAND_TOPICS["left_arm"], 10)
    time.sleep(0.5)  # let the publisher connect before the first command
    try:
        for i in range(len(sequence) - 1):
            if not _stream_left_arm_segment(
                pub, node, sequence[i], sequence[i + 1], segment_duration, command_hz
            ):
                logger.info(
                    "STATE HOME -> PREPARE: stopped early by user (q) at waypoint %d/%d",
                    i,
                    len(sequence) - 1,
                )
                return
            logger.info("  waypoint %d/%d reached", i + 1, len(sequence) - 1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    logger.info("STATE HOME -> PREPARE: done, left arm at prepare position.")


class LeftArmGr00tClientNode(Node):
    """Subscribes to left arm + left hand state feedback and head/left
    cameras; publishes commanded left arm + left hand joint positions.
    Single-arm counterpart to ros2_gr00t_client.py's OpenArmGr00tClientNode
    (see module docstring for why that class doesn't fit this checkpoint)."""

    def __init__(self):
        super().__init__("openarm_left_only_gr00t_client")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._lock = threading.Lock()
        self._camera_topics = {"head": CAMERA_TOPICS["head"], "left": CAMERA_TOPICS["left"]}
        self._images: dict[str, np.ndarray | None] = dict.fromkeys(self._camera_topics)
        self._state: dict[str, np.ndarray | None] = {"left_arm": None, "left_hand": None}

        self._image_msg_counts: dict[str, int] = {}
        self._controller_state_msg_counts: dict[str, int] = {}

        self._image_subs = [
            self.create_subscription(
                Image, topic, lambda msg, name=cam: self._on_image(msg, name), sensor_qos
            )
            for cam, topic in self._camera_topics.items()
        ]
        for cam, topic in self._camera_topics.items():
            self.get_logger().info(f"Subscribed to camera '{cam}': {topic}")

        self._state_topics = {
            "left_arm": CONTROLLER_STATE_TOPICS["left_arm"],
            "left_hand": HAND_CONTROLLER_STATE_TOPICS["left_hand"],
        }
        for key, topic in self._state_topics.items():
            self.create_subscription(
                JointTrajectoryControllerState,
                topic,
                lambda msg, group_key=key: self._on_controller_state(msg, group_key),
                sensor_qos,
            )
            self.get_logger().info(f"Subscribed to '{key}' feedback: {topic}")

        command_topics = {
            "left_arm": TRAJECTORY_COMMAND_TOPICS["left_arm"],
            "left_hand": HAND_TRAJECTORY_COMMAND_TOPICS["left_hand"],
        }
        self._trajectory_pubs = {
            key: self.create_publisher(JointTrajectory, topic, 10)
            for key, topic in command_topics.items()
        }
        for key, topic in command_topics.items():
            self.get_logger().info(f"Publishing '{key}' commands to: {topic}")

    def _on_image(self, msg: Image, cam_name: str) -> None:
        try:
            img = decode_image(msg)
            if img is None:
                self.get_logger().warn(
                    f"[{cam_name}] decode_image returned None (encoding={msg.encoding!r})",
                    throttle_duration_sec=2.0,
                )
                return
            if cam_name in CAMERAS_ROTATED_180:
                img = np.rot90(img, 2)
            with self._lock:
                self._images[cam_name] = img.copy()
            self._image_msg_counts[cam_name] = self._image_msg_counts.get(cam_name, 0) + 1
        except Exception:
            self.get_logger().error(f"_on_image('{cam_name}') raised:", exc_info=True)

    def _on_controller_state(self, msg: JointTrajectoryControllerState, group_key: str) -> None:
        """group_key is 'left_arm' (7 joints) or 'left_hand' (2 joints:
        thumb_metacarpal, index_proximal -- see LEFT_HAND_STATE_JOINTS)."""
        try:
            positions = dict(zip(msg.joint_names, msg.feedback.positions, strict=False))
            joint_names = (
                STATE_JOINT_GROUPS["left_arm"] if group_key == "left_arm" else LEFT_HAND_STATE_JOINTS
            )
            missing = [n for n in joint_names if n not in positions]
            if missing:
                self.get_logger().warn(
                    f"[{group_key}] controller_state missing {missing}; "
                    f"msg.joint_names={list(msg.joint_names)}, "
                    f"len(feedback.positions)={len(msg.feedback.positions)}",
                    throttle_duration_sec=2.0,
                )
                return  # wait for a controller_state message that reports every joint
            state = np.array([positions[n] for n in joint_names], dtype=np.float32)
            with self._lock:
                self._state[group_key] = state
            self._controller_state_msg_counts[group_key] = (
                self._controller_state_msg_counts.get(group_key, 0) + 1
            )
        except Exception:
            self.get_logger().error(f"_on_controller_state('{group_key}') raised:", exc_info=True)

    def observation_status(self) -> str:
        with self._lock:
            missing_state = [k for k, v in self._state.items() if v is None]
            missing_images = [k for k, v in self._images.items() if v is None]
        return (
            f"missing_state={missing_state} missing_images={missing_images} | "
            f"msg_counts: images={self._image_msg_counts} "
            f"controller_states={self._controller_state_msg_counts}"
        )

    def get_observation(self) -> dict[str, np.ndarray] | None:
        with self._lock:
            if any(v is None for v in self._state.values()) or any(
                img is None for img in self._images.values()
            ):
                return None
            return {
                "images": {k: v.copy() for k, v in self._images.items()},
                "state": {k: v.copy() for k, v in self._state.items()},
            }

    def publish_command_real(self, action: dict[str, np.ndarray], dt: float) -> None:
        """left_arm: GR00T's per-joint action published as-is (7 joints).
        left_hand: GR00T's 2 predicted values (thumb_metacarpal,
        index_proximal) expanded into all 6 real finger joints via
        compute_left_hand_finger_positions() -- see module docstring."""
        stamp = self.get_clock().now().to_msg()
        time_from_start = Duration(sec=0, nanosec=int(dt * 1e9))
        for key, topic_pub in self._trajectory_pubs.items():
            msg = JointTrajectory()
            msg.header.stamp = stamp
            if key == "left_hand":
                hand = np.asarray(action[key], dtype=np.float64).reshape(-1)
                msg.joint_names = NAMES_L_HAND
                point_positions = compute_left_hand_finger_positions(
                    float(hand[0]), float(hand[1])
                ).tolist()
            else:
                msg.joint_names = STATE_JOINT_GROUPS[key]
                point_positions = np.asarray(action[key], dtype=np.float64).tolist()
            point = JointTrajectoryPoint()
            point.positions = point_positions
            point.time_from_start = time_from_start
            msg.points = [point]
            topic_pub.publish(msg)


class SyncGr00tSessionLeftOnly:
    """Owns one policy-server connection for the left-arm-only checkpoint.
    Same synchronous control loop as state_machine_sync.py's SyncGr00tSession
    (block on get_action(), execute --steps-per-chunk steps, repeat) --
    unlike that file, there's only one task here (no LIFT/PLACE toggle), so
    'q' just stops/restarts the same task instead of switching to a second
    one."""

    def __init__(
        self,
        host: str,
        port: int,
        fps: float,
        command_hz: float,
        swap_blend_duration: float,
        steps_per_chunk: int,
    ):
        self.fps = fps
        self.command_hz = command_hz
        self.swap_blend_duration = swap_blend_duration
        self.steps_per_chunk = steps_per_chunk
        self.task_stats: dict[str, dict[str, float]] = {}

        rclpy.init()
        self.node = LeftArmGr00tClientNode()
        self.spin_thread = threading.Thread(target=lambda: rclpy.spin(self.node), daemon=True)
        self.spin_thread.start()

        logger.info("Connecting to policy server at %s:%d...", host, port)
        self.policy_client = PolicyClient(host=host, port=port)
        if not self.policy_client.ping():
            raise RuntimeError(f"Server at {host}:{port} did not respond to ping()")

        modality_config = self.policy_client.get_modality_config()
        self.action_keys = modality_config["action"].modality_keys
        self.action_chunk_size = len(modality_config["action"].delta_indices)
        self.lang_key = modality_config["language"].modality_keys[0]

        last_status_log = 0.0
        while self.node.get_observation() is None:
            now = time.perf_counter()
            if now - last_status_log > 2.0:
                logger.info("Waiting for full observation... %s", self.node.observation_status())
                last_status_log = now
            time.sleep(0.1)
        first_obs = self.node.get_observation()
        self.last_published_action: dict = {
            key: first_obs["state"][key] for key in self.action_keys
        }

    def run_task_phase(
        self,
        task: str,
        should_switch: Callable[[], bool] | None = None,
        switch_trigger_desc: str = "press 'q' to stop and restart",
    ) -> None:
        """Runs task until should_switch() returns True, then returns (does
        not quit the process). Every self.steps_per_chunk steps, blocks on a
        fresh, synchronous get_action() call -- see module docstring."""
        if should_switch is None:
            should_switch = check_quit_nonblocking
        logger.info("STATE: running task=%r (%s)", task, switch_trigger_desc)
        phase_start_time = time.perf_counter()

        def record_completion() -> None:
            duration = time.perf_counter() - phase_start_time
            stats = self.task_stats.setdefault(task, {"count": 0, "total_time": 0.0})
            stats["count"] += 1
            stats["total_time"] += duration

        command_dt = 1.0 / self.command_hz
        substeps_per_action = max(1, round(self.command_hz / self.fps))
        swap_substeps = max(1, round(self.command_hz * self.swap_blend_duration))
        steps_this_chunk = min(self.steps_per_chunk, self.action_chunk_size)

        def publish_blended(start, end, n_substeps):
            for sub in range(1, n_substeps + 1):
                sub_loop_start = time.perf_counter()
                alpha = sub / n_substeps
                self.node.publish_command_real(
                    blend_action_dicts(start, end, alpha, self.action_keys), dt=command_dt
                )
                elapsed = time.perf_counter() - sub_loop_start
                sleep_time = command_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.last_published_action = end

        while True:
            if should_switch():
                record_completion()
                logger.info("STATE: task=%r finished (%s).", task, switch_trigger_desc)
                return

            obs = self.node.get_observation()
            if obs is None:
                time.sleep(0.01)
                continue

            action_chunk = request_action_chunk(
                self.policy_client, obs, self.lang_key, task, self.action_keys
            )

            for step in range(steps_this_chunk):
                if should_switch():
                    record_completion()
                    logger.info(
                        "STATE: task=%r finished mid-chunk (%s).", task, switch_trigger_desc
                    )
                    return
                action = {key: action_chunk[key][step] for key in self.action_keys}
                n_substeps = swap_substeps if step == 0 else substeps_per_action
                publish_blended(self.last_published_action, action, n_substeps)

    def log_summary(self) -> None:
        """Total number of times the task ran and its average completion time
        -- logged once, right before exit (see main()'s KeyboardInterrupt
        handler)."""
        total_count = sum(s["count"] for s in self.task_stats.values())
        logger.info("=== Task summary (total runs=%d) ===", total_count)
        for task, stats in self.task_stats.items():
            avg = stats["total_time"] / stats["count"] if stats["count"] else 0.0
            logger.info(
                "  task=%r: completed=%d, avg_completion_time=%.2fs", task, stats["count"], avg
            )

    def close(self) -> None:
        self.policy_client.close()
        self.node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenArm left-arm-only state machine (synchronous inference): "
        "HOME -> PREPARE -> single task, 'q' stops and restarts the task. Executes "
        "--steps-per-chunk steps then blocks on the next get_action() call."
    )
    parser.add_argument(
        "--prepare-csv",
        type=Path,
        default=DEFAULT_PREPARE_CSV,
        help="CSV of waypoints from home to prepare position",
    )
    parser.add_argument(
        "--prepare-duration",
        type=float,
        default=8.0,
        help="Total seconds to move from home through prepare.csv to the prepare "
        "position. Not verified against real motion limits -- tune conservatively "
        "for your hardware before trusting this default.",
    )
    parser.add_argument(
        "--prepare-command-hz",
        type=float,
        default=250.0,
        help="Publish rate for the HOME->PREPARE move.",
    )
    parser.add_argument(
        "--start-mode",
        choices=["prepare", "direct"],
        default="prepare",
        help="'prepare' (default): move HOME -> PREPARE -- left arm only, the "
        "right arm (if the physical robot has one) is left untouched -- before "
        "starting inference. 'direct': skip the prepare move entirely and start "
        "inference from wherever the left arm currently is.",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Policy server host")
    parser.add_argument("--port", type=int, default=5555, help="Policy server port")
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Must match the checkpoint's training data fps.",
    )
    parser.add_argument("--command-hz", type=float, default=250.0, help="Real-mode publish rate")
    parser.add_argument(
        "--steps-per-chunk",
        type=int,
        default=16,
        help="Number of steps to execute from each fetched action chunk before "
        "blocking on the next get_action() call. Capped at the chunk's actual "
        "length (action_horizon) if smaller.",
    )
    parser.add_argument(
        "--swap-blend-duration",
        type=float,
        default=0.1,
        help="Seconds to blend into the first step of each freshly-fetched chunk.",
    )
    args = parser.parse_args()

    if args.start_mode == "prepare":
        if not wait_for_enter(
            "\nPress Enter to move HOME -> PREPARE (left arm only), or 'q' to quit."
        ):
            return
        move_to_prepare_left_only(args.prepare_csv, args.prepare_duration, args.prepare_command_hz)
    else:
        logger.info("STATE: --start-mode=direct -- skipping HOME -> PREPARE move.")

    if not wait_for_enter(f"\nPress Enter to start task: {PICK_PLACE_TASK!r}, or 'q' to quit."):
        return
    session = SyncGr00tSessionLeftOnly(
        args.host,
        args.port,
        args.fps,
        args.command_hz,
        args.swap_blend_duration,
        args.steps_per_chunk,
    )
    try:
        # Single-task loop: 'q' stops the task and immediately restarts it,
        # looping forever. Ctrl+C is the only way out.
        while True:
            session.run_task_phase(PICK_PLACE_TASK)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        session.log_summary()
    finally:
        session.close()


if __name__ == "__main__":
    main()
