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
OpenArm ROS2 <-> GR00T policy-server bridge, for sim testing (e.g. Isaac Sim).

Runs under ROS2's system Python (rclpy), separate from this repo's .venv (which
has torch/transformers and targets Python 3.10). Talks to a running
`gr00t/eval/run_gr00t_server.py` over ZMQ via the standalone client in
`server_client.py` -- no `gr00t` package import needed here, only numpy/zmq/msgpack.

Usage:
    # Terminal 1: policy server (this repo's .venv, GPU)
    cd /home/ws/pnk/vla/Isaac-GR00T && source .venv/bin/activate
    python gr00t/eval/run_gr00t_server.py \
        --model-path <finetuned-checkpoint-dir> \
        --embodiment-tag NEW_EMBODIMENT

    # Terminal 2: Isaac Sim (or whatever publishes /joint_states + the 3 camera
    # topics and subscribes to /joint_command) -- not part of this repo.

    # Terminal 3: this client (system Python, ROS2 Jazzy)
    python3 examples/OpenArm/ros2_gr00t_client.py --task "Pick up all the items on the table and put them into the bin on the right."
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from server_client import PolicyClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TASK = "Pick up all the items on the table and put them into the bin on the right."

# ROS2 topics
JOINT_STATE_TOPIC = "/joint_states"
JOINT_COMMAND_TOPIC = "/joint_command"
CAMERA_TOPICS = {
    "head": "/camera/head/image_raw",
    "left": "/camera/wrist_left/image_raw",
    "right": "/camera/wrist_right/image_raw",
}

# Joint names read from /joint_states, grouped to match openarm_config.py's
# state modality_keys ("left_arm", "right_arm", "left_gripper", "right_gripper").
# openarm_left_finger_joint2 / openarm_right_finger_joint2 / openarm_left_hand /
# openarm_right_hand / openarm_left_ee_tcp_joint / openarm_right_ee_tcp_joint are
# read from /joint_states (if present) but not used -- they carry no independent
# training signal (mimic/passive joints), matching openarm_config.py's 16-dim layout.
STATE_JOINT_GROUPS = {
    "left_arm": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right_arm": [f"openarm_right_joint{i}" for i in range(1, 8)],
    "left_gripper": ["openarm_left_finger_joint1"],
    "right_gripper": ["openarm_right_finger_joint1"],
}

# Joints published on /joint_command, in order, built by concatenating the
# action modality groups below (must match openarm_config.py's action
# modality_keys order: left_arm, right_arm, left_gripper, right_gripper).
COMMAND_JOINT_GROUPS = STATE_JOINT_GROUPS


def decode_image(msg: Image) -> np.ndarray | None:
    """Decode a sensor_msgs/Image into an HWC uint8 RGB array."""
    if msg.encoding == "rgb8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1]
    if msg.encoding == "rgba8":
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return img[:, :, :3]
    logger.warning("Unsupported image encoding: %s", msg.encoding)
    return None


class OpenArmGr00tClientNode(Node):
    """Subscribes to joint states + cameras, publishes commanded joint positions."""

    def __init__(self):
        super().__init__("openarm_gr00t_client")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._lock = threading.Lock()
        self._images: dict[str, np.ndarray | None] = dict.fromkeys(CAMERA_TOPICS)
        self._state: dict[str, np.ndarray] | None = None

        self._image_subs = [
            self.create_subscription(
                Image, topic, lambda msg, name=cam: self._on_image(msg, name), sensor_qos
            )
            for cam, topic in CAMERA_TOPICS.items()
        ]
        for cam, topic in CAMERA_TOPICS.items():
            self.get_logger().info(f"Subscribed to camera '{cam}': {topic}")

        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._on_joint_state, sensor_qos)
        self.get_logger().info(f"Subscribed to: {JOINT_STATE_TOPIC}")

        self.command_pub = self.create_publisher(JointState, JOINT_COMMAND_TOPIC, 10)
        self.get_logger().info(f"Publishing to: {JOINT_COMMAND_TOPIC}")

    def _on_image(self, msg: Image, cam_name: str) -> None:
        img = decode_image(msg)
        if img is None:
            return
        with self._lock:
            self._images[cam_name] = img.copy()

    def _on_joint_state(self, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position, strict=False))
        state = {}
        for key, joint_names in STATE_JOINT_GROUPS.items():
            missing = [n for n in joint_names if n not in positions]
            if missing:
                return  # wait for a complete /joint_states message
            state[key] = np.array([positions[n] for n in joint_names], dtype=np.float32)
        with self._lock:
            self._state = state

    def get_observation(self) -> dict[str, np.ndarray] | None:
        with self._lock:
            if self._state is None or any(img is None for img in self._images.values()):
                return None
            return {
                "images": {k: v.copy() for k, v in self._images.items()},
                "state": {k: v.copy() for k, v in self._state.items()},
            }

    def publish_command(self, action: dict[str, np.ndarray]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [n for group in COMMAND_JOINT_GROUPS.values() for n in group]
        msg.position = np.concatenate(
            [np.asarray(action[key], dtype=np.float64) for key in COMMAND_JOINT_GROUPS]
        ).tolist()
        self.command_pub.publish(msg)


def run(args: argparse.Namespace) -> None:
    rclpy.init()
    node = OpenArmGr00tClientNode()
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    # NOTE: PolicyClient's REQ socket has no recv timeout configured (timeout_ms is stored
    # but unused), so this call blocks indefinitely if the server isn't already up -- start
    # the policy server (Terminal 1 in the module docstring) before running this client.
    logger.info("Connecting to policy server at %s:%d...", args.host, args.port)
    policy_client = PolicyClient(host=args.host, port=args.port)
    modality_config = policy_client.get_modality_config()

    action_keys = modality_config["action"].modality_keys
    action_chunk_size = len(modality_config["action"].delta_indices)
    lang_key = modality_config["language"].modality_keys[0]

    if not (1 <= args.open_loop_horizon <= action_chunk_size):
        raise ValueError(
            f"--open-loop-horizon={args.open_loop_horizon} must satisfy "
            f"1 <= open_loop_horizon <= action_chunk_size={action_chunk_size}."
        )

    while node.get_observation() is None:
        time.sleep(0.1)
    logger.info("Receiving observations. Starting control loop (task=%r).", args.task)

    pred_action_chunk: dict[str, np.ndarray] | None = None
    actions_from_chunk_completed = 0

    try:
        while True:
            loop_start = time.perf_counter()
            obs = node.get_observation()
            if obs is None:
                time.sleep(0.01)
                continue

            if pred_action_chunk is None or actions_from_chunk_completed >= args.open_loop_horizon:
                actions_from_chunk_completed = 0
                video_dict = {
                    cam: img[None, None, ...] for cam, img in obs["images"].items()
                }  # (B=1, T=1, H, W, C)
                state_dict = {
                    key: obs["state"][key][None, None, :] for key in obs["state"]
                }  # (B=1, T=1, D)
                request_data = {
                    "video": video_dict,
                    "state": state_dict,
                    "language": {lang_key: [[args.task]]},
                }
                response, _info = policy_client.get_action(request_data)
                pred_action_chunk = {key: response[key][0] for key in action_keys}  # (chunk, D)

            action = {
                key: pred_action_chunk[key][actions_from_chunk_completed] for key in action_keys
            }
            node.publish_command(action)
            actions_from_chunk_completed += 1

            elapsed = time.perf_counter() - loop_start
            sleep_time = (1.0 / args.fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        policy_client.close()
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenArm ROS2 <-> GR00T policy-server client")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task instruction")
    parser.add_argument("--host", type=str, default="localhost", help="Policy server host")
    parser.add_argument("--port", type=int, default=5555, help="Policy server port")
    parser.add_argument("--fps", type=float, default=30.0, help="Control loop frequency")
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=16,
        help="How many actions to execute from a predicted chunk before re-querying the server",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
