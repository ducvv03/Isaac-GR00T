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
OpenArm ROS2 <-> GR00T policy-server bridge. Supports two modes:

- --mode sim (default): state from /joint_states, commands published as a single
  flat sensor_msgs/JointState to /joint_command. For sim testing (e.g. Isaac Sim).
- --mode real: arm state read from the `feedback` field of
  /left_joint_trajectory_controller/controller_state and
  /right_joint_trajectory_controller/controller_state (control_msgs/JointTrajectoryControllerState),
  commands published as trajectory_msgs/JointTrajectory to
  /left_joint_trajectory_controller/joint_trajectory and
  /right_joint_trajectory_controller/joint_trajectory. With --with-hand, hand
  state is likewise read from /left_revo2_hand_controller/controller_state and
  /right_revo2_hand_controller/controller_state (real sensor feedback, just the
  single index_proximal joint per side -- same scalar proxy the checkpoint was
  trained on), and commands are published as trajectory_msgs/JointTrajectory to
  /left_revo2_hand_controller/joint_trajectory and
  /right_revo2_hand_controller/joint_trajectory, expanding GR00T's predicted
  index_proximal scalar into all 6 real finger joints per hand (see
  compute_hand_finger_positions).

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

    python gr00t/eval/run_gr00t_server.py \
    --model-path gr00t_openarm_revo2_arms_only_real_lora_5000_merged \
    --embodiment-tag new_embodiment \
    --trt-engine-path ./gr00t_trt_deployment_arms_only_real/engines \
    --trt-mode n17_full_pipeline

    # Terminal 2: Isaac Sim, or the real OpenArm's ros2_control stack -- not part
    # of this repo.

    # Terminal 3: this client (system Python, ROS2 Jazzy)
    python3 examples/OpenArm/ros2_gr00t_client.py --mode sim --task "Pick up all the items on the table and put them into the bin on the right."
    python3 examples/OpenArm/ros2_gr00t_client.py --mode real --task "..."
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time

from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from server_client import PolicyClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TASK = "Pick up all the items on the table and put them into the bin on the right."

# ROS2 topics -- sim mode
JOINT_STATE_TOPIC = "/joint_states"
JOINT_COMMAND_TOPIC = "/joint_command"
CAMERA_TOPICS = {
    "head": "/cam_head/cam_head/color/image_raw",
    "left": "/cam_left/cam_left/color/image_raw",
    "right": "/cam_right/cam_right/color/image_raw",
}

# ROS2 topics -- real mode (ros2_control joint_trajectory_controller, one per arm).
CONTROLLER_STATE_TOPICS = {
    "left_arm": "/left_joint_trajectory_controller/controller_state",
    "right_arm": "/right_joint_trajectory_controller/controller_state",
}
TRAJECTORY_COMMAND_TOPICS = {
    "left_arm": "/left_joint_trajectory_controller/joint_trajectory",
    "right_arm": "/right_joint_trajectory_controller/joint_trajectory",
}

# ROS2 topics -- real mode hand feedback/command (Revo2 hand controller, one per
# side), only subscribed/published when no_hand=False. Feedback carries just the
# single index_proximal joint (matches STATE_JOINT_GROUPS["left_hand"/"right_hand"]);
# commands carry all 6 real finger joints (see compute_hand_finger_positions).
HAND_CONTROLLER_STATE_TOPICS = {
    "left_hand": "/left_revo2_hand_controller/controller_state",
    "right_hand": "/right_revo2_hand_controller/controller_state",
}
HAND_TRAJECTORY_COMMAND_TOPICS = {
    "left_hand": "/left_revo2_hand_controller/joint_trajectory",
    "right_hand": "/right_revo2_hand_controller/joint_trajectory",
}

# Joint names read from /joint_states, grouped to match
# openarm_revo2_hand_config.py's state modality_keys ("left_arm", "right_arm",
# "left_hand", "right_hand"). left_hand/right_hand each read a single
# index_proximal joint -- the scalar proxy the checkpoint was trained on for
# hand open/closed state, not a full per-finger reading.
STATE_JOINT_GROUPS = {
    "left_arm": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right_arm": [f"openarm_right_joint{i}" for i in range(1, 8)],
    "left_hand": ["left_index_proximal_joint"],
    "right_hand": ["right_index_proximal_joint"],
}
HAND_KEYS = ("left_hand", "right_hand")

# All 6 Revo2 finger joints per hand, in URDF upper-limit order. GR00T's
# left_hand/right_hand action is a single scalar (the trained index_proximal
# proxy); compute_hand_finger_positions() expands it into all 6 real finger
# joints per side, proportionally (see below).
HAND_FINGERS = [
    "thumb_metacarpal",
    "thumb_proximal",
    "index_proximal",
    "middle_proximal",
    "ring_proximal",
    "pinky_proximal",
]
NAMES_L_HAND = [f"left_{f}_joint" for f in HAND_FINGERS]
NAMES_R_HAND = [f"right_{f}_joint" for f in HAND_FINGERS]

# Finger closed limits (rad), same order as HAND_FINGERS (revo2 URDF upper
# limits); every finger opens at 0.0. Doubles as the per-finger "max" that
# compute_hand_finger_positions() scales against.
HAND_CLOSED = np.array([0.0, 0.0, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)
HAND_OPEN = np.zeros(len(HAND_FINGERS), dtype=np.float64)


def compute_hand_finger_positions(index_scalar: float) -> np.ndarray:
    """Expand GR00T's single index_proximal scalar action into all 6 real
    Revo2 finger-joint targets, in HAND_FINGERS order (thumb_metacarpal,
    thumb_proximal, index_proximal, middle_proximal, ring_proximal,
    pinky_proximal).

    thumb_metacarpal/thumb_proximal are held at their HAND_CLOSED limits
    always -- the checkpoint only proxies hand state/action through a single
    index_proximal scalar, so there's no signal to drive an independent thumb
    pose. index_proximal itself is the model's own prediction, clipped into
    its valid [0, HAND_CLOSED[2]] range. middle/ring/pinky_proximal scale
    proportionally with how closed index_proximal is (ratio = index_scalar /
    HAND_CLOSED[2], the index finger's own closed limit), each capped at
    their own HAND_CLOSED limit -- e.g. index_scalar=0.3 (of max 0.4) ->
    ratio=0.75 -> middle=0.5*0.75, ring=0.6*0.75, pinky=0.7*0.75. All four
    non-thumb fingers curl together, proportionally, instead of only index
    moving.
    """
    index_max = HAND_CLOSED[2]  # index_proximal's own closed limit
    ratio = float(np.clip(index_scalar / index_max, 0.0, 1.0))
    return np.array(
        [
            HAND_CLOSED[0],  # thumb_metacarpal -- fixed
            HAND_CLOSED[1],  # thumb_proximal -- fixed
            ratio * HAND_CLOSED[2],  # index_proximal -- model's own prediction, clipped
            ratio * HAND_CLOSED[3],  # middle_proximal
            ratio * HAND_CLOSED[4],  # ring_proximal
            ratio * HAND_CLOSED[5],  # pinky_proximal
        ],
        dtype=np.float64,
    )


# 0.2 sits with clear margin below the recorded "closed/holding" floor (~0.35-0.41
# in demo data) and clear margin above the "open" ceiling (~0.003-0.01) -- see
# replay_ground_truth.py investigation. 0.3-0.4 range is NOT safe: recorded
# closed-hold values commonly dip into it, which would flicker open/closed.
#
# HandDebouncer/HAND_CLOSE_THRESHOLD/HAND_OPEN_DEBOUNCE_FRAMES are no longer used
# by the live control path (publish_command/publish_command_real now use the
# proportional compute_hand_finger_positions() above) -- kept only because
# replay_ground_truth.py's ground-truth-replay diagnostic still imports
# HandDebouncer for its binary open/closed playback.
HAND_CLOSE_THRESHOLD = 0.2

# Frames the scalar action must stay below HAND_CLOSE_THRESHOLD before
# HandDebouncer actually reports "open". At the dataset's 20fps this is
# ~0.75s. Chosen from demo-data analysis: brief sub-threshold dips during an
# otherwise continuous hold ran up to 15 frames (likely sensor/actuation
# noise), while genuine open/release periods ran 23+ frames -- 15 filters the
# former without delaying the latter meaningfully.
HAND_OPEN_DEBOUNCE_FRAMES = 15


class HandDebouncer:
    """Per-hand hysteresis over GR00T's single-scalar left_hand/right_hand action.

    Closes immediately on any above-threshold reading (reacting fast to "close"
    is safe). Only reports "open" after HAND_OPEN_DEBOUNCE_FRAMES consecutive
    below-threshold readings, so a brief noisy dip during a continuous hold
    doesn't cause a spurious release mid-lift.
    """

    def __init__(self, debounce_frames: int = HAND_OPEN_DEBOUNCE_FRAMES):
        self._debounce_frames = debounce_frames
        self._below_count = 0
        self._is_open = False

    def update(self, scalar: float) -> np.ndarray:
        if scalar > HAND_CLOSE_THRESHOLD:
            self._below_count = 0
            self._is_open = False
        else:
            self._below_count += 1
            if self._below_count >= self._debounce_frames:
                self._is_open = True
        return HAND_OPEN if self._is_open else HAND_CLOSED


# Joints published on /joint_command. Arms pass GR00T's per-joint action
# through unchanged; left_hand/right_hand expand from GR00T's single scalar
# action into all 6 real finger joints per side (see publish_command).
COMMAND_JOINT_GROUPS = {
    "left_arm": STATE_JOINT_GROUPS["left_arm"],
    "right_arm": STATE_JOINT_GROUPS["right_arm"],
    "left_hand": NAMES_L_HAND,
    "right_hand": NAMES_R_HAND,
}


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
    """Subscribes to joint states + cameras, publishes commanded joint positions.

    mode="sim": state from /joint_states (all 4 modality groups from one message),
    commands published as a single flat JointState to /joint_command.

    mode="real": arm state (left_arm, right_arm) comes from the `feedback` field
    of each arm's controller_state topic (control_msgs/JointTrajectoryControllerState).
    Commands for each arm are published as a one-point trajectory_msgs/JointTrajectory
    to that arm's joint_trajectory topic.

    no_hand=False (--with-hand) in real mode: hand state (left_hand, right_hand,
    each the single index_proximal joint) comes from a real sensor too, the
    `feedback` field of each hand's Revo2 controller_state topic -- same
    mechanism as the arms, not self-referential. Commands are published as a
    one-point trajectory_msgs/JointTrajectory per hand, expanding GR00T's
    predicted index_proximal scalar into all 6 real finger joints via
    compute_hand_finger_positions() (thumb fixed, middle/ring/pinky scale
    proportionally with index).

    no_hand=True: for checkpoints trained on an arms-only modality config (no
    left_hand/right_hand keys at all, e.g. examples/openarm_revo2_arms_only_config.py
    on real hardware data). Skips every hand-related subscribe/publish step
    above entirely -- state, observation, and command all cover only
    left_arm/right_arm. Required when running such a checkpoint: it never returns
    "left_hand"/"right_hand" in its action dict, so the hand-handling code above
    would KeyError without this flag.
    """

    def __init__(self, mode: str = "sim", no_hand: bool = False):
        super().__init__("openarm_gr00t_client")
        self._mode = mode
        self._no_hand = no_hand

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._lock = threading.Lock()
        self._images: dict[str, np.ndarray | None] = dict.fromkeys(CAMERA_TOPICS)
        self._state_joint_groups = {
            k: v for k, v in STATE_JOINT_GROUPS.items() if not (no_hand and k in HAND_KEYS)
        }
        self._command_joint_groups = {
            k: v for k, v in COMMAND_JOINT_GROUPS.items() if not (no_hand and k in HAND_KEYS)
        }
        self._state: dict[str, np.ndarray | None] = dict.fromkeys(self._state_joint_groups)

        # Diagnostics only -- not used for control logic.
        self._image_msg_counts: dict[str, int] = {}
        self._joint_state_msg_count = 0
        self._controller_state_msg_counts: dict[str, int] = {}

        self._image_subs = [
            self.create_subscription(
                Image, topic, lambda msg, name=cam: self._on_image(msg, name), sensor_qos
            )
            for cam, topic in CAMERA_TOPICS.items()
        ]
        for cam, topic in CAMERA_TOPICS.items():
            self.get_logger().info(f"Subscribed to camera '{cam}': {topic}")

        if mode == "sim":
            self.create_subscription(
                JointState, JOINT_STATE_TOPIC, self._on_joint_state, sensor_qos
            )
            self.get_logger().info(f"Subscribed to: {JOINT_STATE_TOPIC}")

            self.command_pub = self.create_publisher(JointState, JOINT_COMMAND_TOPIC, 10)
            self.get_logger().info(f"Publishing to: {JOINT_COMMAND_TOPIC}")
        else:
            state_topics = dict(CONTROLLER_STATE_TOPICS)
            command_topics = dict(TRAJECTORY_COMMAND_TOPICS)
            if not no_hand:
                # Real hand feedback: same mechanism as the arms (a
                # controller_state topic), not a self-referential seed.
                state_topics.update(HAND_CONTROLLER_STATE_TOPICS)
                command_topics.update(HAND_TRAJECTORY_COMMAND_TOPICS)

            for key, topic in state_topics.items():
                self.create_subscription(
                    JointTrajectoryControllerState,
                    topic,
                    lambda msg, group_key=key: self._on_controller_state(msg, group_key),
                    sensor_qos,
                )
                self.get_logger().info(f"Subscribed to '{key}' feedback: {topic}")

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
            with self._lock:
                self._images[cam_name] = img.copy()
            self._image_msg_counts[cam_name] = self._image_msg_counts.get(cam_name, 0) + 1
        except Exception:
            self.get_logger().error(f"_on_image('{cam_name}') raised:", exc_info=True)

    def _on_joint_state(self, msg: JointState) -> None:
        try:
            positions = dict(zip(msg.name, msg.position, strict=False))
            state = {}
            for key, joint_names in self._state_joint_groups.items():
                missing = [n for n in joint_names if n not in positions]
                if missing:
                    self.get_logger().warn(
                        f"/joint_states missing {missing} for '{key}' "
                        f"(msg has {len(msg.name)} names: {list(msg.name)[:5]}...)",
                        throttle_duration_sec=2.0,
                    )
                    return  # wait for a complete /joint_states message
                state[key] = np.array([positions[n] for n in joint_names], dtype=np.float32)
            with self._lock:
                self._state.update(state)
            self._joint_state_msg_count += 1
        except Exception:
            self.get_logger().error("_on_joint_state raised:", exc_info=True)

    def _on_controller_state(self, msg: JointTrajectoryControllerState, group_key: str) -> None:
        """group_key is any STATE_JOINT_GROUPS key -- an arm (7 joints) or,
        with --with-hand, a hand (1 joint: index_proximal)."""
        try:
            positions = dict(zip(msg.joint_names, msg.feedback.positions, strict=False))
            joint_names = STATE_JOINT_GROUPS[group_key]
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
        """Human-readable snapshot of what's blocking get_observation(), for diagnostics."""
        with self._lock:
            missing_state = [k for k, v in self._state.items() if v is None]
            missing_images = [k for k, v in self._images.items() if v is None]
        return (
            f"missing_state={missing_state} missing_images={missing_images} | "
            f"msg_counts: images={self._image_msg_counts} "
            f"joint_states={self._joint_state_msg_count} "
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

    def publish_command(self, action: dict[str, np.ndarray]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [n for group in self._command_joint_groups.values() for n in group]
        positions = []
        for key in self._command_joint_groups:
            if key in HAND_KEYS:
                # GR00T's action for this key is a single scalar (the trained
                # index_proximal proxy) -- expand it proportionally into a
                # command spanning all 6 real finger joints for this hand.
                scalar = float(np.asarray(action[key], dtype=np.float64).reshape(-1)[0])
                positions.append(compute_hand_finger_positions(scalar))
            else:
                positions.append(np.asarray(action[key], dtype=np.float64))
        msg.position = np.concatenate(positions).tolist()
        self.command_pub.publish(msg)

    def publish_command_real(self, action: dict[str, np.ndarray], dt: float) -> None:
        """Arms: GR00T's per-joint action published as-is (7 joints). Hands (if
        not self._no_hand): GR00T's single index_proximal scalar expanded into
        all 6 real finger joints via compute_hand_finger_positions() -- see
        class docstring."""
        stamp = self.get_clock().now().to_msg()
        time_from_start = Duration(sec=0, nanosec=int(dt * 1e9))
        for key, topic_pub in self._trajectory_pubs.items():
            msg = JointTrajectory()
            msg.header.stamp = stamp
            if key in HAND_KEYS:
                scalar = float(np.asarray(action[key], dtype=np.float64).reshape(-1)[0])
                msg.joint_names = NAMES_L_HAND if key == "left_hand" else NAMES_R_HAND
                point_positions = compute_hand_finger_positions(scalar).tolist()
            else:
                msg.joint_names = STATE_JOINT_GROUPS[key]
                point_positions = np.asarray(action[key], dtype=np.float64).tolist()
            point = JointTrajectoryPoint()
            point.positions = point_positions
            point.time_from_start = time_from_start
            msg.points = [point]
            topic_pub.publish(msg)


def request_action_chunk(
    policy_client: PolicyClient,
    obs: dict[str, np.ndarray],
    lang_key: str,
    task: str,
    action_keys: list[str],
) -> dict[str, np.ndarray]:
    """Build a get_action() request from an observation and return the predicted chunk.

    Shared by the initial synchronous fetch and the background-thread prefetch in run().
    """
    video_dict = {cam: img[None, None, ...] for cam, img in obs["images"].items()}
    state_dict = {key: obs["state"][key][None, None, :] for key in obs["state"]}
    request_data = {
        "video": video_dict,
        "state": state_dict,
        "language": {lang_key: [[task]]},
    }
    t0 = time.perf_counter()
    logger.info("Requesting get_action() from server...")
    response, _info = policy_client.get_action(request_data)
    logger.info("get_action() -> OK (%.2fs)", time.perf_counter() - t0)
    return {key: response[key][0] for key in action_keys}  # (chunk, D) per key


def calculate_latency_compensated_index(
    inference_delay: float, control_freq: float, action_horizon: int
) -> int:
    """Pick the starting index into a freshly-arrived action chunk.

    ``inference_delay`` seconds have already elapsed since the observation the chunk was
    predicted from, so the first ``round(inference_delay * control_freq)`` steps are
    already stale -- skip them instead of replaying the chunk from index 0. Ported from
    GR00T-WholeBodyControl's ``gear_sonic/utils/inference/vla_utils.py``.
    """
    raw_index = round(inference_delay * control_freq)
    return int(np.clip(raw_index, 0, action_horizon - 1))


def should_trigger_new_inference(
    cached_chunk_exists: bool,
    worker_busy: bool,
    time_since_last_inference: float,
    inference_interval: float,
) -> bool:
    """Rate-limit new get_action() requests: at most one in flight, on a fixed cadence."""
    if not cached_chunk_exists:
        return True
    if worker_busy:
        return False
    return time_since_last_inference >= inference_interval


def blend_action_dicts(
    start: dict[str, np.ndarray], end: dict[str, np.ndarray], alpha: float, action_keys: list[str]
) -> dict[str, np.ndarray]:
    """Linearly interpolate between two action dicts, key by key."""
    return {key: (1.0 - alpha) * start[key] + alpha * end[key] for key in action_keys}


def _inference_worker_loop(
    policy_client: PolicyClient,
    obs_provider,
    lang_key: str,
    task: str,
    action_keys: list[str],
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
) -> None:
    """Persistent background thread: owns every get_action() call so the publish loop
    in run() never blocks on the policy server.

    Reads the latest available observation (not necessarily the one that triggered this
    iteration -- freshness matters more than exact correspondence to the trigger), calls
    the policy server, and pushes ``(action_chunk, inference_start_time)`` onto
    ``result_queue`` (maxsize=1; drops a stale unread result rather than blocking, since
    only the newest chunk is ever useful).
    """
    while not stop_event.is_set():
        try:
            inference_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        busy_event.set()
        try:
            obs = obs_provider()
            if obs is None:
                continue
            inference_start_time = time.perf_counter()
            action_chunk = request_action_chunk(policy_client, obs, lang_key, task, action_keys)
            try:
                result_queue.put_nowait((action_chunk, inference_start_time))
            except queue.Full:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
                result_queue.put_nowait((action_chunk, inference_start_time))
        except Exception:
            logger.error("Inference worker raised:", exc_info=True)
        finally:
            busy_event.clear()


def run(args: argparse.Namespace) -> None:
    rclpy.init()
    node = OpenArmGr00tClientNode(mode=args.mode, no_hand=args.no_hand)
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    # NOTE: PolicyClient's REQ socket has no recv timeout configured (timeout_ms is stored
    # but unused), so this call blocks indefinitely if the server isn't already up -- start
    # the policy server (Terminal 1 in the module docstring) before running this client.
    logger.info("Connecting to policy server at %s:%d...", args.host, args.port)
    policy_client = PolicyClient(host=args.host, port=args.port)

    t0 = time.perf_counter()
    logger.info("Sending ping() ...")
    alive = policy_client.ping()
    logger.info("ping() -> %s (%.2fs)", alive, time.perf_counter() - t0)
    if not alive:
        raise RuntimeError(f"Server at {args.host}:{args.port} did not respond to ping()")

    t0 = time.perf_counter()
    logger.info("Sending get_modality_config() ...")
    modality_config = policy_client.get_modality_config()
    logger.info("get_modality_config() -> OK (%.2fs)", time.perf_counter() - t0)

    action_keys = modality_config["action"].modality_keys
    action_chunk_size = len(modality_config["action"].delta_indices)
    lang_key = modality_config["language"].modality_keys[0]

    last_status_log = 0.0
    while node.get_observation() is None:
        now = time.perf_counter()
        if now - last_status_log > 2.0:
            logger.info("Waiting for full observation... %s", node.observation_status())
            last_status_log = now
        time.sleep(0.1)
    first_obs = node.get_observation()
    logger.info("Receiving observations. Starting control loop (task=%r).", args.task)

    # Async inference: a persistent worker thread owns every get_action() call, so the
    # publish loop below never blocks on the policy server. New chunks are swapped in the
    # instant they're available (see calculate_latency_compensated_index), not gated on the
    # previous chunk running out -- if the server is slow, the loop just keeps stepping
    # through (and holding at the end of) the current chunk instead of freezing.
    inference_queue: queue.Queue = queue.Queue(maxsize=1)
    result_queue: queue.Queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()
    inference_worker = threading.Thread(
        target=_inference_worker_loop,
        args=(
            policy_client,
            node.get_observation,
            lang_key,
            args.task,
            action_keys,
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
        ),
        daemon=True,
    )
    inference_worker.start()

    cached_action_chunk: dict[str, np.ndarray] | None = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / args.inference_rate

    # Anchor for interpolation/blending. Needed in both modes now -- chunk-swap blending
    # applies to sim mode too, which previously published targets with no smoothing at all.
    last_published_action: dict[str, np.ndarray] = {
        key: first_obs["state"][key] for key in action_keys
    }

    # Real mode only: openarm_bimanual_controllers.yaml sets interpolation_method: "none"
    # on the arm controllers (a controller_manager running at update_rate: 750Hz), i.e. they
    # expect a high-frequency command stream and do no smoothing of their own. GR00T only
    # produces a new target every --fps (bound by inference latency), so publishing its raw
    # per-step targets directly at --fps looks jerky -- each one is a fresh single-point
    # trajectory with a hard step to the next target. Instead we linearly interpolate between
    # the last commanded target and the new one and stream that at --command-hz.
    command_dt = 1.0 / args.command_hz
    substeps_per_action = max(1, round(args.command_hz / args.fps))
    # Chunk swaps use a longer blend window than a normal same-chunk step transition, since
    # consecutive chunks are independent model predictions and can diverge from wherever the
    # previous one left off (see module docstring / --swap-blend-duration help).
    swap_substeps_real = max(1, round(args.command_hz * args.swap_blend_duration))
    swap_substeps_sim = max(1, round(args.fps * args.swap_blend_duration))

    def publish_blended_real(start, end, n_substeps):
        nonlocal last_published_action
        for sub in range(1, n_substeps + 1):
            sub_loop_start = time.perf_counter()
            alpha = sub / n_substeps
            node.publish_command_real(
                blend_action_dicts(start, end, alpha, action_keys), dt=command_dt
            )
            elapsed = time.perf_counter() - sub_loop_start
            sleep_time = command_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        last_published_action = end

    def publish_blended_sim(start, end, n_substeps):
        nonlocal last_published_action
        sub_dt = 1.0 / args.fps
        for sub in range(1, n_substeps + 1):
            sub_loop_start = time.perf_counter()
            alpha = sub / n_substeps
            node.publish_command(blend_action_dicts(start, end, alpha, action_keys))
            elapsed = time.perf_counter() - sub_loop_start
            sleep_time = sub_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        last_published_action = end

    try:
        while True:
            loop_start = time.perf_counter()
            obs = node.get_observation()
            if obs is None:
                time.sleep(0.01)
                continue

            # Non-blocking: swap in a new chunk the instant it's available, regardless of
            # how much of the current chunk has been consumed.
            just_swapped = False
            try:
                new_chunk, inference_start_time = result_queue.get_nowait()
                inference_delay = time.perf_counter() - inference_start_time
                action_chunk_index = calculate_latency_compensated_index(
                    inference_delay, args.fps, action_chunk_size
                )
                cached_action_chunk = new_chunk
                last_inference_time = time.perf_counter()
                just_swapped = True
                logger.info(
                    "New action chunk (latency=%.2fs, start_index=%d/%d)",
                    inference_delay,
                    action_chunk_index,
                    action_chunk_size,
                )
            except queue.Empty:
                pass

            if should_trigger_new_inference(
                cached_action_chunk is not None,
                inference_busy_event.is_set(),
                time.perf_counter() - last_inference_time,
                inference_interval,
            ):
                try:
                    inference_queue.put_nowait(None)
                except queue.Full:
                    pass

            if cached_action_chunk is None:
                logger.info("No action chunk yet, waiting for first inference...")
                time.sleep(0.05)
                continue

            action = {key: cached_action_chunk[key][action_chunk_index] for key in action_keys}

            if args.mode == "real":
                n_substeps = swap_substeps_real if just_swapped else substeps_per_action
                publish_blended_real(last_published_action, action, n_substeps)
            elif just_swapped:
                publish_blended_sim(last_published_action, action, swap_substeps_sim)
            else:
                node.publish_command(action)
                last_published_action = action
                elapsed = time.perf_counter() - loop_start
                sleep_time = (1.0 / args.fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            action_chunk_index = min(action_chunk_index + 1, action_chunk_size - 1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        inference_stop_event.set()
        inference_worker.join(timeout=1.0)
        policy_client.close()
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenArm ROS2 <-> GR00T policy-server client")
    parser.add_argument(
        "--mode",
        choices=["sim", "real"],
        default="sim",
        help="'sim': /joint_states + flat /joint_command (Isaac Sim). "
        "'real': per-arm controller_state feedback + per-arm joint_trajectory commands "
        "(ros2_control on real hardware); unless --no-hand is passed, hand "
        "feedback/commands use the same mechanism via the Revo2 hand controller topics.",
    )
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="For checkpoints trained on an arms-only modality config (no left_hand/"
        "right_hand keys, e.g. examples/openarm_revo2_arms_only_config.py). Skips all "
        "hand state reading and command publishing -- required for such checkpoints, "
        "since they never return left_hand/right_hand in their action dict.",
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task instruction")
    parser.add_argument("--host", type=str, default="localhost", help="Policy server host")
    parser.add_argument("--port", type=int, default=5555, help="Policy server port")
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Rate at which GR00T action-chunk steps are consumed (matches the training "
        "data's native timestep rate). In real mode this is decoupled from the actual "
        "wire publish rate -- see --command-hz.",
    )
    parser.add_argument(
        "--command-hz",
        type=float,
        default=250.0,
        help="Real mode only: rate at which linearly-interpolated joint_trajectory commands "
        "are streamed to the arm controllers. openarm_bimanual_controllers.yaml runs "
        "the arm controllers with interpolation_method: none at update_rate: 750Hz, i.e. "
        "they expect a high-frequency command stream and do no smoothing themselves -- "
        "publishing GR00T's raw per-step targets directly at --fps (e.g. 30Hz) looks jerky.",
    )
    parser.add_argument(
        "--inference-rate",
        type=float,
        default=2.0,
        help="Max rate (Hz) at which new get_action() requests are triggered on the "
        "background inference thread. Only one request is ever in flight; the publish "
        "loop keeps consuming the current action chunk (holding at its last step if "
        "needed) and is never blocked waiting for a response, regardless of this rate.",
    )
    parser.add_argument(
        "--swap-blend-duration",
        type=float,
        default=0.1,
        help="Seconds to linearly blend from the last published action to a freshly "
        "swapped-in action chunk's (latency-compensated) start action. A new chunk is a "
        "fresh, independent model prediction and can diverge from wherever the previous "
        "chunk left off -- this ramp avoids a visible jerk at the swap. 0 = snap instantly.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
