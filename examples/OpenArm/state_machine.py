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
State machine for the OpenArm lift/place demo on real hardware. Every phase
is gated by Enter and stoppable early with 'q' (type it, then press Enter --
this is line-buffered stdin, not raw single-keystroke input):

  HOME (0)
    |  [Enter]                          ['q' any time before Enter -> quit]
    v
  PREPARE move (prepare.csv)  --['q' during move]--> stop early, hold position
    |  [Enter]
    v
  LIFT ("Lift...")            --['q' during LIFT]--> stop this phase
    |  [Enter]
    v
  PLACE ("Place...")          --['q' during PLACE]--> stop this phase, exit

So the full sequence is: Enter -> move to prepare (q stops the move) ->
Enter -> LIFT (q stops it) -> Enter -> PLACE (q stops it, program ends).
'q' pressed while waiting for Enter (instead of pressing Enter) quits
immediately without starting that phase. Ctrl+C works as a hard-stop at any
point too.

State 1 (HOME -> PREPARE): streams interpolated trajectory_msgs/JointTrajectory
commands (same topics/joint order as ros2_gr00t_client.py's real mode) from the
zero/home pose through every waypoint in prepare.csv, at --prepare-command-hz.
Precondition: the robot is actually at the zero/home pose when this starts.

State 2 (LIFT, then PLACE): connects once to a running GR00T policy server
(same as ros2_gr00t_client.py --mode real --no-hand) and keeps the connection
and inference worker alive across both task phases -- only the live-mutable
task instruction changes between them, no reconnect.

This file does NOT modify ros2_gr00t_client.py. It imports that file's reusable
building blocks (OpenArmGr00tClientNode, request_action_chunk,
calculate_latency_compensated_index, should_trigger_new_inference,
blend_action_dicts, STATE_JOINT_GROUPS, TRAJECTORY_COMMAND_TOPICS) and
implements its own control loop with a mutable task, since
ros2_gr00t_client.py's task is a fixed string closed over by its inference
worker thread at startup -- there is no existing way to change it live there.

Prerequisite: a GR00T policy server already running on an arms-only checkpoint
(e.g. examples/openarm_revo2_arms_only_config.py), matching --no-hand.

Usage:
    cd examples/OpenArm && source .venv-ros/bin/activate
    python3 state_machine.py --host <server-host>
"""

from __future__ import annotations

import argparse
import csv
import logging
import queue
import select
import sys
import threading
import time
from pathlib import Path

from builtin_interfaces.msg import Duration
import numpy as np
import rclpy
from rclpy.node import Node
from server_client import PolicyClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ros2_gr00t_client import (
    OpenArmGr00tClientNode,
    STATE_JOINT_GROUPS,
    TRAJECTORY_COMMAND_TOPICS,
    blend_action_dicts,
    calculate_latency_compensated_index,
    request_action_chunk,
    should_trigger_new_inference,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIFT_TASK = "Lift the silver-gray rectangular metal box with both hands and hold it steady."
PLACE_TASK = "Place the silver-gray rectangular metal box back on the table with both hands."

DEFAULT_PREPARE_CSV = Path(__file__).parent / "prepare.csv"


# ============================================================================
# stdin helpers -- shared by both states
# ============================================================================


def wait_for_enter(prompt: str) -> bool:
    """Blocking wait for a line on stdin. Returns True on blank Enter (proceed),
    False on 'q' (quit before starting). Other input is ignored, re-prompts."""
    print(prompt)
    while True:
        try:
            line = input()
        except EOFError:
            return False
        stripped = line.strip().lower()
        if stripped == "q":
            return False
        if stripped == "":
            return True
        # ignore anything else and keep waiting


def check_quit_nonblocking() -> bool:
    """Non-blocking: True if a 'q' line is available on stdin right now. Safe to
    call from inside a tight timed loop (publish loops, etc.) -- never blocks."""
    while select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        if line.strip().lower() == "q":
            return True
        # non-'q' input during a running phase is ignored; keep draining stdin
        # in case more than one line queued up.
    return False


# ============================================================================
# State 1: HOME -> PREPARE
# ============================================================================


def load_prepare_waypoints(csv_path: Path) -> list[dict[str, np.ndarray]]:
    """Read prepare.csv into a list of {"left_arm": (7,), "right_arm": (7,)}
    waypoints, in file order. segment_idx is recording metadata, not used for
    playback -- the rows already form a dense, smooth, monotonic trajectory."""
    left_cols = STATE_JOINT_GROUPS["left_arm"]
    right_cols = STATE_JOINT_GROUPS["right_arm"]
    waypoints = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get(left_cols[0]):
                continue  # skip blank trailing rows
            waypoints.append(
                {
                    "left_arm": np.array([float(row[c]) for c in left_cols], dtype=np.float64),
                    "right_arm": np.array([float(row[c]) for c in right_cols], dtype=np.float64),
                }
            )
    return waypoints


def _stream_segment(
    pubs: dict[str, object],
    node: Node,
    start: dict[str, np.ndarray],
    end: dict[str, np.ndarray],
    duration_s: float,
    command_hz: float,
) -> bool:
    """Linearly interpolate start -> end and stream single-point JointTrajectory
    commands at command_hz -- same streaming pattern as ros2_gr00t_client.py's
    publish_command_real (the arm controllers run interpolation_method: none and
    expect a high-frequency command stream, not sparse waypoints).

    Returns False (stopping mid-segment, holding the last published position) if
    'q' was pressed; True if the segment completed."""
    n_substeps = max(1, round(duration_s * command_hz))
    dt = 1.0 / command_hz
    time_from_start = Duration(sec=0, nanosec=int(dt * 1e9))
    for sub in range(1, n_substeps + 1):
        if check_quit_nonblocking():
            return False
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
    return True


def move_to_prepare(csv_path: Path, duration_s: float, command_hz: float) -> None:
    """STATE: HOME -> PREPARE. Assumes the robot is currently at the zero/home
    pose (this script's precondition) and streams through every waypoint in
    prepare.csv, splitting duration_s evenly across all
    home->row0->row1->...->row_last segments. Stops early (holding position)
    if 'q' is pressed."""
    waypoints = load_prepare_waypoints(csv_path)
    if not waypoints:
        raise RuntimeError(f"No waypoints parsed from {csv_path}")
    logger.info(
        "STATE HOME -> PREPARE: %d waypoints from %s over %.1fs (press 'q' to stop early)",
        len(waypoints), csv_path, duration_s,
    )

    home = {"left_arm": np.zeros(7, dtype=np.float64), "right_arm": np.zeros(7, dtype=np.float64)}
    sequence = [home, *waypoints]
    segment_duration = duration_s / (len(sequence) - 1)

    rclpy.init()
    node = Node("openarm_prepare_move")
    pubs = {
        key: node.create_publisher(JointTrajectory, topic, 10)
        for key, topic in TRAJECTORY_COMMAND_TOPICS.items()
    }
    time.sleep(0.5)  # let publishers connect before the first command
    try:
        for i in range(len(sequence) - 1):
            if not _stream_segment(pubs, node, sequence[i], sequence[i + 1], segment_duration, command_hz):
                logger.info("STATE HOME -> PREPARE: stopped early by user (q) at waypoint %d/%d", i, len(sequence) - 1)
                return
            logger.info("  waypoint %d/%d reached", i + 1, len(sequence) - 1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    logger.info("STATE HOME -> PREPARE: done, at prepare position.")


# ============================================================================
# State 2: LIFT, then PLACE (single connection, task changes between phases)
# ============================================================================


class TaskState:
    """Thread-safe mutable holder for the current task instruction, read fresh
    by the inference worker on every get_action() call instead of being a fixed
    string closed over at thread start."""

    def __init__(self, task: str):
        self._lock = threading.Lock()
        self._task = task

    def get(self) -> str:
        with self._lock:
            return self._task

    def set(self, task: str) -> None:
        with self._lock:
            self._task = task


def _inference_worker_loop(
    policy_client: PolicyClient,
    obs_provider,
    lang_key: str,
    task_state: TaskState,
    action_keys: list[str],
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
) -> None:
    """Same architecture as ros2_gr00t_client.py's _inference_worker_loop, except
    the task is read fresh from task_state each call rather than being fixed."""
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
            action_chunk = request_action_chunk(
                policy_client, obs, lang_key, task_state.get(), action_keys
            )
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


class Gr00tSession:
    """Owns one policy-server connection + inference worker, reused across
    multiple sequential task phases (LIFT, then PLACE) with no reconnect --
    only run_task_phase()'s task argument changes between calls."""

    def __init__(self, host: str, port: int, no_hand: bool, fps: float, command_hz: float,
                 inference_rate: float, swap_blend_duration: float):
        self.fps = fps
        self.command_hz = command_hz
        self.inference_rate = inference_rate
        self.swap_blend_duration = swap_blend_duration

        rclpy.init()
        self.node = OpenArmGr00tClientNode(mode="real", no_hand=no_hand)
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
        self.last_published_action: dict[str, np.ndarray] = {
            key: first_obs["state"][key] for key in self.action_keys
        }

        self.task_state = TaskState(LIFT_TASK)
        self.inference_queue: queue.Queue = queue.Queue(maxsize=1)
        self.result_queue: queue.Queue = queue.Queue(maxsize=1)
        self.inference_stop_event = threading.Event()
        self.inference_busy_event = threading.Event()
        self.inference_worker = threading.Thread(
            target=_inference_worker_loop,
            args=(
                self.policy_client,
                self.node.get_observation,
                self.lang_key,
                self.task_state,
                self.action_keys,
                self.inference_queue,
                self.result_queue,
                self.inference_stop_event,
                self.inference_busy_event,
            ),
            daemon=True,
        )
        self.inference_worker.start()

    def run_task_phase(self, task: str) -> None:
        """Runs the control loop under `task` until 'q' is pressed. Returns
        (does not quit the process) -- caller decides what happens next."""
        self.task_state.set(task)
        logger.info("STATE: running task=%r (press 'q' to stop this phase)", task)
        try:
            self.inference_queue.put_nowait(None)  # force a fresh inference under this task
        except queue.Full:
            pass

        cached_action_chunk: dict[str, np.ndarray] | None = None
        action_chunk_index = 0
        last_inference_time = 0.0
        inference_interval = 1.0 / self.inference_rate
        command_dt = 1.0 / self.command_hz
        substeps_per_action = max(1, round(self.command_hz / self.fps))
        swap_substeps = max(1, round(self.command_hz * self.swap_blend_duration))

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
            if check_quit_nonblocking():
                logger.info("STATE: task=%r stopped by user (q).", task)
                return

            just_swapped = False
            try:
                new_chunk, inference_start_time = self.result_queue.get_nowait()
                inference_delay = time.perf_counter() - inference_start_time
                action_chunk_index = calculate_latency_compensated_index(
                    inference_delay, self.fps, self.action_chunk_size
                )
                cached_action_chunk = new_chunk
                last_inference_time = time.perf_counter()
                just_swapped = True
                logger.info(
                    "New action chunk (latency=%.2fs, start_index=%d/%d)",
                    inference_delay, action_chunk_index, self.action_chunk_size,
                )
            except queue.Empty:
                pass

            if should_trigger_new_inference(
                cached_action_chunk is not None,
                self.inference_busy_event.is_set(),
                time.perf_counter() - last_inference_time,
                inference_interval,
            ):
                try:
                    self.inference_queue.put_nowait(None)
                except queue.Full:
                    pass

            if cached_action_chunk is None:
                time.sleep(0.05)
                continue

            action = {
                key: cached_action_chunk[key][action_chunk_index] for key in self.action_keys
            }
            n_substeps = swap_substeps if just_swapped else substeps_per_action
            publish_blended(self.last_published_action, action, n_substeps)
            action_chunk_index = min(action_chunk_index + 1, self.action_chunk_size - 1)

    def close(self) -> None:
        self.inference_stop_event.set()
        self.inference_worker.join(timeout=1.0)
        self.policy_client.close()
        self.node.destroy_node()
        rclpy.shutdown()


# ============================================================================
# Entry point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenArm lift/place state machine: Enter -> PREPARE -> Enter -> LIFT -> Enter -> PLACE, 'q' stops each phase"
    )
    parser.add_argument(
        "--prepare-csv", type=Path, default=DEFAULT_PREPARE_CSV,
        help="CSV of waypoints from home to prepare position",
    )
    parser.add_argument(
        "--prepare-duration", type=float, default=8.0,
        help="Total seconds to move from home through prepare.csv to the prepare "
        "position. Not verified against real motion limits -- tune conservatively "
        "for your hardware before trusting this default.",
    )
    parser.add_argument(
        "--prepare-command-hz", type=float, default=250.0,
        help="Publish rate for the HOME->PREPARE move (same rationale as "
        "ros2_gr00t_client.py's --command-hz: interpolation_method: none needs a "
        "high-frequency command stream).",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Policy server host")
    parser.add_argument("--port", type=int, default=5555, help="Policy server port")
    parser.add_argument(
        "--no-hand", dest="no_hand", action="store_true", default=True,
        help="Arms-only checkpoint (default: on, matches "
        "examples/openarm_revo2_arms_only_config.py).",
    )
    parser.add_argument(
        "--with-hand", dest="no_hand", action="store_false",
        help="Use if the server checkpoint actually has left_hand/right_hand "
        "(overrides the --no-hand default).",
    )
    parser.add_argument(
        "--fps", type=float, default=20.0,
        help="Must match the checkpoint's training data fps (20 for the real "
        "arms-only dataset in openarm_revo2_arms_only_config.py).",
    )
    parser.add_argument("--command-hz", type=float, default=250.0, help="GR00T-phase publish rate")
    parser.add_argument("--inference-rate", type=float, default=2.0, help="Max get_action() Hz")
    parser.add_argument("--swap-blend-duration", type=float, default=0.1, help="Chunk-swap blend seconds")
    args = parser.parse_args()

    if not wait_for_enter("\nPress Enter to move HOME -> PREPARE, or 'q' to quit."):
        return
    move_to_prepare(args.prepare_csv, args.prepare_duration, args.prepare_command_hz)

    if not wait_for_enter(f"\nPress Enter to start LIFT: {LIFT_TASK!r}, or 'q' to quit."):
        return
    session = Gr00tSession(
        args.host, args.port, args.no_hand, args.fps,
        args.command_hz, args.inference_rate, args.swap_blend_duration,
    )
    try:
        session.run_task_phase(LIFT_TASK)

        if not wait_for_enter(f"\nPress Enter to start PLACE: {PLACE_TASK!r}, or 'q' to quit."):
            return
        session.run_task_phase(PLACE_TASK)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        session.close()


if __name__ == "__main__":
    main()
