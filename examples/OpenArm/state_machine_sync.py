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
Synchronous variant of state_machine.py: HOME/PREPARE/LIFT/PLACE flow and
Enter/'q' control semantics are identical (see state_machine.py's docstring),
but the LIFT/PLACE control loop itself is fundamentally different.

state_machine.py's Gr00tSession runs a background inference worker thread
that keeps requesting fresh action chunks while the publish loop keeps
consuming/blending the current chunk -- new chunks swap in as soon as
they're ready (calculate_latency_compensated_index skips the steps that went
stale during the request), so motion never pauses waiting on the server.

This file's SyncGr00tSession instead: requests one action chunk with a
single BLOCKING get_action() call, executes exactly --steps-per-chunk (16 by
default) steps from it, then blocks again on a fresh get_action() call
before continuing -- no background worker thread, no queues, no
latency-compensated index selection, no prefetching while the current chunk
is still running. Simpler and more predictable (each 16-step segment is
always index 0..15 of a chunk requested right before it started), at the
cost of a visible pause (one inference round-trip, whatever that takes) at
every chunk boundary instead of a continuous seamless stream.

A chunk swap is still smoothed with the same --swap-blend-duration ramp
state_machine.py uses (the first step of every freshly-fetched chunk is an
independent model prediction and can diverge from wherever the previous
chunk left off); the following steps within a chunk step at the plain
--fps-derived rate.

This file does NOT modify ros2_gr00t_client.py or state_machine.py, and
reuses state_machine.py's wait_for_enter / check_quit_nonblocking /
move_to_prepare / LIFT_TASK / PLACE_TASK / DEFAULT_PREPARE_CSV as-is.

Prerequisite: same as state_machine.py -- a GR00T policy server already
running on a checkpoint matching --no-hand/--with-hand.

Usage:
    cd examples/OpenArm && source .venv-ros/bin/activate
    python3 state_machine_sync.py --host <server-host>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import threading
import time
from typing import Callable

import rclpy
from ros2_gr00t_client import OpenArmGr00tClientNode, blend_action_dicts, request_action_chunk
from server_client import PolicyClient
from state_machine import (
    DEFAULT_PREPARE_CSV,
    LIFT_TASK,
    PLACE_TASK,
    check_quit_nonblocking,
    move_to_prepare,
    wait_for_enter,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyncGr00tSession:
    """Owns one policy-server connection, reused across sequential task
    phases (LIFT, then PLACE) with no reconnect -- see module docstring for
    how its control loop differs from state_machine.py's Gr00tSession."""

    def __init__(
        self,
        host: str,
        port: int,
        no_hand: bool,
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
        self.last_published_action: dict = {
            key: first_obs["state"][key] for key in self.action_keys
        }

    def run_task_phase(
        self,
        task: str,
        should_switch: Callable[[], bool] | None = None,
        switch_trigger_desc: str = "press 'q' to switch to the other task",
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
                logger.info("STATE: task=%r finished (%s) -- switching.", task, switch_trigger_desc)
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
                        "STATE: task=%r finished mid-chunk (%s) -- switching.",
                        task,
                        switch_trigger_desc,
                    )
                    return
                action = {key: action_chunk[key][step] for key in self.action_keys}
                n_substeps = swap_substeps if step == 0 else substeps_per_action
                publish_blended(self.last_published_action, action, n_substeps)

    def log_summary(self) -> None:
        """Total number of times each task ran and its average completion
        time (phase entry to 'q'/switch-trigger) -- logged once, right before
        exit (see main()'s KeyboardInterrupt handler). A phase still running
        when Ctrl+C hits is not counted (never reached its switch point)."""
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
        description="OpenArm lift/place state machine (synchronous inference): "
        "HOME -> PREPARE -> LIFT -> PLACE, 'q' stops each phase. Unlike "
        "state_machine.py, executes --steps-per-chunk steps then blocks on the "
        "next get_action() call instead of continuously prefetching/blending."
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
    parser.add_argument("--host", type=str, default="localhost", help="Policy server host")
    parser.add_argument("--port", type=int, default=5555, help="Policy server port")
    parser.add_argument(
        "--no-hand",
        dest="no_hand",
        action="store_true",
        default=True,
        help="Arms-only checkpoint (default: on).",
    )
    parser.add_argument(
        "--with-hand",
        dest="no_hand",
        action="store_false",
        help="Use if the server checkpoint has left_hand/right_hand "
        "(overrides the --no-hand default).",
    )
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

    if not wait_for_enter("\nPress Enter to move HOME -> PREPARE, or 'q' to quit."):
        return
    move_to_prepare(args.prepare_csv, args.prepare_duration, args.prepare_command_hz)

    if not wait_for_enter(f"\nPress Enter to start LIFT: {LIFT_TASK!r}, or 'q' to quit."):
        return
    session = SyncGr00tSession(
        args.host,
        args.port,
        args.no_hand,
        args.fps,
        args.command_hz,
        args.swap_blend_duration,
        args.steps_per_chunk,
    )
    try:
        # LIFT <-> PLACE toggle loop: 'q' during either one stops it and
        # immediately starts the other, looping forever. Ctrl+C is the only
        # way out.
        task = LIFT_TASK
        while True:
            session.run_task_phase(task)
            task = PLACE_TASK if task == LIFT_TASK else LIFT_TASK
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        session.log_summary()
    finally:
        session.close()


if __name__ == "__main__":
    main()
