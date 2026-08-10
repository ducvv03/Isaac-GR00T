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
Fully-automatic variant of state_machine.py: no keyboard input at all once
launched -- only Ctrl+C stops it. Every Enter/'q' gate is replaced with an
automatic trigger:

  HOME (0)
    |  (auto, starts immediately -- no Enter)
    v
  PREPARE move (prepare.csv)  -- runs to completion
    |  (auto, as soon as the PREPARE move finishes)
    v
  LIFT ("Lift...")   <--[arms held still for --stability-window]--  switches to PLACE
    |                                                                          ^
    '--[arms held still for --stability-window]--> switches to PLACE ---------'
       (starts as LIFT)

"Held still" = a JointStabilityMonitor watches the robot's real left_arm/
right_arm feedback (not the commanded action, which is always changing at
--fps even when nearly stationary) and considers the current phase done once
every sample in the trailing --stability-window seconds is within
--stability-threshold radians of the newest sample -- i.e. GR00T has settled
into a hold (lift completed and holding, or place completed and released)
rather than still actively moving.

A --stability-grace-period at the start of each phase is excluded from the
monitor: the robot is at (near) rest right when a phase starts (holding the
end of the previous motion, or freshly arrived at the prepare pose), which
would otherwise trip an instant false "stable" before GR00T has even begun
the new motion.

This file does NOT modify ros2_gr00t_client.py, and reuses state_machine.py's
Gr00tSession / move_to_prepare / LIFT_TASK / PLACE_TASK as-is --
Gr00tSession.run_task_phase() takes an optional `should_switch` predicate
(defaulting to state_machine.py's original check_quit_nonblocking behavior),
so this file plugs in stability-based switching instead of duplicating the
async-inference control loop.

--stability-window (0.5s per the initial spec), --stability-threshold, and
--stability-grace-period are NOT verified against real motion/noise
characteristics -- tune them for your hardware before trusting the defaults
(same caveat as state_machine.py's --prepare-duration).

Each completed phase logs its own wall-clock duration (task1=LIFT, task2=PLACE).
On Ctrl+C, prints a summary: number of completed LIFT+PLACE cycles, and each
task's run count / average / min / max duration.

Prerequisite: same as state_machine.py -- a GR00T policy server already
running on an arms-only checkpoint (e.g. examples/openarm_revo2_arms_only_config.py),
matching --no-hand.

Usage:
    cd examples/OpenArm && source .venv-ros/bin/activate
    python3 auto_state_machine.py --host <server-host>
"""

from __future__ import annotations

import argparse
import collections
import logging
from pathlib import Path
import time

import numpy as np
from state_machine import DEFAULT_PREPARE_CSV, LIFT_TASK, PLACE_TASK, Gr00tSession, move_to_prepare


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JointStabilityMonitor:
    """Detects when the robot's arm joints have stopped changing meaningfully.

    Call start_phase() when a new task phase begins, then update() with the
    latest real joint-state vector on every control-loop tick. is_stable()
    returns True once: (a) --grace_period_s has elapsed since start_phase(),
    (b) the sample window spans at least --window_s, and (c) every sample in
    that window is within --threshold (radians, per-joint max) of the newest
    sample -- i.e. the arms have been holding roughly the same pose for the
    whole window, not just at one instant.
    """

    def __init__(self, window_s: float, threshold: float, grace_period_s: float):
        self._window_s = window_s
        self._threshold = threshold
        self._grace_period_s = grace_period_s
        self._samples: collections.deque[tuple[float, np.ndarray]] = collections.deque()
        self._phase_start: float | None = None

    def start_phase(self) -> None:
        self._samples.clear()
        self._phase_start = time.perf_counter()

    def update(self, joint_vec: np.ndarray) -> None:
        now = time.perf_counter()
        self._samples.append((now, joint_vec))
        cutoff = now - self._window_s
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def is_stable(self) -> bool:
        now = time.perf_counter()
        if self._phase_start is None or now - self._phase_start < self._grace_period_s:
            return False
        if len(self._samples) < 2:
            return False
        oldest_t, _ = self._samples[0]
        newest_t, newest_vec = self._samples[-1]
        if newest_t - oldest_t < self._window_s:
            return False  # window not full yet
        max_delta = max(float(np.max(np.abs(vec - newest_vec))) for _, vec in self._samples)
        return max_delta <= self._threshold


def log_summary(phase_durations: dict[str, list[float]], task_label: dict[str, str]) -> None:
    """Printed on Ctrl+C: how many LIFT+PLACE cycles completed and the average
    (min/max) wall-clock duration of each task phase. A "cycle" is one
    completed LIFT followed by one completed PLACE -- min(count) across the
    two, since they strictly alternate starting with LIFT."""
    counts = {task: len(durations) for task, durations in phase_durations.items()}
    cycles = min(counts.values()) if counts else 0
    logger.info("=" * 60)
    logger.info("SUMMARY: %d LIFT+PLACE cycle(s) completed", cycles)
    for task, durations in phase_durations.items():
        if not durations:
            logger.info("  %s: ran 0 times", task_label[task])
            continue
        avg = sum(durations) / len(durations)
        logger.info(
            "  %s: ran %d time(s), avg %.2fs (min %.2fs, max %.2fs)",
            task_label[task],
            len(durations),
            avg,
            min(durations),
            max(durations),
        )
    logger.info("=" * 60)


def make_stability_switch(session: Gr00tSession, monitor: JointStabilityMonitor):
    """Builds the `should_switch` predicate passed to Gr00tSession.run_task_phase():
    samples the robot's real arm state (session.node.get_observation(), not the
    commanded action) into `monitor` on every call, then reports whether it has
    settled into a hold."""

    def should_switch() -> bool:
        obs = session.node.get_observation()
        if obs is not None:
            joint_vec = np.concatenate(
                [np.asarray(obs["state"][k], dtype=np.float64) for k in session.action_keys]
            )
            monitor.update(joint_vec)
        return monitor.is_stable()

    return should_switch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenArm lift/place fully-automatic state machine: "
        "HOME -> PREPARE -> LIFT <-> PLACE, switching phases automatically when "
        "the arms hold still, no keyboard input except Ctrl+C to stop."
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
        help="Publish rate for the HOME->PREPARE move (interpolation_method: none "
        "needs a high-frequency command stream, see ros2_gr00t_client.py's --command-hz).",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Policy server host")
    parser.add_argument("--port", type=int, default=5555, help="Policy server port")
    parser.add_argument(
        "--no-hand",
        dest="no_hand",
        action="store_true",
        default=True,
        help="Arms-only checkpoint (default: on, matches "
        "examples/openarm_revo2_arms_only_config.py).",
    )
    parser.add_argument(
        "--with-hand",
        dest="no_hand",
        action="store_false",
        help="Use if the server checkpoint actually has left_hand/right_hand "
        "(overrides the --no-hand default).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Must match the checkpoint's training data fps (20 for the real "
        "arms-only dataset in openarm_revo2_arms_only_config.py).",
    )
    parser.add_argument("--command-hz", type=float, default=250.0, help="GR00T-phase publish rate")
    parser.add_argument("--inference-rate", type=float, default=2.0, help="Max get_action() Hz")
    parser.add_argument(
        "--swap-blend-duration", type=float, default=0.1, help="Chunk-swap blend seconds"
    )
    parser.add_argument(
        "--stability-window",
        type=float,
        default=0.5,
        help="Seconds the arms must hold within --stability-threshold of each "
        "other before a phase is considered done and switches to the other task.",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.02,
        help="Max per-joint radian deviation across --stability-window to still "
        "count as 'holding still'. NOT verified against real sensor/prediction "
        "noise -- widen if phases never switch, narrow if they switch too early.",
    )
    parser.add_argument(
        "--stability-grace-period",
        type=float,
        default=1.5,
        help="Seconds at the start of each phase where stability is NOT checked, "
        "so the near-rest pose the robot starts a phase in doesn't immediately "
        "look 'stable' before GR00T has begun the new motion.",
    )
    args = parser.parse_args()

    logger.info("AUTO: moving HOME -> PREPARE (no Enter gate).")
    move_to_prepare(args.prepare_csv, args.prepare_duration, args.prepare_command_hz)

    logger.info("AUTO: starting LIFT <-> PLACE loop (Ctrl+C to stop).")
    session = Gr00tSession(
        args.host,
        args.port,
        args.no_hand,
        args.fps,
        args.command_hz,
        args.inference_rate,
        args.swap_blend_duration,
    )
    monitor = JointStabilityMonitor(
        args.stability_window, args.stability_threshold, args.stability_grace_period
    )
    task_label = {LIFT_TASK: "task1 (LIFT)", PLACE_TASK: "task2 (PLACE)"}
    phase_durations: dict[str, list[float]] = {LIFT_TASK: [], PLACE_TASK: []}
    try:
        task = LIFT_TASK
        while True:
            monitor.start_phase()
            switch_desc = (
                f"auto: arms held within {args.stability_threshold:.3f} rad "
                f"for {args.stability_window:.1f}s"
            )
            phase_start = time.perf_counter()
            session.run_task_phase(
                task,
                should_switch=make_stability_switch(session, monitor),
                switch_trigger_desc=switch_desc,
            )
            phase_duration = time.perf_counter() - phase_start
            phase_durations[task].append(phase_duration)
            logger.info("STATE: %s took %.2fs", task_label[task], phase_duration)
            task = PLACE_TASK if task == LIFT_TASK else LIFT_TASK
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        log_summary(phase_durations, task_label)
        session.close()


if __name__ == "__main__":
    main()
