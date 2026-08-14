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
Dataset-driven chunked inference, executed on real hardware. Combines two
existing pieces of this repo that were previously separate:

- gr00t/eval/open_loop_eval.py's stepping pattern: state (and here, images)
  for the model's input at step t always comes from the RECORDED episode,
  never from the robot's actual resulting state -- so the model never sees
  compounding drift from its own prior predictions. That script only
  computes MSE/MAE against the recorded actions and plots a comparison; it
  never touches real hardware.
- replay_ground_truth.py's real-hardware publishing (stream_real_segment,
  TRAJECTORY_COMMAND_TOPICS/HAND_TRAJECTORY_COMMAND_TOPICS, interpolated
  streaming at --command-hz).

This script's loop, for chunk size C (--steps-per-chunk, default 16):
    t = 0
    while t < num_frames:
        obs = {state: episode row t, images: episode video frame t}  (NOT live camera/joint feedback)
        action_chunk = policy_server.get_action(obs)   # single blocking call, C+ steps
        for step in range(C):
            publish action_chunk[step] to the real robot (interpolated at --command-hz)
        t += C
        # next iteration's obs comes from episode row t (=old t + C), regardless of
        # where the robot's actual feedback ended up after executing the chunk

So the model's inputs follow the recorded reference trajectory exactly (like
open_loop_eval.py), while its predicted actions are actually executed on the
real robot (like replay_ground_truth.py) -- useful for judging the quality of
the model's action predictions on real hardware in isolation from state-drift
compounding, which run_state_machine.py-style live-feedback control does not
give you.

Images come from the episode's own recorded video files (decoded with
OpenCV, frame index == row index), not the live camera topics -- consistent
with state also coming from the recording, not live feedback.

The language instruction sent with each chunk request is read fresh from
that chunk's starting frame's task_index (meta/tasks.jsonl), NOT fixed from
frame 0 -- some recorded episodes concatenate multiple task segments
back-to-back in a single file (e.g. a "Place" demo followed by a "Lift" demo
in the same episode, observed in
raw_data/lerobot_v2_data_filtered/20260806/001/episode_000000.parquet: 304
frames, task_index switches 1->0 once at frame 148/149). A fixed
frame-0-derived instruction would silently go stale once t crosses into a
later segment, feeding the model a state/instruction pair that never
co-occurred in training.

Stops after reaching the end of the episode (does not loop).

Talks to a running gr00t/eval/run_gr00t_server.py over ZMQ, same as
ros2_gr00t_client.py -- no `gr00t` package import needed here.

Prerequisite: a GR00T policy server already running on a checkpoint matching
--with-hand (14-dim arms-only by default, 16-dim with --with-hand).

Usage:
    cd examples/OpenArm && source .venv-ros/bin/activate
    python3 replay_chunked_inference.py --with-hand \
        --episode-path /home/ws/data/real/raw_data/lerobot_v2_data_filtered/20260806/001/data/chunk-000/episode_000000.parquet \
        --host <server-host>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from replay_ground_truth import split_state_row, stream_real_segment
from ros2_gr00t_client import (
    HAND_TRAJECTORY_COMMAND_TOPICS,
    TRAJECTORY_COMMAND_TOPICS,
    request_action_chunk,
)
from server_client import PolicyClient
from trajectory_msgs.msg import JointTrajectory


def load_episode_meta(episode_path: Path) -> tuple[Path, dict[str, Path], dict[int, str]]:
    """Returns (episode_root, {camera_key: video_path}, {task_index: task_instruction}).
    episode_path is .../<episode_root>/data/chunk-000/episode_XXXXXX.parquet."""
    episode_root = episode_path.parents[2]
    ep_name = episode_path.stem

    modality = json.loads((episode_root / "meta/modality.json").read_text())
    video_paths = {
        cam: episode_root / "videos/chunk-000" / info["original_key"] / f"{ep_name}.mp4"
        for cam, info in modality["video"].items()
    }

    tasks: dict[int, str] = {}
    with open(episode_root / "meta/tasks.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            tasks[rec["task_index"]] = rec["task"]

    return episode_root, video_paths, tasks


def read_frame(cap: cv2.VideoCapture, frame_idx: int, cam: str) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx} from '{cam}' video")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset-driven chunked inference: model input at every step comes "
        "from the recorded episode (state + images), predicted action chunks are "
        "actually executed on the real robot. See module docstring."
    )
    parser.add_argument(
        "--episode-path", type=Path, required=True, help="Path to episode_XXXXXX.parquet"
    )
    parser.add_argument("--host", type=str, default="localhost", help="Policy server host")
    parser.add_argument("--port", type=int, default=5555, help="Policy server port")
    parser.add_argument(
        "--with-hand",
        action="store_true",
        help="Episode is 16-dim (arms + hand) instead of the 14-dim arms-only default; "
        "also streams hand commands via HAND_TRAJECTORY_COMMAND_TOPICS.",
    )
    parser.add_argument(
        "--steps-per-chunk",
        type=int,
        default=16,
        help="Actions executed from each predicted chunk before fetching the next "
        "observation (from episode row t + steps-per-chunk). Capped at the chunk's "
        "actual length (action_horizon) if smaller.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Must match the checkpoint's training data fps -- sets the per-step "
        "duration (1/fps) each predicted action is interpolated over.",
    )
    parser.add_argument("--command-hz", type=float, default=250.0, help="Real-mode publish rate")
    args = parser.parse_args()

    df = pd.read_parquet(args.episode_path)
    states = np.stack(df["observation.state"].to_numpy())
    expected_dim = 16 if args.with_hand else 14
    if states.shape[1] != expected_dim:
        raise ValueError(
            f"--with-hand={args.with_hand} expects {expected_dim}-dim observation.state, "
            f"got {states.shape[1]} from {args.episode_path}. Wrong dataset, or missing/extra --with-hand?"
        )
    state_keys = ["left_arm", "right_arm"] + (["left_hand", "right_hand"] if args.with_hand else [])

    episode_root, video_paths, tasks = load_episode_meta(args.episode_path)
    task_indices = (
        df["task_index"].to_numpy() if "task_index" in df.columns else np.zeros(len(df), dtype=int)
    )
    print(f"Loaded {len(df)} frames from {args.episode_path}")
    print(
        f"task_index changes at frames: "
        f"{np.where(np.diff(task_indices) != 0)[0].tolist() or 'none (single task)'}"
    )

    video_caps = {cam: cv2.VideoCapture(str(p)) for cam, p in video_paths.items()}
    for cam, p in video_paths.items():
        print(f"Video '{cam}': {p}")

    print(f"Connecting to policy server at {args.host}:{args.port}...")
    policy_client = PolicyClient(host=args.host, port=args.port)
    if not policy_client.ping():
        raise RuntimeError(f"Server at {args.host}:{args.port} did not respond to ping()")

    modality_config = policy_client.get_modality_config()
    action_keys = modality_config["action"].modality_keys
    action_chunk_size = len(modality_config["action"].delta_indices)
    lang_key = modality_config["language"].modality_keys[0]
    steps_per_chunk = min(args.steps_per_chunk, action_chunk_size)
    print(f"steps_per_chunk={steps_per_chunk} (action_horizon={action_chunk_size})")

    rclpy.init()
    node = Node("openarm_chunked_inference_replay")
    command_topics = dict(TRAJECTORY_COMMAND_TOPICS)
    if args.with_hand:
        command_topics.update(HAND_TRAJECTORY_COMMAND_TOPICS)
    pubs = {
        key: node.create_publisher(JointTrajectory, topic, 10)
        for key, topic in command_topics.items()
    }
    for key, topic in command_topics.items():
        print(f"Publishing '{key}' commands to: {topic}")
    time.sleep(0.5)  # let publishers connect before the first command

    n_frames = len(states)
    last_action = split_state_row(states[0], state_keys)
    segment_duration = 1.0 / args.fps

    try:
        t = 0
        while t < n_frames:
            state_dict = split_state_row(states[t], state_keys)
            images = {cam: read_frame(cap, t, cam) for cam, cap in video_caps.items()}
            obs = {"images": images, "state": state_dict}

            # Re-read task_index fresh at every chunk start -- some episodes (see
            # module docstring) concatenate multiple task segments back-to-back, so
            # a fixed task read once from frame 0 would send the wrong instruction
            # once t crosses into a later segment.
            task = tasks[int(task_indices[min(t, n_frames - 1)])]
            print(f"[t={t}/{n_frames}] task={task!r} -- requesting action chunk...")
            t0 = time.perf_counter()
            action_chunk = request_action_chunk(policy_client, obs, lang_key, task, action_keys)
            print(
                f"  got chunk in {time.perf_counter() - t0:.2f}s -- executing {steps_per_chunk} steps"
            )

            for step in range(steps_per_chunk):
                action = {key: action_chunk[key][step] for key in action_keys}
                stream_real_segment(
                    pubs, node, last_action, action, segment_duration, args.command_hz
                )
                last_action = action

            t += steps_per_chunk
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        for cap in video_caps.values():
            cap.release()
        node.destroy_node()
        rclpy.shutdown()
        policy_client.close()


if __name__ == "__main__":
    main()
