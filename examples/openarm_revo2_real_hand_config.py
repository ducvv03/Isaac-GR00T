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

# Modality config for the openarm_revo2_follower_real dataset at
# /home/ws/data/real/lerobot_v2_data_filtered/{20260806,20260807,20260810}/ --
# REAL hardware recordings, growing (new dates keep appearing as more data is
# collected) -- 51 episode folders / 306 episodes / ~80.8k frames (~67 min @
# 20fps) as of 2026-08-10. Unlike the earlier real recordings at
# .../{20260731,20260804}/ (see openarm_revo2_arms_only_config.py, 14-dim, no
# hand at all -- those dates are no longer present under this path), these
# recordings DO include hand state/action: left_arm(7) + right_arm(7) +
# left_hand(1) + right_hand(1) = 16 dims total, identical meta/modality.json
# structure across all 51 folders, fps=20 uniform, only 2 distinct tasks
# (lift/place the box) -- all confirmed by a full per-episode consistency
# check (dims, NaN, frame counts vs metadata, video counts vs episode counts,
# 2026-08-10, zero issues found). Same schema as the sim dataset in
# openarm_revo2_hand_config.py (single index_proximal joint per hand).
#
# Real recorded left_hand/right_hand values cluster at ~0 (open) and
# ~0.30-0.40 (closed/holding) (checked across all 51 episodes) -- consistent
# with ros2_gr00t_client.py's HAND_CLOSE_THRESHOLD=0.2, so a checkpoint trained
# on this config can run through ros2_gr00t_client.py / state_machine.py /
# auto_state_machine.py WITHOUT --no-hand (unlike the arms-only real
# checkpoint, which requires it).

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


openarm_revo2_real_hand_config = {
    # Video: current frame only; keys must match "video" entries in meta/modality.json.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head", "left", "right"],
    ),
    # State: current proprioceptive reading; keys must match "state" entries in meta/modality.json
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",  # 7 joint positions, openarm_left_joint1..7
            "right_arm",  # 7 joint positions, openarm_right_joint1..7
            "left_hand",  # 1 joint position: left_index_proximal_joint
            "right_hand",  # 1 joint position: right_index_proximal_joint
        ],
    ),
    # Action: 40-step prediction horizon (architectural max, Gr00tN1d7Config.action_horizon
    # = 40; 2.0s of motion at this dataset's fps=20). One ActionConfig per modality key,
    # in the same order as modality_keys.
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
        action_configs=[
            # Arms: RELATIVE = delta from current joint state (better generalization)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Hands: ABSOLUTE = target joint position, same convention as
            # openarm_revo2_hand_config.py (single index_proximal joint per side).
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction from annotation field in the dataset
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    openarm_revo2_real_hand_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
