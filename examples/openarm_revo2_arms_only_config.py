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
# /home/ws/data/real/lerobot_v2_data_filtered/{20260731,20260804}/ -- REAL
# hardware recordings, arms only (no hand/gripper state or action recorded at
# all -- meta/modality.json has no "hand" key, state/action are 14-dim:
# left_arm(7) + right_arm(7)). Identical meta/modality.json across all 25
# episode folders in both recording dates (confirmed 2026-08-05). fps=20,
# same video keys (head/left/right) and task set (lift/place the metal box)
# as the sim dataset in openarm_revo2_hand_config.py.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


openarm_revo2_arms_only_config = {
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
        ],
    ),
    # Action: 40-step prediction horizon (architectural max, matches
    # openarm_revo2_hand_config.py; 2.0s of motion at this dataset's fps=20).
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "left_arm",
            "right_arm",
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
        ],
    ),
    # Language: task instruction from annotation field in the dataset
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(openarm_revo2_arms_only_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
