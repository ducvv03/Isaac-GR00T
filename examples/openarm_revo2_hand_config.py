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

# Modality config for the openarm_revo2_follower_sim dataset at
# /home/ws/data/sim/lerobot_v2_data_filtered/20260803/ -- differs from
# examples/OpenArm/openarm_config.py in that the "hand" key maps to a single
# index_proximal joint per side (not a 1-joint gripper, but structurally
# equivalent -- 1 scalar per hand), and the dataset was recorded at fps=20
# (not 30). state/action layout: left_arm(7) + right_arm(7) + left_hand(1) +
# right_hand(1) = 16 dims total, confirmed against meta/modality.json
# (identical across all 13 episode folders in that recording as of 2026-08-04).

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


openarm_revo2_hand_config = {
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
            # Hands: ABSOLUTE = target joint position, same convention as grippers
            # in openarm_config.py (single index_proximal joint per side here).
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

register_modality_config(openarm_revo2_hand_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
