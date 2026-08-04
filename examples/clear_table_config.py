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

# Same OpenArm modality layout as examples/OpenArm/openarm_config.py, but with the
# action horizon raised from 16 to 40 -- the architectural max for this base model
# (Gr00tN1d7Config.action_horizon = 40; see config.json on the base/finetuned
# checkpoints). Going past 40 is not possible without retraining a different base
# architecture. Everything else (video/state history, action reps, LoRA patch,
# dataset) is unchanged from the original clear_the_table_v2 training recipe.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


clear_table_config = {
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
            "left_gripper",  # openarm_left_finger_joint1
            "right_gripper",  # openarm_right_finger_joint1
        ],
    ),
    # Action: 40-step prediction horizon (architectural max); one ActionConfig per
    # modality key, in the same order as modality_keys.
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_gripper",
            "right_gripper",
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
            # Grippers: ABSOLUTE = target open/close position
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

register_modality_config(clear_table_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
