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

"""Merge a raw (unmerged) PEFT/LoRA-wrapped GR00T DiT checkpoint into a plain,
directly-loadable checkpoint.

Background: this repo's launch_finetune.py has no built-in LoRA support. If DiT LoRA
finetuning was done with external/custom training code that wraps model.action_head.model
with peft.get_peft_model(...) and then saves via a plain trainer.save_model() (no
merge_and_unload(), no proper PEFT adapter export), every wrapped key ends up with a
"base_model.model." prefix (PEFT's wrapping convention) that AutoModel.from_pretrained
doesn't recognize -- so it silently reinitializes the entire DiT instead of erroring,
discarding the LoRA training.

This script:
1. Loads the raw state dict from the checkpoint's safetensors shards.
2. For each base_model.model.-prefixed key: strips the prefix, and for LoRA-adapted
   linears (base_layer + lora_A + lora_B), computes
   merged_weight = base_weight + (lora_B @ lora_A) * (lora_alpha / r).
3. Builds the real model skeleton via AutoModel.from_pretrained (accepting its
   missing-key warning -- we're about to overwrite those weights anyway) so key names
   come from the actual current architecture, not a guess.
4. Verifies the renamed/merged key set exactly matches the skeleton's own state_dict
   keys, modulo --expected-missing-keys (a hard assertion, not just a warning).
5. load_state_dict(strict=False, then asserts the only gaps are --expected-missing-keys),
   then save_pretrained() to a clean output dir.

Usage:
    python scripts/merge_lora_dit_checkpoint.py --ckpt-dir checkpoints/9500 --out-dir checkpoints/9500_merged
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

from safetensors.torch import load_file
import tyro


@dataclass
class MergeLoraConfig:
    ckpt_dir: str
    """Path to the raw (unmerged) LoRA checkpoint directory."""

    out_dir: str
    """Path to write the merged, directly-loadable checkpoint to."""

    lora_r: int = 4
    """LoRA rank used during training."""

    lora_alpha: int = 8
    """LoRA alpha used during training. scaling = lora_alpha / lora_r."""

    peft_prefix: str = "base_model.model"
    """PEFT's module-wrapping prefix to strip (peft.get_peft_model's default)."""

    expected_missing_keys: tuple[str, ...] = field(
        default_factory=lambda: ("backbone.model.lm_head.weight",)
    )
    """Skeleton keys allowed to stay randomly-initialized (not an error). Default is
    lm_head, which this training pipeline never saves since GR00T's action-prediction
    path doesn't use it -- true for any checkpoint saved this way, LoRA or not."""


def merge(config: MergeLoraConfig) -> None:
    ckpt_dir = Path(config.ckpt_dir)
    out_dir = Path(config.out_dir)
    scaling = config.lora_alpha / config.lora_r
    prefix = config.peft_prefix

    print(f"Loading raw state dict from {ckpt_dir} ...")
    index = json.load(open(ckpt_dir / "model.safetensors.index.json"))["weight_map"]
    shard_files = sorted(set(index.values()))
    raw = {}
    for shard in shard_files:
        raw.update(load_file(ckpt_dir / shard))
    print(f"  {len(raw)} raw tensors across {len(shard_files)} shards")

    merged = {}
    skipped_lora_only = 0
    merged_lora_count = 0

    for key, tensor in raw.items():
        if prefix not in key:
            merged[key] = tensor
            continue
        if ".lora_A." in key or ".lora_B." in key:
            skipped_lora_only += 1
            continue  # consumed via the .base_layer. branch below
        if ".base_layer." in key:
            new_key = key.replace(f"{prefix}.", "").replace(".base_layer.", ".")
            if new_key.endswith(".weight"):
                lora_prefix = key.rsplit(".base_layer.weight", 1)[0]
                lora_a_key = f"{lora_prefix}.lora_A.default.weight"
                lora_b_key = f"{lora_prefix}.lora_B.default.weight"
                if lora_a_key in raw and lora_b_key in raw:
                    delta = (raw[lora_b_key].float() @ raw[lora_a_key].float()) * scaling
                    merged[new_key] = (tensor.float() + delta).to(tensor.dtype)
                    merged_lora_count += 1
                else:
                    merged[new_key] = tensor
            else:
                merged[new_key] = tensor  # bias: unaffected when LoRA bias="none"
            continue
        # prefixed key with no LoRA adapter (e.g. norm1, proj_out, timestep_encoder)
        merged[key.replace(f"{prefix}.", "")] = tensor

    print(f"  merged {merged_lora_count} LoRA-adapted linears (scaling={scaling})")
    print(f"  {len(merged)} merged tensors ({skipped_lora_only} raw lora_A/B tensors consumed)")

    print("Building model skeleton via AutoModel.from_pretrained (missing-key warning expected)...")
    import gr00t.model  # noqa: F401  registers Gr00tN1d7 with the transformers Auto* registry
    from transformers import AutoModel

    model = AutoModel.from_pretrained(ckpt_dir)
    skeleton_keys = set(model.state_dict().keys())
    merged_keys = set(merged.keys())
    expected_missing = set(config.expected_missing_keys)

    only_in_skeleton = skeleton_keys - merged_keys
    only_in_merged = merged_keys - skeleton_keys
    unexpected_missing = only_in_skeleton - expected_missing
    if unexpected_missing or only_in_merged:
        print(
            f"KEY MISMATCH: {len(unexpected_missing)} unexpected-missing, "
            f"{len(only_in_merged)} only in merged"
        )
        print("  sample unexpected_missing:", sorted(unexpected_missing)[:10])
        print("  sample only_in_merged:", sorted(only_in_merged)[:10])
        raise SystemExit(1)
    print(f"  key sets match (modulo expected-missing {expected_missing}).")

    missing, unexpected = model.load_state_dict(merged, strict=False)
    assert set(missing) == expected_missing, f"unexpected missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    print(f"load_state_dict OK (only expected-missing keys left random-init: {missing})")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {out_dir} ...")
    model.save_pretrained(out_dir, max_shard_size="5GB")

    for fname in ["processor_config.json", "statistics.json", "embodiment_id.json"]:
        src = ckpt_dir / fname
        if src.exists():
            (out_dir / fname).write_bytes(src.read_bytes())
            print(f"  copied {fname}")

    print("Done.")


if __name__ == "__main__":
    merge(tyro.cli(MergeLoraConfig))
