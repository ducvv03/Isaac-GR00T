# Finetuning OpenArm (Bimanual, 7 Joints x 2 Arms)

This guide shows how to finetune GR00T N1.7 on the OpenArm bimanual manipulator using the
`clear_the_table_sim` dataset, and evaluate the result with open-loop evaluation.

## Robot Layout

16-dim state/action, split by [`openarm_config.py`](openarm_config.py) into:

| Modality key | Dims | Source joints |
|---|---|---|
| `left_arm` | 7 | `openarm_left_joint1..7` |
| `right_arm` | 7 | `openarm_right_joint1..7` |
| `left_gripper` | 1 | `openarm_left_finger_joint1` |
| `right_gripper` | 1 | `openarm_right_finger_joint1` |

Arms use `RELATIVE` actions (deltas from current joint state); grippers use `ABSOLUTE`
(target open/close position).

Cameras: `head` and `right` are used for training. `left` (`observation.images.cam_left`) is
**excluded** — it is a solid black feed in this recording (verified across multiple episodes
and timestamps). Check that camera before recording again, and re-add it to the `"video"`
block in `openarm_config.py` once it's fixed.

## Dataset

Recorded in LeRobot v3.0 format and converted in place to GR00T's v2.1 flavor at:
```
/home/ws/.cache/huggingface/lerobot/local/clear_the_table_sim_20260706_100401
```
The original v3.0 data is preserved alongside it at the `..._v3.0` suffix. `meta/modality.json`
in that dataset directory maps the state/action arrays and video keys as described above.

If you record a new v3.0 dataset and need to convert it the same way:
```bash
cd scripts/lerobot_conversion
uv venv && source .venv/bin/activate && uv pip install -e . --verbose
python convert_v3_to_v2.py --repo-id local/<your_dataset_name>
```
Videos must be H.264, not AV1 (this repo's video backend, `torchcodec`, does not reliably
decode AV1) — re-encode with `examples/SimplerEnv/convert_av1_to_h264.py <dataset_root>` if
needed, then copy `openarm_config.py`'s modality.json layout (or adjust it) into the new
dataset's `meta/modality.json`.

## Environment Setup

From the repository root:
```bash
cd /home/ws/pnk/vla/Isaac-GR00T
source .venv/bin/activate
```
This also exports two env vars needed only because this machine has no root access (no
`apt`/CUDA-toolkit install) — see `.venv/bin/activate` if you need to recreate the venv from
scratch:
- `LD_LIBRARY_PATH`, so `torchcodec` finds FFmpeg shared libraries, repurposed from the `av`
  package's bundled ones.
- `CUDA_HOME`, pointing at a stub `nvcc` (`.venv/fake_cuda/bin/nvcc`) that only prints a
  version string. `accelerate` unconditionally imports `deepspeed` to check
  `isinstance(model, DeepSpeedEngine)`, and `deepspeed`'s own `__init__` probes
  `$CUDA_HOME/bin/nvcc -V` at import time even though this repo's single-GPU finetune path
  (`adamw_torch`, no `--deepspeed` config) never actually invokes it. Without this, finetuning
  fails immediately with `MissingCUDAException: CUDA_HOME does not exist`. If you later want
  real multi-GPU DeepSpeed ZeRO training, you'll need an actual CUDA toolkit here instead.

Both `uv run python ...` and plain `python ...` work correctly once `activate` has been
sourced in the current shell.

## Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /home/ws/.cache/huggingface/lerobot/local/clear_the_table_sim_20260706_100401 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/OpenArm/openarm_config.py \
    --num-gpus 1 \
    --output-dir /tmp/openarm_finetune \
    --global-batch-size 8 \
    --gradient-accumulation-steps 4 \
    --save-steps 500 \
    --save-total-limit 5 \
    --max-steps 5000 \
    --dataloader-num-workers 4
```

`--global-batch-size 8` / `--gradient-accumulation-steps 4` (effective batch 32) is a starting
point — adjust to your GPU. On a 16GB card (RTX 5060 Ti), a smoke test here showed even
`--global-batch-size 1` hitting `CUDA out of memory` with the default `--tune-projector
True --tune-diffusion-model True` (the ~1.6B trainable params' AdamW optimizer state alone
uses ~13.7GB, leaving almost nothing for activations). If you hit this, options include
running on a larger GPU, or freezing more of the model
(`--tune-diffusion-model False` keeps only the new embodiment's small state/action
encoder-decoder trainable, freezing the shared ~1.3B DiT) — not yet tried against this
dataset. The first run downloads the ~6GB base checkpoint from HuggingFace.

Add `--use-wandb` to log to Weights & Biases.

## Open-Loop Evaluation

```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path /home/ws/.cache/huggingface/lerobot/local/clear_the_table_sim_20260706_100401 \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path /tmp/openarm_finetune/checkpoint-500 \
    --traj-ids 0 \
    --action-horizon 16 \
    --steps 400 \
    --modality-keys left_arm right_arm left_gripper right_gripper
```

Run this against each saved checkpoint and confirm `Average MSE`/`MAE` trends down as training
progresses. See [Interpreting the Result: Is My Fine-tune Working?](../../getting_started/finetune_new_embodiment.md#interpreting-the-result-is-my-fine-tune-working)
for how to read these numbers.

## Sim Testing (ROS2 / Isaac Sim)

[`ros2_gr00t_client.py`](ros2_gr00t_client.py) bridges a running sim (e.g. Isaac Sim) to the
GR00T policy server over ROS2. It subscribes to `/joint_states` and 3 camera topics, and
publishes commanded joint positions to `/joint_command`.

This runs under **ROS2's system Python (3.12, rclpy)**, not this repo's `.venv` (Python 3.10,
torch/transformers). The two only talk over ZMQ — [`server_client.py`](server_client.py) is a
standalone client (numpy/zmq/msgpack only, no `gr00t` import) so the ROS2 side never needs the
heavy ML deps.

**One-time setup** — a dedicated venv that inherits ROS2 (`rclpy`, `sensor_msgs`) via
`--system-site-packages` plus the two extra deps:
```bash
cd examples/OpenArm
python3.12 -m venv --system-site-packages .venv-ros
source .venv-ros/bin/activate
pip install pyzmq msgpack
```

**Terminal 1 — policy server** (this repo's `.venv`, GPU):
```bash
cd /home/ws/pnk/vla/Isaac-GR00T && source .venv/bin/activate
python gr00t/eval/run_gr00t_server.py \
    --model-path <finetuned-checkpoint-dir> \
    --embodiment-tag NEW_EMBODIMENT
```

**Terminal 2 — Isaac Sim** (or whatever publishes `/joint_states` + the camera topics and
subscribes to `/joint_command`) — not part of this repo.

**Terminal 3 — this client** (`.venv-ros` from setup above):
```bash
cd examples/OpenArm && source .venv-ros/bin/activate
python ros2_gr00t_client.py --task "Pick up all the items on the table and put them into the bin on the right."
```

### Joint mapping

Of the joints on `/joint_states`, only the 16 that carry training signal are used (mimic/passive
joints — the second finger of each gripper, `*_hand`, `*_ee_tcp_joint` — are read if present but
ignored), matching `openarm_config.py`'s state/action layout:

| Modality key | Dims | Source joints (read from `/joint_states`, by name) |
|---|---|---|
| `left_arm` | 7 | `openarm_left_joint1..7` |
| `right_arm` | 7 | `openarm_right_joint1..7` |
| `left_gripper` | 1 | `openarm_left_finger_joint1` |
| `right_gripper` | 1 | `openarm_right_finger_joint1` |

`/joint_command` is published with exactly these same 16 joint names/values — the model's raw
output, no re-expansion to the second finger joint. Camera topics map to video modality keys as
`head` -> `/camera/head/image_raw`, `left` -> `/camera/wrist_left/image_raw`,
`right` -> `/camera/wrist_right/image_raw`.

Arms are `RELATIVE` actions and grippers `ABSOLUTE` (see Robot Layout above), but the client
doesn't need to handle that conversion itself — the policy server's `unapply_action` already
converts relative deltas back to absolute target positions server-side (using the state sent in
the same request), so `/joint_command` values are ready to command directly.
