# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills.
The repo contains the model, training pipeline, evaluation harness, and deployment tooling.

- **Language:** Python 3.10 (dGPU, Orin); Python 3.12 (Thor, DGX Spark — see deployment dir)
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build system:** setuptools (see `pyproject.toml`)
- **CI:** internal GitLab CI (`.gitlab-ci.yml` + includes under `ci/`, not shipped to the public GitHub EA repo); public GitHub Actions (`.github/workflows/main.yml`) runs `ruff check`/`ruff format --check` and a build on Python 3.10 for every PR into `main`

## Setup

- The repo uses git submodules for benchmark deps (`external_dependencies/{LIBERO,SimplerEnv,robocasa,robocasa-gr1-tabletop-tasks}`); clone with `--recurse-submodules` or run `git submodule update --init --recursive`.
- `git-lfs` is required for the parquet files under `demo_data/`.
- Video decoding uses [`torchcodec`](https://github.com/pytorch/torchcodec) exclusively (requires FFmpeg); `decord`/`pyav` are not supported. On aarch64 (Thor, Orin), `torchcodec` is built from source during `install_deps.sh`.
- If fine-tuning fails with `CUDA_HOME is unset`: run `scripts/deployment/dgpu/install_deps.sh` once, or `export CUDA_HOME=/usr/local/cuda`.
- On CUDA 13.x (Thor, Spark, GB300): PyTorch 2.7 pins Triton 3.3.1, which doesn't recognize CUDA 13 — run `bash scripts/patch_triton_cuda13.sh` after install.

## Quick-start commands

```bash
# Install (dev mode with all extras)
uv sync --all-extras

# Lint and format (uses ruff via pre-commit)
pre-commit run --all-files

# Run CPU tests
python -m pytest tests/ -m "not gpu" -v --timeout=300

# Run GPU tests
python -m pytest tests/ -m gpu -v --timeout=300

# Run a single test file / single test
python -m pytest tests/gr00t/data/test_embodiment_tags.py -v
python -m pytest tests/gr00t/model/test_model_forward.py -k some_case -v

# Build package
uv build

# Validate lockfile
uv lock --locked
```

## Code style

- Formatter: `ruff format` (double quotes, spaces, line-length 100)
- Linter: `ruff check` with rules E, F, I (ignores E501)
- Config lives in `pyproject.toml` under `[tool.ruff]`
- Run `pre-commit run --all-files` before committing

## Directory layout

```
gr00t/              # Main package
  configs/          #   Training, data, and model configs (dataclasses, tyro CLI, YAML)
  data/             #   Data loading, embodiment tags, dataset processing
  deployment/       #   Shared mode enums used by scripts/deployment CLIs
  eval/             #   Evaluation (run_gr00t_server.py, open_loop_eval.py, rollout_policy.py)
  experiment/       #   Training pipeline (launch_finetune.py, launch_train.py, trainer.py)
  model/            #   Model architecture (N1.7, base, modules) + MODEL_REGISTRY
  policy/           #   Policy inference (Gr00tPolicy, server/client)
  utils/            #   Determinism, video, and initial-actions helpers
examples/           # Per-embodiment example configs and READMEs (DROID, LIBERO, SimplerEnv, SO100, ...)
scripts/            # Deployment, conversion, and utility scripts
  deployment/       #   Platform install scripts (dgpu, orin, thor, spark) + ONNX/TRT export/build/benchmark CLIs
  lerobot_conversion/ #   Standalone venv for converting LeRobot v3 -> GR00T v2 dataset format
tests/              # pytest suite, mirrors gr00t/ and scripts/ layout (markers: gpu, edge_device, multigpu)
getting_started/    # User-facing guides and notebooks
demo_data/          # Small LeRobot-format datasets for smoke-testing each embodiment (git-lfs)
```

## Key entry points

- **Fine-tune:** `bash examples/finetune.sh --base-model-path <path> --dataset-path <path> --embodiment-tag <tag> --output-dir <dir>`
- **Inference server:** `python gr00t/eval/run_gr00t_server.py --model-path <path> --embodiment-tag <tag>`
- **Open-loop eval client:** `python gr00t/eval/open_loop_eval.py --dataset-path <path> --embodiment-tag <tag> --host <host> --port <port>`
- **Standalone inference (no server):** `python scripts/deployment/standalone_inference_script.py --model-path <path> --dataset-path <path> --embodiment-tag <tag> --inference-mode {pytorch,trt_full_pipeline}`
- **ONNX export:** `python scripts/deployment/export_onnx_n1d7.py`
- **TensorRT build:** `python scripts/deployment/build_trt_pipeline.py`
- **TensorRT verify:** `python scripts/deployment/verify_n1d7_trt.py`
- **Benchmark:** `python scripts/deployment/benchmark_inference.py`

## Architecture

### Data flow: dataset → embodiment tag → model

Datasets use a GR00T-flavored LeRobot v2 format (`meta/info.json`, `meta/modality.json`,
`data/chunk-*/*.parquet`, `videos/chunk-*/*.mp4`). Every training/inference entry point takes a
**`--embodiment-tag`** (`gr00t/data/embodiment_tags.py::EmbodimentTag`, case-insensitive) that selects a
`ModalityConfig` describing how the flat state/action arrays split into named fields and which video keys exist.
Tags fall into three groups: pretrain tags (baked into the base checkpoint, inference-ready zero-shot),
pre-registered posttrain tags (require a finetuned checkpoint), and finetune-only tags (`NEW_EMBODIMENT`, for
onboarding a custom robot). A `BaseProcessor` (`gr00t/data/interfaces.py`) converts raw dataset rows into
model-ready tensors and decodes model outputs back into actions; `ShardedDataset` is the abstract base for the
sharded dataset loaders.

### Model: VLM backbone + flow-matching action head

`Gr00tN1d7` (`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`) combines a Qwen3-VL vision-language backbone
(`gr00t/model/modules/qwen3_backbone.py`) with a diffusion transformer (DiT) action head that denoises continuous
actions via flow matching (`gr00t/model/modules/dit.py`: `DiT`, `AlternateVLDiT`, `SelfAttentionTransformer`).
Per-embodiment state/action encoding goes through `CategorySpecificMLP` / `MultiEmbodimentActionEncoder`
(`gr00t/model/modules/embodiment_conditioned_mlp.py`). Model configs register into a global `MODEL_REGISTRY`
(`gr00t/model/registry.py`) as a `config_cls -> pipeline_cls` mapping via `register_model()`, which is how a
config resolves to a concrete model implementation. `processing_gr00t_n1d7.py` holds the N1.7-specific
`BaseProcessor` (image transform/letterboxing, tokenization, action normalization).

### Config composition

Configs are plain dataclasses parsed via `tyro` (`gr00t/configs/base_config.py`,
`gr00t/configs/finetune_config.py`): a top-level config composes `DataConfig`/`SingleDatasetConfig`
(`gr00t/configs/data/`), a model config (`gr00t/configs/model/gr00t_n1d7.py`), and `TrainingConfig`
(`gr00t/configs/training/`). Configs serialize to/from YAML with `yaml.safe_load`/`safe_dump` only; a legacy
config using `!!python/object` tags raises an explicit migration error rather than being executed.

### Policy and deployment layering

`BasePolicy` (`gr00t/policy/policy.py`) is the abstract interface (`check_observation`/`check_action`/
`_get_action`/`reset`) implemented by `Gr00tPolicy` (`gr00t/policy/gr00t_policy.py`) for local PyTorch inference,
and mirrored by a client/server pair (`gr00t/policy/server_client.py`, driven by `gr00t/eval/run_gr00t_server.py`)
that runs the policy on a GPU server and talks to lightweight robot-side clients over ZMQ — the path used for real
hardware and sim benchmarks where the control loop can't colocate with the GPU.

Deployment CLIs under `scripts/deployment/` (`export_onnx_n1d7.py`, `build_trt_pipeline.py`,
`verify_n1d7_trt.py`, `benchmark_inference.py`, `build_tensorrt_engine.py`) share their mode-flag value sets from
one place, `gr00t/deployment/modes.py` (`ExportMode`, `VerifyMode`, `BenchmarkMode`, `BuildEngineMode`), so the
CLIs can't drift on accepted values. TRT export has three modes with different component coverage —
`dit_only` (legacy, action-head DiT only), `action_head` (state/action encoders + DiT + decoder), and
`full_pipeline` (backbone + action head, recommended on dGPU/Thor/Spark); see `scripts/deployment/README.md` for
the component/engine matrix.

## Testing

- Test markers (declared in `pyproject.toml`): `gpu` (requires a GPU), `edge_device` (Orin/Thor/Spark runners
  only), `multigpu` (uses all visible GPUs); default (no marker filter) is CPU-safe
- `pythonpath` includes both `.` and `tests`, and tests run with `--import-mode=importlib`
- Fixtures live in `tests/fixtures/` and `demo_data/`; the test tree mirrors `gr00t/` and `scripts/`
- CI runs CPU and GPU tests in separate jobs with a 300s timeout

## Deployment platforms

- **dGPU (H100, A100, RTX):** CUDA 12.8 — install via `scripts/deployment/dgpu/install_deps.sh`, container via top-level `docker/Dockerfile` (supports x86_64 and aarch64)
- **Jetson Orin:** CUDA 12.6 — install via `scripts/deployment/orin/install_deps.sh`, container via `scripts/deployment/orin/Dockerfile`
- **Jetson Thor:** CUDA 13.0 — install via `scripts/deployment/thor/install_deps.sh`, container via `scripts/deployment/thor/Dockerfile`
- **DGX Spark:** CUDA 13.0 — install via `scripts/deployment/spark/install_deps.sh`, container via `scripts/deployment/spark/Dockerfile`

Each Jetson/Spark platform ships an `activate_*.sh` helper (`scripts/activate_orin.sh`, `scripts/activate_spark.sh`, `scripts/activate_thor.sh`) that exports platform-specific library paths. For dGPU, the standard `source .venv/bin/activate` is sufficient.

On aarch64 platforms (Thor, Spark, Orin), after `install_deps.sh` always invoke Python with **plain `python`**,
not `uv run python` — `uv run` re-syncs against the root `pyproject.toml` (which targets x86_64 Python 3.10) and
will destroy the platform-specific venv.
