# DeepSpeed Pretrain Adaptation Plan

## Goal

This plan only covers `pretrain`.

Target constraints:

- Training framework: `DeepSpeed`
- Hardware target: `2 x 48GB` GPUs
- Preference: do not fully saturate VRAM
- Keep the current repo usable in single-GPU and non-DeepSpeed mode
- Preserve the current downstream weight format as much as practical

Recommended baseline:

- First target `ZeRO-2 + bf16 + no offload`
- Keep `ZeRO-3` as a second profile when model size, sequence length, or activation memory pushes VRAM too high
- Do not enable CPU/NVMe offload in the first pass because it increases complexity and usually hurts throughput

Reasoning:

- For a roughly `2B` dense model on `2 x 48GB`, `ZeRO-2` is often already enough if `micro_batch_per_gpu` and sequence length are controlled
- `ZeRO-3` gives more headroom, but it complicates initialization, save/load, and exported weight handling
- This repo is still in a learning-oriented, hand-written PyTorch style; the first DeepSpeed integration should minimize behavioral drift

## Current Code Observations

### 1. The training loop is tightly coupled to plain PyTorch optimizer/scaler calls

Current `pretrain` uses:

- manual `autocast`
- manual `GradScaler`
- manual `optimizer.step()`
- manual gradient clipping

Relevant code:

- [trainer/train_pretrain.py](/home/ziyao/minimind/trainer/train_pretrain.py#L32)
- [trainer/train_pretrain.py](/home/ziyao/minimind/trainer/train_pretrain.py#L37)
- [trainer/train_pretrain.py](/home/ziyao/minimind/trainer/train_pretrain.py#L39)
- [trainer/train_pretrain.py](/home/ziyao/minimind/trainer/train_pretrain.py#L43)

This must be abstracted for DeepSpeed because DeepSpeed should own:

- backward
- step
- gradient accumulation boundary handling
- mixed precision policy
- gradient clipping

### 2. Distributed init is DDP-specific

Current distributed bootstrap is:

- `dist.init_process_group(...)`
- `torch.cuda.set_device(local_rank)`
- wrapping model with `DistributedDataParallel`

Relevant code:

- [trainer/trainer_utils.py](/home/ziyao/minimind/trainer/trainer_utils.py#L44)
- [trainer/train_pretrain.py](/home/ziyao/minimind/trainer/train_pretrain.py#L142)

This is fine for DDP but not enough for a proper DeepSpeed path.

### 3. Model initialization always moves the full model onto one device

Current model init ends with `return model.to(device), tokenizer`.

Relevant code:

- [trainer/trainer_utils.py](/home/ziyao/minimind/trainer/trainer_utils.py#L119)

This is a critical issue for `ZeRO-3`. If the full model is moved to a CUDA device before DeepSpeed initializes partitioning, the memory-saving path is partially defeated.

### 4. Checkpoint and resume are built around full `state_dict`

Current checkpoint logic:

- saves a full `.pth` model file into `out/`
- saves a resume file containing model state, optimizer state, scaler state, epoch, and step

Relevant code:

- [trainer/train_pretrain.py](/home/ziyao/minimind/trainer/train_pretrain.py#L58)
- [trainer/trainer_utils.py](/home/ziyao/minimind/trainer/trainer_utils.py#L63)

This will not map cleanly onto DeepSpeed `ZeRO-2/3` resume behavior. DeepSpeed wants its own checkpoint directory structure.

### 5. Dependency and config plumbing for DeepSpeed is absent

There is no `deepspeed` dependency in the current `requirements.txt`.

Relevant code:

- [requirements.txt](/home/ziyao/minimind/requirements.txt#L1)

There are also no DeepSpeed JSON configs in the repository.

## Recommended Strategy

### Primary integration target

Implement a dual-path `pretrain` script:

- default path: current PyTorch/DDP behavior remains available
- new path: `--use_deepspeed 1` switches training to DeepSpeed

This is better than hard-replacing the current implementation because:

- it reduces migration risk
- it preserves existing learning value
- it allows a quick rollback when a DeepSpeed regression appears

### Memory strategy for `2 x 48GB`

Use two DeepSpeed profiles:

1. `ZeRO-2` baseline profile
2. `ZeRO-3` fallback profile

Recommended initial profile:

- precision: `bf16`
- ZeRO stage: `2`
- offload: disabled
- gradient clipping: handled by DeepSpeed config
- `train_micro_batch_size_per_gpu`: start from `1` or `2`
- `gradient_accumulation_steps`: keep explicit and conservative

Recommended fallback profile:

- precision: `bf16`
- ZeRO stage: `3`
- parameter offload: disabled
- optimizer offload: disabled
- `stage3_gather_16bit_weights_on_model_save`: enabled only when exporting a full model

Practical recommendation:

- Start implementation and validation on `ZeRO-2`
- Keep `ZeRO-3` in the same patch set if the abstraction is clean
- If time must be cut, land `ZeRO-2` first and add `ZeRO-3` next

## Planned File Changes

### 1. Add DeepSpeed config files

New files:

- `configs/deepspeed/pretrain_zero2_2x48.json`
- `configs/deepspeed/pretrain_zero3_2x48.json`

These files should define:

- `train_micro_batch_size_per_gpu`
- `gradient_accumulation_steps`
- `bf16.enabled`
- `zero_optimization`
- `gradient_clipping`
- logging / wall clock options only if useful

Notes:

- Do not put CPU offload in the first profile
- Do not overfit the config to one exact model size; keep it valid for several hidden sizes

### 2. Refactor trainer utilities to support both DDP and DeepSpeed

Target file:

- [trainer/trainer_utils.py](/home/ziyao/minimind/trainer/trainer_utils.py)

Required changes:

- split distributed bootstrap into generic helpers instead of a DDP-only helper
- add a helper to detect whether DeepSpeed is active
- refactor `init_model(...)` so it can optionally return a CPU model without `.to(device)`
- add a DeepSpeed-compatible checkpoint helper
- preserve the existing logging helpers

Important design choice:

- `init_model(...)` should support something like `move_to_device=True/False`
- For `ZeRO-3`, model construction should stay on CPU until `deepspeed.initialize(...)`

### 3. Rework `train_pretrain.py` into two execution paths

Target file:

- [trainer/train_pretrain.py](/home/ziyao/minimind/trainer/train_pretrain.py)

New CLI arguments:

- `--use_deepspeed`
- `--ds_config`
- `--ds_ckpt_dir`
- `--save_full_weights`

Optional CLI arguments:

- `--local_rank` for launcher compatibility
- `--zero_stage` only if you want CLI override; otherwise keep stage in JSON

Training loop changes for the DeepSpeed path:

- remove manual `autocast` control from the DeepSpeed branch
- remove `GradScaler` from the DeepSpeed branch
- replace `scaler.scale(loss).backward()` with `model_engine.backward(loss)`
- replace `optimizer/scaler.step()` with `model_engine.step()`
- use `model_engine.is_gradient_accumulation_boundary()` to decide logging and step-sensitive work when needed
- use `model_engine.optimizer.param_groups` if the current manual LR schedule is preserved

Recommended compatibility rule:

- keep the current cosine LR logic in phase 1
- do not add DeepSpeed scheduler config in the first patch unless needed

### 4. Replace resume logic with DeepSpeed-native checkpoints in the DeepSpeed branch

Current `lm_checkpoint(...)` behavior is not the right primitive for DeepSpeed resume.

DeepSpeed branch should:

- save DeepSpeed engine checkpoints into a directory, for example `checkpoints_ds/pretrain_<hidden_size>/global_step...`
- store client state with `epoch`, `step`, and optional `wandb_id`
- load resume state from `engine.load_checkpoint(...)`

Compatibility requirement:

- still export a consolidated weight file into `out/pretrain_*.pth`
- keep this export separate from resume checkpoints

Why this matters:

- evaluation and downstream scripts currently expect a single `.pth`
- DeepSpeed resume checkpoints are not a drop-in replacement for that format

### 5. Preserve downstream `.pth` export

The repo currently assumes a full model file exists under `out/`.

Relevant code:

- [trainer/train_pretrain.py](/home/ziyao/minimind/trainer/train_pretrain.py#L61)
- [trainer/trainer_utils.py](/home/ziyao/minimind/trainer/trainer_utils.py#L123)

Plan:

- keep exported full weights for compatibility
- use DeepSpeed-native checkpoint dirs for resume

Implementation note:

- `ZeRO-2`: exporting full 16-bit weights is straightforward
- `ZeRO-3`: exported model gathering must be explicit and should happen only on save intervals you really need

This is the main place where memory spikes must be watched.

### 6. Keep the non-DeepSpeed path intact

Do not delete:

- current DDP path
- current plain single-GPU path

This gives three useful execution modes:

- single GPU PyTorch
- multi-GPU DDP
- multi-GPU DeepSpeed

That makes debugging much easier.

## Proposed Implementation Phases

### Phase 0: non-code prep

- add a small environment note for DeepSpeed installation
- prefer a separate setup note instead of immediately hard-pinning `deepspeed` in `requirements.txt`

Reason:

- DeepSpeed installation is sensitive to CUDA, compiler, and Torch build combinations
- keeping it out of the main hard-pinned file avoids breaking users who only want the original repo behavior

### Phase 1: minimal DeepSpeed integration for `ZeRO-2`

Deliverables:

- one DeepSpeed config JSON
- `--use_deepspeed` branch in `train_pretrain.py`
- DeepSpeed engine save/load support
- preserved `out/pretrain_*.pth` export

Validation target:

- run on `2 GPUs`
- confirm decreasing loss
- confirm resume works
- confirm exported `.pth` can be loaded by the existing model loader

This is the minimum viable patch.

### Phase 2: add `ZeRO-3` profile

Deliverables:

- second JSON config
- safe initialization path that does not pre-place the full model on one GPU
- tested full-weight export under `ZeRO-3`

Validation target:

- confirm lower per-GPU VRAM than `ZeRO-2`
- confirm no correctness regression in exported weight loading

### Phase 3: optional memory headroom improvements

Only do this if `2 x 48GB` is still too tight.

Options:

- activation checkpointing over transformer blocks
- lower micro-batch per GPU
- slightly reduce sequence length for the first successful runs

Do not start with:

- CPU offload
- NVMe offload

Those are fallback tools, not the first plan.

## Validation Plan

### Smoke test

- launch `pretrain` on `2 GPUs`
- use a tiny dataset subset
- run a few dozen optimizer steps
- confirm loss logging works
- confirm checkpoints are written

### Resume test

- stop after one saved checkpoint
- resume from DeepSpeed checkpoint
- verify step and epoch continue correctly

### Export compatibility test

- export `out/pretrain_*.pth`
- load it through the existing `init_model(...)` path
- run a short inference or forward pass

### Memory observation

Track on both GPUs:

- peak VRAM
- stability over repeated save intervals
- any save-time memory spike during full-weight export

Success target for the baseline profile:

- training is stable on `2 x 48GB`
- there is meaningful memory headroom left instead of running at the limit

## Main Risks

### Risk 1: `init_model(...).to(device)` breaks `ZeRO-3` headroom

Mitigation:

- refactor `init_model(...)` before adding `ZeRO-3`

### Risk 2: full-weight export spikes memory

Mitigation:

- keep save frequency modest
- export only on rank 0
- validate on `ZeRO-2` first

### Risk 3: resume semantics differ from the current `step` skipping logic

Mitigation:

- treat DeepSpeed resume as authoritative
- keep `epoch` and `step` in DeepSpeed client state
- simplify skip logic if DeepSpeed checkpoint state already captures progress accurately

### Risk 4: `torch.compile` adds noise

Mitigation:

- disable `--use_compile` in the first DeepSpeed version
- only revisit it after the baseline path is stable

## Acceptance Criteria

The DeepSpeed pretrain adaptation is considered complete when:

- `train_pretrain.py` can run with `--use_deepspeed 1`
- `2 x 48GB` training is stable with a conservative memory profile
- resume works from DeepSpeed checkpoints
- an exported `out/pretrain_*.pth` remains loadable by the existing repo code
- the original non-DeepSpeed training path still works

## Recommended Next Execution Order

1. Implement Phase 1 only
2. Run a 2-GPU smoke test
3. Measure VRAM and save-time spikes
4. Add Phase 2 only if the baseline memory profile is still too tight

If implementation time must be minimized, do not start with `ZeRO-3`. Start with a clean `ZeRO-2` path that preserves the repo's current behavior model.
