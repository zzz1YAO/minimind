# DeepSpeed Setup Note

This repo keeps DeepSpeed as an optional dependency for `trainer/train_pretrain.py --use_deepspeed 1`.

Recommended install flow:

```bash
pip install deepspeed
ds_report
```

Notes:

- Install PyTorch first, then install DeepSpeed in the same environment.
- `ds_report` helps confirm whether your CUDA environment and optional ops are usable.
- The provided pretrain configs target `bf16` on multi-GPU CUDA training and do not enable CPU/NVMe offload in the first pass.

Useful launch examples:

```bash
deepspeed trainer/train_pretrain.py --use_deepspeed 1 --ds_config configs/deepspeed/pretrain_zero2_2x48.json
```

```bash
deepspeed trainer/train_pretrain.py --use_deepspeed 1 --ds_config configs/deepspeed/pretrain_zero3_2x48.json
```
