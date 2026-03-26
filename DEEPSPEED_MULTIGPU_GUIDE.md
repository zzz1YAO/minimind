# DeepSpeed 多卡训练速查

这份文档面向 `MiniMind` 这类手写 PyTorch 训练仓库，目标是把 DeepSpeed 多卡训练的准备顺序、关键配置和调参逻辑讲清楚。

## 1. 先理解这个仓库里的训练模式

这个 repo 里有两条路：

- 普通 PyTorch / DDP
- DeepSpeed 分支，打开 `--use_deepspeed 1`

当前预训练脚本会读取：

- `trainer/train_pretrain.py`
- `configs/deepspeed/pretrain_zero2_2x48.json`
- `configs/deepspeed/pretrain_zero3_2x48.json`

DeepSpeed 这条路的关键点是：

- DeepSpeed 接管 `backward`
- DeepSpeed 接管 `step`
- DeepSpeed 接管梯度累积边界
- DeepSpeed 接管混合精度
- DeepSpeed 接管 checkpoint 保存与恢复

## 2. 环境准备顺序

先装 PyTorch，再装 DeepSpeed。

建议顺序：

1. 创建 Python 环境
2. 安装和你 CUDA 匹配的 PyTorch
3. 安装 `requirements.txt`
4. 单独安装 `deepspeed`
5. 用 `ds_report` 检查环境
6. 先跑单卡或小规模多卡，再上正式训练

建议先检查：

```bash
python -c "import torch; print(torch.cuda.is_available())"
ds_report
```

如果 `ds_report` 报 CUDA / 编译器 / op 不可用，先别训练，先把环境修掉。

## 3. 训练启动顺序

### 3.1 先确认数据

预训练数据需要是 jsonl，且每条至少长这样：

```json
{"text": "这里是一段训练文本"}
```

如果你有多个分片，先合并成一个文件最省事：

```bash
cat part*.jsonl > pretrain.jsonl
```
3785606条数据，skypile数据集

### 3.2 先准备模型结构配置和 tokenizer

这次改动后，`trainer/train_pretrain.py` 已经支持：

- `--model_config`：从一个 `config.json` 读取 Transformer 结构参数
- `--tokenizer_path`：指定 tokenizer 目录或 Hugging Face 模型名
- 命令行 override：如果你同时传了 `--hidden_size` 之类的参数，命令行优先级更高

建议做法是：

1. 先把模型结构放进一个 `config.json`
2. 再明确指定训练要用的 tokenizer
3. 让脚本在训练开始时自动把 `vocab_size`、`bos/eos/pad_token_id` 和 tokenizer 对齐

一个最小例子：

```json
{
  "hidden_size": 1024,
  "num_hidden_layers": 16,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "max_position_embeddings": 32768,
  "use_moe": false
}
```

如果你打算用 Qwen2 tokenizer，建议直接传：

```bash
--tokenizer_path /path/to/Qwen2-tokenizer
```

这时脚本会按 tokenizer 自动同步：

- `vocab_size`
- `bos_token_id`
- `eos_token_id`
- `pad_token_id`

训练开始后，脚本还会把**本次真正生效的模型配置**保存到输出目录，方便你后续转成 HF 格式。

### 3.3 先跑 ZeRO-2

这是推荐的第一步。

```bash
deepspeed trainer/train_pretrain.py \
  --use_deepspeed 1 \
  --ds_config configs/deepspeed/pretrain_zero2_2x48.json \
  --model_config configs/model/pretrain_1b.json \
  --tokenizer_path /path/to/tokenizer \
  --data_path ../dataset/pretrain.jsonl
```

先验证下面这些东西都正常：

- 能起多卡
- loss 正常下降
- checkpoint 能保存
- resume 能恢复

如果你只是想临时覆盖某个结构参数，也可以直接在命令里加，例如：

```bash
deepspeed trainer/train_pretrain.py \
  --use_deepspeed 1 \
  --ds_config configs/deepspeed/pretrain_zero2_2x48.json \
  --model_config configs/model/pretrain_1b.json \
  --tokenizer_path /path/to/tokenizer \
  --num_hidden_layers 20 \
  --data_path ../dataset/pretrain.jsonl
```

这里的效果是：

- 大部分结构参数从 `config.json` 读取
- `num_hidden_layers` 由命令行覆盖为 `20`

### 3.4 再考虑 ZeRO-3

如果显存还是不够，再切到：

```bash
deepspeed trainer/train_pretrain.py \
  --use_deepspeed 1 \
  --ds_config configs/deepspeed/pretrain_zero3_2x48.json \
  --model_config configs/model/pretrain_1b.json \
  --tokenizer_path /path/to/tokenizer \
  --data_path ../dataset/pretrain.jsonl
```

ZeRO-3 更省显存，但初始化、保存、恢复都更复杂，优先级低于 ZeRO-2。

## 4. 这几个 DeepSpeed 配置项你要重点学

### 4.1 `train_micro_batch_size_per_gpu`

含义：每张卡一次前向/反向吃多少样本。

它不是全局 batch，而是**单卡 micro batch**。

这个值越大：

- 显存占用越高
- 吞吐可能更好

这个值越小：

- 更省显存
- 但可能需要更大的 `gradient_accumulation_steps` 才能维持全局 batch

### 4.2 `gradient_accumulation_steps`

含义：梯度累积多少步后再真正更新一次参数。

它的作用是把显存受限的小 micro batch，拼成更大的有效 batch。

大致关系：

```text
global_batch = micro_batch_per_gpu × accumulation_steps × GPU_num
```

例如：

- `micro_batch_per_gpu=2`
- `accumulation_steps=8`
- `GPU_num=4`

那么全局 batch 约等于 `64`

### 4.3 `bf16.enabled`

这个 repo 的 DeepSpeed 配置默认开 `bf16`。

含义：

- 训练用 bfloat16
- 通常比 fp16 稳
- 适合较新的 NVIDIA GPU

如果你的卡不支持 bf16，再考虑改成别的精度，但这不属于这套默认方案。

### 4.4 `zero_optimization.stage`

这是最核心的 DeepSpeed 开关。

- `stage = 2`：先推荐
- `stage = 3`：更省显存，但更复杂

理解方式：

- ZeRO-2 主要分片优化器状态和梯度
- ZeRO-3 连模型参数也分片

训练初期优先选 `ZeRO-2`，因为它更稳，更容易排错。

### 4.5 `stage3_gather_16bit_weights_on_model_save`

这个只对 ZeRO-3 特别重要。

含义：

- 保存时把分片的 16bit 权重聚合出来
- 方便导出完整模型文件

如果你要训练完导出一个可单独使用的模型，这个开关很关键。

### 4.6 `gradient_clipping`

梯度裁剪，防止梯度爆掉。

对于大模型预训练，通常保留。

### 4.7 `overlap_comm`

让通信和计算尽量重叠。

一般是提吞吐的选项，通常保留开启。

### 4.8 `contiguous_gradients`

让梯度存储更连续，通常有利于性能和显存整理。

### 4.9 `allgather_partitions` / `reduce_scatter`

这是 ZeRO 通信相关的性能选项。

可以先保留默认值，不要一开始就改。

## 5. 这个仓库里 DeepSpeed 分支的行为

你需要知道几件事：

- `--use_deepspeed 1` 会切到 DeepSpeed 路径
- DeepSpeed 配置文件路径由 `--ds_config` 指定
- DeepSpeed checkpoint 目录由 `--ds_ckpt_dir` 控制
- DeepSpeed 模式下，脚本会避开普通 PyTorch 的 `autocast` / `GradScaler` 路径
- DeepSpeed 模式下，`torch.compile` 默认被禁用，先减少变量

也就是说，DeepSpeed 分支不是简单换一个 launcher，而是训练逻辑本身会切换。

## 6. 推荐的学习顺序

如果你现在是第一次把这个 repo 跑到多卡 DeepSpeed，我建议按这个顺序学：

1. 先看 `train_pretrain.py` 里 `--use_deepspeed` 分支
2. 再看 `configs/deepspeed/pretrain_zero2_2x48.json`
3. 理解 `micro_batch`、`accumulation`、`global batch`
4. 理解 `ZeRO-2` 和 `ZeRO-3` 的区别
5. 学会用 `ds_report` 检查环境
6. 学会保存和恢复 DeepSpeed checkpoint
7. 最后再调 batch size、seq len、zero stage

## 7. 实战建议

### 先跑 ZeRO-2

不要一上来就 ZeRO-3。

原因很简单：

- ZeRO-2 更稳
- 报错更少
- 容易确认是不是环境问题
- 容易确认是不是数据 / 模型 / batch 设置问题

### 先小 batch 跑通

不要一上来就把 batch 拉大。

先保证：

- 能启动
- 能训练
- 能保存
- 能恢复

### 先固定一个配置

DeepSpeed 调参时，最好只改一个变量：

- 先改 `micro_batch`
- 再改 `accumulation`
- 再改 `zero stage`

不要同时改太多，否则很难定位问题。

## 8. 最小可用命令

```bash
deepspeed trainer/train_pretrain.py \
  --use_deepspeed 1 \
  --ds_config configs/deepspeed/pretrain_zero2_2x48.json \
  --model_config configs/model/pretrain_1b.json \
  --tokenizer_path /path/to/tokenizer \
  --data_path ../dataset/pretrain.jsonl
```

如果要续训：

```bash
deepspeed trainer/train_pretrain.py \
  --use_deepspeed 1 \
  --from_resume 1 \
  --ds_config configs/deepspeed/pretrain_zero2_2x48.json \
  --model_config configs/model/pretrain_1b.json \
  --tokenizer_path /path/to/tokenizer \
  --data_path ../dataset/pretrain.jsonl
```

## 9. 你后面最值得补的知识点

- `torchrun` 和 `deepspeed` launcher 的区别
- `DDP` 和 `ZeRO` 的区别
- `micro batch` vs `global batch`
- `gradient accumulation` 的本质
- `bf16`、`fp16`、`fp32` 的差异
- checkpoint 的保存与恢复逻辑

如果你先把这些点打通，后面再扩模型、扩数据、扩卡数会顺很多。
