import os
import sys
import shutil

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    SkipBatchSampler,
    Logger,
    build_model_config,
    get_checkpoint_snapshot_name,
    get_deepspeed_zero_stage,
    get_model_config_signature,
    get_lr,
    get_wandb_run_id,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    load_json_config,
    load_deepspeed_training_checkpoint,
    save_deepspeed_training_checkpoint,
    save_full_weights,
    save_model_config,
    prune_deepspeed_checkpoint_history,
    prune_versioned_checkpoint_files,
    setup_seed,
    sync_deepspeed_train_args,
)

warnings.filterwarnings('ignore')


def require_deepspeed():
    try:
        import deepspeed
    except ImportError as exc:
        raise ImportError("DeepSpeed 未安装，请先参考 docs/deepspeed_setup.md 完成安装。") from exc
    return deepspeed


def init_wandb(resume_id=None):
    if not args.use_wandb or not is_main_process():
        return None

    import wandb

    resume = 'must' if resume_id else None
    wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    wandb.init(project=args.wandb_project, name=wandb_run_name, id=resume_id, resume=resume)
    return wandb


def log_train_status(epoch, micro_step, total_steps, processed_steps, loss_value, aux_loss, current_lr, start_time, wandb=None):
    spend_time = time.time() - start_time
    current_logits_loss = loss_value - aux_loss
    eta_min = spend_time / max(processed_steps, 1) * total_steps // 60 - spend_time // 60
    Logger(
        f'Epoch:[{epoch + 1}/{args.epochs}]({micro_step}/{total_steps}), '
        f'loss: {loss_value:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {aux_loss:.4f}, '
        f'lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min'
    )
    if wandb:
        wandb.log({
            "loss": loss_value,
            "logits_loss": current_logits_loss,
            "aux_loss": aux_loss,
            "learning_rate": current_lr,
            "epoch_time": eta_min
        })


def should_save_checkpoint(epoch, micro_step, total_steps, global_step):
    if args.save_by == 'step':
        return global_step % args.save_interval == 0 or micro_step == total_steps

    if args.save_by == 'epoch':
        if micro_step != total_steps:
            return False
        return (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1

    raise ValueError(f'不支持的保存粒度: {args.save_by}')


def maybe_save_pytorch_checkpoint(epoch, micro_step, total_steps, global_step, model, optimizer, scaler, wandb):
    if not should_save_checkpoint(epoch, micro_step, total_steps, global_step):
        return

    lm_checkpoint(
        lm_config,
        weight=checkpoint_weight_name,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        step=micro_step,
        global_step=global_step,
        wandb=wandb,
        save_dir=args.save_dir,
        resume_dir='../checkpoints',
        snapshot_name=get_checkpoint_snapshot_name(args.save_by, epoch, global_step),
        keep_last=args.max_ckpts,
    )


def maybe_save_deepspeed_checkpoint(epoch, micro_step, total_steps, global_step, model_engine, wandb):
    if not should_save_checkpoint(epoch, micro_step, total_steps, global_step):
        return
    if not model_engine.is_gradient_accumulation_boundary():
        return

    model_engine.eval()
    snapshot_name = get_checkpoint_snapshot_name(args.save_by, epoch, global_step)
    if args.save_full_weights == 1:
        latest_weight_path = save_full_weights(model_engine, args.save_dir, save_filename)
        snapshot_weight_path = os.path.join(args.save_dir, f'{checkpoint_weight_name}_{snapshot_name}.pth')
        if latest_weight_path is not None and snapshot_weight_path != latest_weight_path:
            shutil.copy2(latest_weight_path, snapshot_weight_path)
        prune_versioned_checkpoint_files(args.save_dir, checkpoint_weight_name, args.max_ckpts, kind='weight')

    client_state = {
        'epoch': epoch,
        'step': micro_step,
        'global_step': global_step,
        'world_size': dist.get_world_size() if dist.is_initialized() else 1,
    }
    wandb_id = get_wandb_run_id(wandb)
    if wandb_id:
        client_state['wandb_id'] = wandb_id

    tag = snapshot_name
    save_deepspeed_training_checkpoint(model_engine, ds_checkpoint_dir, tag=tag, client_state=client_state)
    prune_deepspeed_checkpoint_history(ds_checkpoint_dir, args.max_ckpts)
    model_engine.train()


def train_epoch_pytorch(epoch, loader, total_steps, start_step=0, wandb=None):
    start_time = time.time()
    for processed_steps, (input_ids, labels) in enumerate(loader, start=1):
        micro_step = start_step + processed_steps
        global_step = epoch * total_steps + micro_step
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * total_steps + micro_step, args.epochs * total_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            raw_loss = res.loss + res.aux_loss
            loss = raw_loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if micro_step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if micro_step % args.log_interval == 0 or micro_step == total_steps:
            current_loss = raw_loss.item()
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            log_train_status(epoch, micro_step, total_steps, processed_steps, current_loss, current_aux_loss, current_lr, start_time, wandb)

        if is_main_process():
            maybe_save_pytorch_checkpoint(epoch, micro_step, total_steps, global_step, model, optimizer, scaler, wandb)

        del input_ids, labels, res, loss, raw_loss


def train_epoch_deepspeed(epoch, loader, total_steps, start_step=0, wandb=None):
    start_time = time.time()
    for processed_steps, (input_ids, labels) in enumerate(loader, start=1):
        micro_step = start_step + processed_steps
        global_step = epoch * total_steps + micro_step
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        lr = get_lr(epoch * total_steps + micro_step, args.epochs * total_steps, args.learning_rate)
        if model.optimizer is not None:
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = lr

        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss
        model.backward(loss)
        model.step()

        if micro_step % args.log_interval == 0 or micro_step == total_steps:
            current_loss = loss.item()
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_lr = model.optimizer.param_groups[-1]['lr'] if model.optimizer is not None else lr
            log_train_status(epoch, micro_step, total_steps, processed_steps, current_loss, current_aux_loss, current_lr, start_time, wandb)

        maybe_save_deepspeed_checkpoint(epoch, micro_step, total_steps, global_step, model, wandb)
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--model_config", type=str, default=None, help="模型结构配置文件路径（config.json）")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="tokenizer目录或HF模型名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_by", type=str, default="step", choices=["step", "epoch"], help="checkpoint保存粒度（step=按batch步，epoch=按轮）")
    parser.add_argument("--save_interval", type=int, default=1000, help="在所选粒度下的保存间隔")
    parser.add_argument("--max_ckpts", type=int, default=None, help="最多保留的历史checkpoint快照数量；None表示不限制")
    parser.add_argument('--hidden_size', default=None, type=int, help="隐藏层维度，优先级高于model_config")
    parser.add_argument('--intermediate_size', default=None, type=int, help="FFN中间层维度")
    parser.add_argument('--num_hidden_layers', default=None, type=int, help="隐藏层数量，优先级高于model_config")
    parser.add_argument('--num_attention_heads', default=None, type=int, help="注意力头数")
    parser.add_argument('--num_key_value_heads', default=None, type=int, help="KV头数（GQA/MQA）")
    parser.add_argument('--vocab_size', default=None, type=int, help="词表大小；默认会被tokenizer自动同步")
    parser.add_argument('--max_position_embeddings', default=None, type=int, help="模型位置编码长度")
    parser.add_argument('--hidden_act', default=None, type=str, help="激活函数")
    parser.add_argument('--dropout', default=None, type=float, help="dropout比例")
    parser.add_argument('--rope_theta', default=None, type=float, help="RoPE基数")
    parser.add_argument('--flash_attn', default=None, type=int, choices=[0, 1], help="是否开启SDPA/Flash Attention路径")
    parser.add_argument('--bos_token_id', default=None, type=int, help="BOS token id；默认从tokenizer同步")
    parser.add_argument('--eos_token_id', default=None, type=int, help="EOS token id；默认从tokenizer同步")
    parser.add_argument('--pad_token_id', default=None, type=int, help="PAD token id；默认从tokenizer同步")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=None, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--num_experts_per_tok', default=None, type=int, help="每个token激活的专家数")
    parser.add_argument('--n_routed_experts', default=None, type=int, help="路由专家总数")
    parser.add_argument('--n_shared_experts', default=None, type=int, help="共享专家数")
    parser.add_argument('--aux_loss_alpha', default=None, type=float, help="MoE辅助loss权重")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--use_deepspeed", default=0, type=int, choices=[0, 1], help="是否使用DeepSpeed训练（0=否，1=是）")
    parser.add_argument("--ds_config", type=str, default="../configs/deepspeed/pretrain_zero2_2x48.json", help="DeepSpeed配置文件路径")
    parser.add_argument("--ds_ckpt_dir", type=str, default="../checkpoints_ds", help="DeepSpeed检查点根目录")
    parser.add_argument("--save_full_weights", default=1, type=int, choices=[0, 1], help="DeepSpeed分支是否额外导出单文件权重")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式launcher兼容参数")
    args = parser.parse_args()

    ds_config = load_json_config(args.ds_config) if args.use_deepspeed == 1 else None
    zero_stage = get_deepspeed_zero_stage(ds_config) if ds_config is not None else 0
    if ds_config is not None:
        sync_deepspeed_train_args(args, ds_config)

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode(local_rank=args.local_rank if args.local_rank >= 0 else None)
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    if args.use_deepspeed == 1 and args.use_compile == 1:
        args.use_compile = 0
        Logger('DeepSpeed 模式下已禁用 torch.compile，以降低首版集成噪音')
    if args.use_deepspeed == 1:
        Logger(f'DeepSpeed enabled, ZeRO stage = {zero_stage}')

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = build_model_config(args)

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if args.use_deepspeed == 1 or device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        move_to_device=(args.use_deepspeed == 0)
    )
    model_signature = get_model_config_signature(lm_config)
    checkpoint_weight_name = f'{args.save_weight}_{model_signature}'
    save_filename = f'{checkpoint_weight_name}.pth'
    config_filename = f'{checkpoint_weight_name}.json'
    ds_checkpoint_dir = os.path.join(args.ds_ckpt_dir, checkpoint_weight_name)
    if is_main_process():
        saved_config_path = save_model_config(lm_config, args.save_dir, config_filename)
        Logger(f'已保存本次训练使用的模型配置: {saved_config_path}')
    if args.use_compile == 1 and args.use_deepspeed == 0:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[rank {rank}] before dataset", flush=True)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    print(f"[rank {rank}] after dataset", flush=True)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16' and device_type == 'cuda')) if args.use_deepspeed == 0 else None


    if dist.is_initialized():
        rank = dist.get_rank()
        t = torch.tensor([rank + 1.0], device=args.device)
        print(f"[rank {rank}] before all_reduce: {t.item()}", flush=True)
        dist.all_reduce(t)
        print(f"[rank {rank}] after all_reduce: {t.item()}", flush=True)
    if args.save_interval < 1:
        raise ValueError('--save_interval 必须是正整数')
    if args.max_ckpts is not None and args.max_ckpts < 1:
        raise ValueError('--max_ckpts 必须是正整数或留空')
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    wandb_resume_id = None
    if args.use_deepspeed == 1:


        deepspeed = require_deepspeed()
        print(f"[rank {rank}] before deepspeed.initialize", flush=True)
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=model.parameters(),
            config=args.ds_config
        )
        print(f"[rank {rank}] after deepspeed.initialize", flush=True)
        if args.from_resume == 1:
            load_path, client_state = load_deepspeed_training_checkpoint(model, ds_checkpoint_dir)
            if load_path is not None:
                start_epoch = client_state.get('epoch', 0)
                start_step = client_state.get('step', 0)
                wandb_resume_id = client_state.get('wandb_id')
                saved_ws = client_state.get('world_size')
                current_ws = dist.get_world_size() if dist.is_initialized() else 1
                if saved_ws is not None and saved_ws != current_ws and is_main_process():
                    Logger(f'DeepSpeed检查点 world_size={saved_ws}，当前 world_size={current_ws}，请确认该恢复流程兼容当前并行规模')
    else:
        ckp_data = lm_checkpoint(lm_config, weight=checkpoint_weight_name, save_dir='../checkpoints') if args.from_resume == 1 else None
        if ckp_data:
            model.load_state_dict(ckp_data['model'])
            optimizer.load_state_dict(ckp_data['optimizer'])
            scaler.load_state_dict(ckp_data['scaler'])
            start_epoch = ckp_data['epoch']
            start_step = ckp_data.get('step', 0)
            wandb_resume_id = ckp_data.get('wandb_id')

    # ========== 7. 配wandb & DDP包模型 ==========
    wandb = init_wandb(wandb_resume_id)
    if args.use_deepspeed == 0 and dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        total_steps = len(loader) + skip if skip > 0 else len(loader)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
        if args.use_deepspeed == 1:
            train_epoch_deepspeed(epoch, loader, total_steps, start_step=skip, wandb=wandb)
        else:
            train_epoch_pytorch(epoch, loader, total_steps, start_step=skip, wandb=wandb)
        start_step = 0

    # ========== 9. 清理分布进程 ==========
    if wandb and hasattr(wandb, 'finish') and is_main_process():
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()
