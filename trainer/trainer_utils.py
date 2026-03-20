"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

MODEL_CONFIG_DEFAULTS = {
    'hidden_size': 512,
    'intermediate_size': None,
    'max_position_embeddings': 32768,
    'num_attention_heads': 8,
    'num_hidden_layers': 8,
    'num_key_value_heads': 2,
    'vocab_size': 6400,
    'hidden_act': 'silu',
    'dropout': 0.0,
    'rope_theta': 1000000.0,
    'flash_attn': True,
    'use_moe': False,
    'num_experts_per_tok': 2,
    'n_routed_experts': 4,
    'n_shared_experts': 1,
    'aux_loss_alpha': 0.01,
}

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode(local_rank=None, backend="nccl"):
    if dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", local_rank if local_rank is not None else 0))

    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend=backend)
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap_model(model):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw_model, '_orig_mod', raw_model)


def get_checkpoint_base_name(lm_config, weight='full_sft'):
    moe_suffix = '_moe' if lm_config.use_moe else ''
    return f'{weight}_{lm_config.hidden_size}{moe_suffix}'


def get_checkpoint_paths(lm_config, weight='full_sft', save_dir='../checkpoints'):
    base_name = get_checkpoint_base_name(lm_config, weight)
    return {
        'base_name': base_name,
        'weight_path': os.path.join(save_dir, f'{base_name}.pth'),
        'resume_path': os.path.join(save_dir, f'{base_name}_resume.pth'),
    }


def get_deepspeed_checkpoint_dir(lm_config, weight='pretrain', save_dir='../checkpoints_ds'):
    return os.path.join(save_dir, get_checkpoint_base_name(lm_config, weight))


def is_deepspeed_active(model):
    class_name = model.__class__.__name__.lower()
    module_name = model.__class__.__module__.lower()
    return 'deepspeed' in class_name or 'deepspeed' in module_name


def load_json_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_model_config(args):
    config_data = load_json_config(args.model_config) if getattr(args, 'model_config', None) else {}
    model_kwargs = dict(MODEL_CONFIG_DEFAULTS)
    model_kwargs.update(config_data)

    arg_overrides = {
        'hidden_size': getattr(args, 'hidden_size', None),
        'intermediate_size': getattr(args, 'intermediate_size', None),
        'max_position_embeddings': getattr(args, 'max_position_embeddings', None),
        'num_attention_heads': getattr(args, 'num_attention_heads', None),
        'num_hidden_layers': getattr(args, 'num_hidden_layers', None),
        'num_key_value_heads': getattr(args, 'num_key_value_heads', None),
        'vocab_size': getattr(args, 'vocab_size', None),
        'hidden_act': getattr(args, 'hidden_act', None),
        'dropout': getattr(args, 'dropout', None),
        'rope_theta': getattr(args, 'rope_theta', None),
        'flash_attn': None if getattr(args, 'flash_attn', None) is None else bool(args.flash_attn),
        'use_moe': None if getattr(args, 'use_moe', None) is None else bool(args.use_moe),
        'num_experts_per_tok': getattr(args, 'num_experts_per_tok', None),
        'n_routed_experts': getattr(args, 'n_routed_experts', None),
        'n_shared_experts': getattr(args, 'n_shared_experts', None),
        'aux_loss_alpha': getattr(args, 'aux_loss_alpha', None),
        'bos_token_id': getattr(args, 'bos_token_id', None),
        'eos_token_id': getattr(args, 'eos_token_id', None),
        'pad_token_id': getattr(args, 'pad_token_id', None),
    }
    for key, value in arg_overrides.items():
        if value is not None:
            model_kwargs[key] = value

    return MiniMindConfig(**model_kwargs)


def ensure_tokenizer_special_tokens(tokenizer):
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 必须提供 eos_token_id，当前预训练数据管线依赖它")

    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
        Logger('tokenizer 未设置 bos_token，已回退到 eos_token')

    if tokenizer.pad_token_id is None:
        fallback_token = tokenizer.eos_token or tokenizer.bos_token
        tokenizer.pad_token = fallback_token
        Logger('tokenizer 未设置 pad_token，已回退到 eos_token/bos_token')

    return tokenizer


def sync_model_config_with_tokenizer(lm_config, tokenizer):
    tokenizer = ensure_tokenizer_special_tokens(tokenizer)
    tokenizer_vocab_size = len(tokenizer)

    if lm_config.vocab_size != tokenizer_vocab_size:
        Logger(f'vocab_size 从 {lm_config.vocab_size} 同步为 tokenizer 词表大小 {tokenizer_vocab_size}')
        lm_config.vocab_size = tokenizer_vocab_size

    for attr_name in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
        tokenizer_value = getattr(tokenizer, attr_name, None)
        if tokenizer_value is None:
            continue
        if getattr(lm_config, attr_name, None) != tokenizer_value:
            Logger(f'{attr_name} 从 {getattr(lm_config, attr_name, None)} 同步为 tokenizer 的 {tokenizer_value}')
            setattr(lm_config, attr_name, tokenizer_value)

    return lm_config, tokenizer


def get_model_config_signature(lm_config):
    parts = [
        f'h{lm_config.hidden_size}',
        f'l{lm_config.num_hidden_layers}',
        f'a{lm_config.num_attention_heads}',
        f'kv{lm_config.num_key_value_heads}',
        f'ffn{lm_config.intermediate_size}',
        f'v{lm_config.vocab_size}',
        f'ctx{lm_config.max_position_embeddings}',
    ]
    if lm_config.use_moe:
        parts.extend([
            f'moe{lm_config.n_routed_experts}',
            f'topk{lm_config.num_experts_per_tok}',
            f'shared{lm_config.n_shared_experts}',
        ])
    return '_'.join(parts)


def save_model_config(lm_config, save_dir, save_filename='config.json'):
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, save_filename)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(lm_config.to_dict(), f, indent=2, ensure_ascii=False)
    return config_path


def get_deepspeed_zero_stage(ds_config):
    if isinstance(ds_config, str):
        ds_config = load_json_config(ds_config)
    zero_optimization = ds_config.get('zero_optimization', {})
    if isinstance(zero_optimization, bool):
        return int(zero_optimization)
    return int(zero_optimization.get('stage', 0))


def sync_deepspeed_train_args(args, ds_config):
    if isinstance(ds_config, str):
        ds_config = load_json_config(ds_config)

    micro_batch = ds_config.get('train_micro_batch_size_per_gpu')
    grad_acc = ds_config.get('gradient_accumulation_steps')

    if micro_batch is not None and args.batch_size != micro_batch:
        Logger(f'DeepSpeed train_micro_batch_size_per_gpu={micro_batch}，覆盖 batch_size={args.batch_size}')
        args.batch_size = micro_batch

    if grad_acc is not None and args.accumulation_steps != grad_acc:
        Logger(f'DeepSpeed gradient_accumulation_steps={grad_acc}，覆盖 accumulation_steps={args.accumulation_steps}')
        args.accumulation_steps = grad_acc

    return args


def get_wandb_run_id(wandb):
    if wandb is None:
        return None
    if hasattr(wandb, 'get_run'):
        run = wandb.get_run()
        return getattr(run, 'id', None) if run else None
    return getattr(wandb, 'id', None)


def save_full_weights(model, save_dir, save_filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_filename)

    if is_deepspeed_active(model):
        saved = model.save_16bit_model(save_dir, save_filename=save_filename)
        return save_path if saved else None

    state_dict = unwrap_model(model).state_dict()
    state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
    ckp_tmp = save_path + '.tmp'
    torch.save(state_dict, ckp_tmp)
    os.replace(ckp_tmp, save_path)
    del state_dict
    return save_path


def save_deepspeed_training_checkpoint(model_engine, checkpoint_dir, tag=None, client_state=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    return model_engine.save_checkpoint(checkpoint_dir, tag=tag, client_state=client_state or {}, save_latest=True)


def load_deepspeed_training_checkpoint(model_engine, checkpoint_dir, tag=None, **kwargs):
    if not os.path.isdir(checkpoint_dir):
        return None, None
    if tag is None and not os.path.exists(os.path.join(checkpoint_dir, 'latest')):
        return None, None
    return model_engine.load_checkpoint(checkpoint_dir, tag=tag, **kwargs)


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_paths = get_checkpoint_paths(lm_config, weight, save_dir)
    ckp_path = checkpoint_paths['weight_path']
    resume_path = checkpoint_paths['resume_path']

    if model is not None:
        state_dict = unwrap_model(model).state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = get_wandb_run_id(wandb)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda', move_to_device=True):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    lm_config, tokenizer = sync_model_config_with_tokenizer(lm_config, tokenizer)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        candidate_paths = []
        if os.path.isfile(from_weight):
            candidate_paths.append(from_weight)
        if not from_weight.endswith('.pth'):
            candidate_paths.append(os.path.join(save_dir, f'{from_weight}.pth'))
        candidate_paths.append(os.path.join(save_dir, f'{get_checkpoint_base_name(lm_config, from_weight)}.pth'))
        weight_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        if weight_path is None:
            raise FileNotFoundError(f'未找到待加载权重: from_weight={from_weight}, save_dir={save_dir}')
        map_location = device if move_to_device else 'cpu'
        weights = torch.load(weight_path, map_location=map_location)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    Logger(f'Tokenizer Path: {tokenizer_path}')
    if move_to_device:
        model = model.to(device)
    return model, tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
