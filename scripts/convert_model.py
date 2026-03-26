import os
import sys
import json
import argparse

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore', category=UserWarning)


DEFAULT_MODEL_CONFIG = {
    "hidden_size": 512,
    "intermediate_size": None,
    "max_position_embeddings": 32768,
    "num_attention_heads": 8,
    "num_hidden_layers": 8,
    "num_key_value_heads": 2,
    "vocab_size": 6400,
    "hidden_act": "silu",
    "dropout": 0.0,
    "rope_theta": 1000000.0,
    "flash_attn": True,
    "use_moe": False,
}


def load_model_config(config_path=None):
    if config_path is None:
        return MiniMindConfig(**DEFAULT_MODEL_CONFIG)

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    return MiniMindConfig(**config_data)


def resolve_dtype(dtype_name):
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"不支持的dtype: {dtype_name}，可选值: {', '.join(dtype_map)}")
    return dtype_map[dtype_name]


def save_tokenizer(tokenizer_path, transformers_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(transformers_path)
    # 兼容 transformers-5.0 的写法
    config_path = os.path.join(transformers_path, "tokenizer_config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        tokenizer_config = json.load(f)
    tokenizer_config.update({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "extra_special_tokens": {},
    })
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)


# MoE模型需使用此函数转换
def convert_torch2transformers_minimind(torch_path, transformers_path, lm_config, tokenizer_path='../model', dtype=torch.float16):
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    save_tokenizer(tokenizer_path, transformers_path)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


# LlamaForCausalLM结构兼容第三方生态
def convert_torch2transformers_llama(torch_path, transformers_path, lm_config, tokenizer_path='../model', dtype=torch.float16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    intermediate_size = lm_config.intermediate_size
    if intermediate_size is None:
        intermediate_size = 64 * ((int(lm_config.hidden_size * 8 / 3) + 64 - 1) // 64)
    llama_config = LlamaConfig(
        vocab_size=lm_config.vocab_size,
        hidden_size=lm_config.hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=lm_config.num_hidden_layers,
        num_attention_heads=lm_config.num_attention_heads,
        num_key_value_heads=lm_config.num_key_value_heads,
        max_position_embeddings=lm_config.max_position_embeddings,
        rms_norm_eps=lm_config.rms_norm_eps,
        rope_theta=lm_config.rope_theta,
        tie_word_embeddings=True
    )
    llama_model = LlamaForCausalLM(llama_config)
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model = llama_model.to(dtype)  # 转换模型权重精度
    llama_model.save_pretrained(transformers_path)
    model_params = sum(p.numel() for p in llama_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    save_tokenizer(tokenizer_path, transformers_path)
    print(f"模型已保存为 Transformers-Llama 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
    print(f"模型已保存为 PyTorch 格式 (half精度): {torch_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert MiniMind torch checkpoint to HuggingFace format")
    parser.add_argument('--torch_path', type=str, required=True, help="待转换的 .pth 权重路径")
    parser.add_argument('--transformers_path', type=str, required=True, help="输出的 HuggingFace 目录")
    parser.add_argument('--model_config', type=str, default=None, help="模型配置 JSON 路径，例如 configs/model/pretrain_0_6b.json")
    parser.add_argument('--tokenizer_path', type=str, default='../model', help="tokenizer 目录或 HF tokenizer 名称")
    parser.add_argument('--format', type=str, default='auto', choices=['auto', 'llama', 'minimind'], help="输出格式，auto 会根据 use_moe 自动选择")
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'], help="导出权重精度")
    args = parser.parse_args()

    lm_config = load_model_config(args.model_config)
    dtype = resolve_dtype(args.dtype)

    export_format = args.format
    if export_format == 'auto':
        export_format = 'minimind' if lm_config.use_moe else 'llama'

    if export_format == 'minimind' and not lm_config.use_moe:
        print('警告: 当前配置 use_moe=False，但你选择了 minimind 格式导出。')

    if export_format == 'minimind':
        convert_torch2transformers_minimind(
            args.torch_path,
            args.transformers_path,
            lm_config=lm_config,
            tokenizer_path=args.tokenizer_path,
            dtype=dtype,
        )
    else:
        convert_torch2transformers_llama(
            args.torch_path,
            args.transformers_path,
            lm_config=lm_config,
            tokenizer_path=args.tokenizer_path,
            dtype=dtype,
        )


if __name__ == '__main__':
    main()
