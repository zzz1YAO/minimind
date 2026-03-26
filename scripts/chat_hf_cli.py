import argparse
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

warnings.filterwarnings('ignore')


def load_model_and_tokenizer(model_path, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    return model.eval().to(device), tokenizer


def trim_history(messages, history_turns):
    if history_turns <= 0:
        return []
    return messages[-history_turns * 2:]


def main():
    parser = argparse.ArgumentParser(description="Simple interactive HuggingFace chat for MiniMind")
    parser.add_argument('--model_path', type=str, default='MiniMind2-0_6B', help="HuggingFace格式模型目录")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="运行设备")
    parser.add_argument('--max_new_tokens', type=int, default=1024, help="单轮最大生成长度")
    parser.add_argument('--temperature', type=float, default=0.7, help="生成温度")
    parser.add_argument('--top_p', type=float, default=0.9, help="nucleus采样阈值")
    parser.add_argument('--history_turns', type=int, default=0, help="保留最近多少轮对话，0表示不保留")
    parser.add_argument('--do_sample', type=int, choices=[0, 1], default=0, help="是否采样生成（0=贪心，1=采样）")
    parser.add_argument('--prompt_mode', type=str, choices=['chat', 'pretrain'], default='chat', help="提示词模式：chat使用对话模板，pretrain使用纯续写")
    parser.add_argument('--dtype', type=str, choices=['float16', 'bfloat16', 'float32'], default='float16', help="模型加载精度")
    parser.add_argument('--seed', type=int, default=None, help="随机种子，可选")
    args = parser.parse_args()

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, dtype_map[args.dtype])
    conversation = []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("Enter an empty line to exit.")
    while True:
        prompt = input("user> ").strip()
        if not prompt:
            break

        conversation = trim_history(conversation, args.history_turns)
        conversation.append({"role": "user", "content": prompt})

        if args.prompt_mode == 'chat':
            prompt_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = f"{tokenizer.bos_token}{prompt}"
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True).to(args.device)

        print("assistant> ", end="", flush=True)
        with torch.no_grad():
            do_sample = bool(args.do_sample)
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": args.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "streamer": streamer,
            }
            if do_sample:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p
                generation_kwargs["remove_invalid_values"] = True

            generated_ids = model.generate(**generation_kwargs)

        response = tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        conversation.append({"role": "assistant", "content": response})
        print()


if __name__ == "__main__":
    main()
