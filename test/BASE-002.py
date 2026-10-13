import os
import json
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

model_path = "/root/autodl-tmp/ScholarAlign/models/base/Qwen2.5-7B-Instruct"

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())

tokenizer = AutoTokenizer.from_pretrained(model_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
)

load_allocated_gb = torch.cuda.memory_allocated() / 1024**3
load_reserved_gb = torch.cuda.memory_reserved() / 1024**3

print(f"After loading allocated: {load_allocated_gb:.4f} GB")
print(f"After loading reserved: {load_reserved_gb:.4f} GB")

messages = [
    {
        "role": "user",
        "content": "What is supervised fine-tuning? Answer in two sentences."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True,
)

peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3

print("\nResponse:")
print(response)

print(f"\nPeak allocated: {peak_allocated_gb:.4f} GB")

metrics = {
    "experiment_id": "BASE-002",
    "stage": "4bit_base_inference",
    "model": "Qwen2.5-7B-Instruct",
    "gpu": torch.cuda.get_device_name(0),
    "base_storage_precision": "4bit",
    "quantization_type": "NF4",
    "compute_dtype": "bfloat16",
    "double_quantization": True,
    "memory_allocated_after_load_gb": round(load_allocated_gb, 4),
    "memory_reserved_after_load_gb": round(load_reserved_gb, 4),
    "peak_memory_allocated_gb": round(peak_allocated_gb, 4),
    "max_new_tokens": 100,
    "do_sample": False,
    "response": response,
}

output_dir = "/root/autodl-tmp/ScholarAlign/outputs/experiments/BASE-002"
os.makedirs(output_dir, exist_ok=True)

with open(
    os.path.join(output_dir, "metrics.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("\nMetrics saved to:")
print(os.path.join(output_dir, "metrics.json"))