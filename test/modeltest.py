import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/root/autodl-tmp/ScholarAlign/models/base/Qwen2.5-7B-Instruct"

print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())

torch.cuda.reset_peak_memory_stats()

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
)

print(
    "After loading allocated:",
    torch.cuda.memory_allocated() / 1024**3,
    "GB"
)

print(
    "After loading reserved:",
    torch.cuda.memory_reserved() / 1024**3,
    "GB"
)

messages = [
    {
        "role": "user",
        "content": "What is supervised fine-tuning? Answer in two sentences."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True
)

print("\nResponse:")
print(response)

print(
    "\nPeak allocated:",
    torch.cuda.max_memory_allocated() / 1024**3,
    "GB"
)

import json
import os

metrics = {
    "experiment_id": "BASE-001",
    "stage": "base_inference_test",
    "model": "Qwen2.5-7B-Instruct",
    "gpu": torch.cuda.get_device_name(0),
    "dtype": "bfloat16",
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "memory_allocated_after_load_gb": round(
        torch.cuda.memory_allocated() / 1024**3, 4
    ),
    "memory_reserved_after_load_gb": round(
        torch.cuda.memory_reserved() / 1024**3, 4
    ),
    "peak_memory_allocated_gb": round(
        torch.cuda.max_memory_allocated() / 1024**3, 4
    ),
    "max_new_tokens": 100,
    "do_sample": False,
    "response": response
}

output_dir = "../outputs/experiments/BASE-001"
os.makedirs(output_dir, exist_ok=True)

with open(
    os.path.join(output_dir, "metrics.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)