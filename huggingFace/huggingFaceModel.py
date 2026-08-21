import os
import torch
os.getenv("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForCausalLM

modal_name = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(modal_name)

input_prompt = ["The capital of India is"]

tokenized = tokenizer(input_prompt, return_tensors='pt')

model = AutoModelForCausalLM.from_pretrained(
    modal_name,
    dtype=torch.bfloat16
)

gen_result = model.generate(tokenized["input_ids"], max_new_tokens=25)
print("Input tokens: ", tokenized["input_ids"])
print("Output Tokens: ", gen_result)

gen_text = tokenizer.batch_decode(gen_result)
print(gen_text)