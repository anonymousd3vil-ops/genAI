import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

os.getenv("HF_TOKEN")

modal_name = "google/gemma-3-1b-it"  # here i am using google's gemma-3 model

tokenizer = AutoTokenizer.from_pretrained(modal_name)

# print(tokenizer.get_vocab())
input_tokens = tokenizer("Hello, This is Vivek Patel!!")
# print(input_tokens['input_ids'])

model = AutoModelForCausalLM.from_pretrained(
    modal_name,
    dtype=torch.bfloat16
)

gen_pipline = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(gen_pipline("Hey, There"))