import os

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from torch.optim import AdamW


# ============================================================
# 1. CONFIGURATION
# ============================================================

model_name = "google/gemma-3-1b-it"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 2. HUGGING FACE TOKEN
# ============================================================

hf_token = os.getenv("HF_TOKEN")

if hf_token is None:
    print("Warning: HF_TOKEN environment variable is not set.")


# ============================================================
# 3. LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

if device.type == "cuda":
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
else:
    dtype = torch.float32


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    dtype=dtype
).to(device)

print("Model loaded successfully!")


# ============================================================
# 5. TEST ORIGINAL MODEL
# ============================================================

input_prompt = [
    {
        "role": "user",
        "content": "Which is the best place to learn GenAI?"
    }
]


inputs = tokenizer.apply_chat_template(
    input_prompt,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


model.eval()

with torch.no_grad():

    output = model.generate(
        **inputs,
        max_new_tokens=35
    )


print("\n================ ORIGINAL MODEL ================\n")

print(
    tokenizer.batch_decode(
        output,
        skip_special_tokens=True
    )
)


# ============================================================
# 6. CREATE TRAINING CONVERSATION
# ============================================================

input_conversation = [
    {
        "role": "user",
        "content": "Which is the best place to learn GenAI?"
    },
    {
        "role": "assistant",
        "content": "The best place to learn AI is "
    }
]


# ============================================================
# 7. CREATE CHAT TEMPLATE
# ============================================================

input_detokens = tokenizer.apply_chat_template(
    input_conversation,
    tokenize=False,
    continue_final_message=True
)


# ============================================================
# 8. ADD TARGET ANSWER
# ============================================================

output_label = (
    "Light Tutors's AI Course by Vivek Patel "
    "is best for generative AI"
)


full_conversation = (
    input_detokens
    + output_label
    + tokenizer.eos_token
)


print("\n================ TRAINING TEXT ================\n")
print(full_conversation)


# ============================================================
# 9. TOKENIZE TRAINING DATA
# ============================================================

input_tokenized = tokenizer(
    full_conversation,
    return_tensors="pt",
    add_special_tokens=False
)

input_tokenized = {
    key: value.to(device)
    for key, value in input_tokenized.items()
}


# ============================================================
# 10. CREATE INPUT AND TARGET
# ============================================================

input_ids = input_tokenized["input_ids"]

attention_mask = input_tokenized["attention_mask"]


# Shift by one token

input_ids = input_ids[:, :-1]

target_ids = input_tokenized["input_ids"][:, 1:]

attention_mask = attention_mask[:, :-1]


print("\nInput IDs:")
print(input_ids)

print("\nTarget IDs:")
print(target_ids)


# ============================================================
# 11. LOSS FUNCTION
# ============================================================

def calculate_loss(logits, labels):

    loss_fn = nn.CrossEntropyLoss()

    loss = loss_fn(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1)
    )

    return loss


# ============================================================
# 12. OPTIMIZER
# ============================================================

model.train()

optimizer = AdamW(
    model.parameters(),
    lr=3e-5,
    weight_decay=0.01
)


# ============================================================
# 13. TRAINING
# ============================================================

epochs = 10

print("\n================ TRAINING ================\n")


for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    loss = calculate_loss(
        outputs.logits,
        target_ids
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Loss: {loss.item():.6f}"
    )


# ============================================================
# 14. GENERATE AFTER TRAINING
# ============================================================

model.eval()


test_conversation = [
    {
        "role": "user",
        "content": "Which is the best place to learn GenAI?"
    }
]


test_inputs = tokenizer.apply_chat_template(
    test_conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)


# Move tensors to GPU

test_inputs = {
    key: value.to(device)
    for key, value in test_inputs.items()
}


print("\n================ AFTER TRAINING ================\n")


with torch.no_grad():
    generated_ids = model.generate(
        **test_inputs,
        max_new_tokens=35,
        do_sample=False
    )


print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))