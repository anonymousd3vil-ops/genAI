import os
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-6f8d7f2a-795d-4a5e-812d-c52a50e1d180")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-60cf399d-e3e1-4761-9e3b-c71ee898feac")
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")

# Drop-in replacement to get full logging by changing only the import
from langfuse.openai import OpenAI

# Configure the OpenAI client to use http://localhost:11434/v1 as base url
client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
    model="llama3.1:latest",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who was the first person to step on the moon?"},
        {"role": "assistant", "content": "Neil Armstrong was the first person to step on the moon on July 20, 1969, during the Apollo 11 mission."},
        {"role": "user", "content": "What were his first words when he stepped on the moon?"}
    ]
)
print(response.choices[0].message.content)