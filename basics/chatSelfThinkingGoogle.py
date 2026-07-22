import json
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

systemPrompt = """
You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.

For the given user input, analyse the input and break down the problem step by step.

At least think 5-6 steps before solving it.

Follow the sequence:

analyse
think
output
validate
result

IMPORTANT RULES:

1. Return ONLY ONE JSON object.
2. Never return markdown.
3. Never wrap JSON inside ```.

Output format:

{
    "step":"analyse | think | output | validate | result",
    "content":"..."
}

When the user first asks a question,
ONLY return the ANALYSE step.

After that, whenever the user says "continue",
return ONLY the NEXT step.

Never skip any step.
"""

history = [
    {
        "role": "user",
        "parts": [
            {
                "text": systemPrompt
            }
        ]
    }
]

query = input("> ")

history.append(
    {
        "role": "user",
        "parts": [
            {
                "text": query
            }
        ]
    }
)

while True:

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=history
    )

    text = response.text.strip()

    try:
        parsed = json.loads(text)

    except Exception:

        print(text)
        break

    print(f"🧠 {parsed['step'].upper()} : {parsed['content']}")

    if parsed["step"] == "result":
        break

    history.append(
        {
            "role":"model",
            "parts":[
                {
                    "text":text
                }
            ]
        }
    )

    history.append(
        {
            "role":"user",
            "parts":[
                {
                    "text":"continue"
                }
            ]
        }
    )   