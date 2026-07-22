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
Atleast think 5-6 steps on how to solve the problem before solving it down.

The steps are you get a user input, you analayse, you think, you again think for several times and then return an output with explaination and finally you validate the output as well before giving final result

Follow the steps in squuence that is "analyse", "think", "output", "validate" and finally "result"

Rules: 
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input.
3. Carefully analyse

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
        model="gemini-3.5-flash-lite",
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