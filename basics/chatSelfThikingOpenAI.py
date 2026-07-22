import json
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

systemPrompt = """
You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.

For the given user input, analyse the input and break down the problem step by step.
Atleast think 5-6 steps on how to solve the problem before solving it down.

The steps are you get a user input, you analayse, you think, you again think for several times and then return an output with explaination and finally you validate the output as well before giving final result

Follow the steps in squuence that is "analyse", "think", "output", "validate" and finally "result"

Rules: 
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input.
3. Carefully analyse the user query.

Output Format: 
{{step: "string" content: "string"}}

Example: 
Input: What is 2 + 2?
Output: {{step: "analyse", content: "The user is intrested in maths query and He is asking basic Arithmetic Operation"}}
Output: {{step: "think", content: "To perform the addition I must go from left to right and add all the operands"}}
Output: {{step: "output", content: "4"}}
Output: {{step: "validate", content: "Seems like 4 is correct answer for 2 + 2"}}
Output: {{step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers"}}
"""

messages=[
    {"role": "system", "content": systemPrompt},
        
]

query = input("> ")
messages.append({"role": "user", "content": query})

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type":"json_object"},
        messages=messages
    )

    parsed_response = json.loads(response.choices[0].messages.content)
    messages.append({"role": "assistant", "content": json.dumps(parsed_response)})

    if parsed_response.get("step") != "output":
        print(f"🧠: {parsed_response.get("content")}")
        continue

    print(f"🤖: {parsed_response.get("content")}")
    break

# result = client.chat.completions.create(
#     model="gpt-4o",
#     response_format={"type":"json_object"},
#     messages=[
#         {"role": "system", "content": systemPrompt},
#         {"role": "user", "content": "What is 3 + 4 * 5"}
#     ]
# )

# print(result.choices[0].messages.content)