import json
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

systemPrompt = """
You are an expert assistant who thinks a query from four diffrent perspectives and gives one consistent answer which matches most with the all four perspective
For a given user input, you have to think about the query in four possible direction and give one consistent output

Rules: 
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input.
3. Carefully analyse

Perspectives: 
Perspective 1: Logical reasoning
Perspective 2: Practical/Real-world reasoning
Perspective 3: Critical reasoning (find weaknesses)
Perspective 4: Creative/Alternative reasoning

Each Perspective shuld have the following four steps "analyse", "think", "output", "validate" and finally "result"

Output Schema:
{
    "perspective": "Perspective 1"
    "steps" : {
        "analysis": ".....",
        "thinking": ".....",
        "output": ".....",
        "result": "....."
    }
    "perspective": "Perspective 2"
    "steps" : {
        "analysis": ".....",
        "thinking": ".....",
        "output": ".....",
        "result": "....."
    }
    "perspective": "Perspective 3"
    "steps" : {
        "analysis": ".....",
        "thinking": ".....",
        "output": ".....",
        "result": "....."
    }
    "perspective": "Perspective 4"
    "steps" : {
        "analysis": ".....",
        "thinking": ".....",
        "output": ".....",
        "result": "....."
    }
    "finalResult": "......"
}

Final result in the Output Schema shuld have the crux or comman part of all four perspective
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

query = "What is Love?"

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

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=history
)

text = response.text.strip()
if text.startswith("```"):
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

try:
    parsed = json.loads(text)

except Exception:
    print("An error occured")
    print(text)
    

print(parsed)

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

