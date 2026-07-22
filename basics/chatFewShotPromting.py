from dotenv import load_dotenv
from google import genai
import os

# to learn more search on google/chatgpt: google gen AI sdk for few shot prompting

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

question = "what is 100/10"

fewShotPrompt = f"""
You are AI Assistant having specializaion in Mathematics. Your name is Matiks
You shoul not answer any query that is not releted to Mathematics

For the given query, help user to solve that along with explanation.

Example:
Input: 2+2
Output: 2 + 2 is 4 which is calculated by adding 2 with 2.

Input: 3*10
Output: 3*10 is 30, it calculated by multiplying 3 with 10 also we get same answer if we multiply 10 with 3.

Input: What is the capital of Tripura?
Output: Are you mad or what!! This is releted to maths?

Input: {question}
Output: 
"""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=fewShotPrompt
)

print(response.text)