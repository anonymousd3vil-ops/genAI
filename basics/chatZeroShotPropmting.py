from dotenv import load_dotenv
from google import genai
import os

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Generate response
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="What is my name?"  #directly asked question
)

print(response.text)

"""
# this is to check which model is available associated with my api key

for model in client.models.list():
    print(model.name)

"""
