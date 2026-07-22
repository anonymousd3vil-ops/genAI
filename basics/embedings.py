import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Create client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

text = "Taj Mahal is an iconic landmark in Agra, India. It was built by Shah Jahan."

# Generate embedding
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text
)

print("Vector Embedding:")
print(response.embeddings[0].values)