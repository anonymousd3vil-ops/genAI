from mem0 import Memory
import ollama

# import logging
# logging.basicConfig(level=logging.DEBUG)

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "VeSBEeylDgvlqppqv2sJpZX9bBxe8IMTO6i7QP0haso"

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "qwen3-embedding:4b",
            "ollama_base_url": "http://localhost:11434"
        }
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:latest",
            "temperature": 0.1,
            "max_tokens": 2000
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "embedding_model_dims": 2560
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.1:latest",
                "temperature": 0.1,
                "max_tokens": 2000,
                "ollama_base_url": "http://localhost:11434"
            }
        }
    }
}

mem_client = Memory.from_config(config)

def chat(message):
    messages = [
        { "role": "user", "content": message}
    ]
    result = ollama.chat(
        model="qwen3:4b",
        messages=messages
    )

    response = result["message"]["content"]
    messages.append({"role": "assistant", "content": response})

    # result = mem_client.add( messages, user_id="anonymousd3vil")
    res = mem_client.add("Vivek is a BTech CSE student at NIT Agartala who is learning about knowledge graphs.", user_id="test_graph")
    print(res)
    
    return response

while True:

    message = input(">> ")

    if message.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    print("BOT:", chat(message))