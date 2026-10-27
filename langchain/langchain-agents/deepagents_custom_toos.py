
import os
from typing import Literal

import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from deepagents import create_deep_agent

from langfuse import get_client
from langfuse.langchain import CallbackHandler

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY is missing from your .env file")

tavily_client = TavilyClient(api_key=tavily_api_key)

langfuse = get_client()
langfuse_handler = CallbackHandler()


@tool
def fetch_pokemon_details(name: str):
    """Fetch details of a Pokémon by its name."""

    url = f"https://pokeapi.co/api/v2/pokemon/{name.lower()}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            "id": data["id"],
            "name": data["name"],
            "height": data["height"],
            "weight": data["weight"],
            "types": [item["type"]["name"] for item in data["types"]],
            "image_url": data["sprites"]["other"]["dream_world"]["front_default"],
        }

    except requests.RequestException as e:
        return {"error": f"Could not fetch Pokémon details: {e}"}
    

@tool
def internet_search(query: str, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = False):
    """
    Search the web for general research, current information,
    news, or facts not available through specialized tools.

    Do NOT use for basic Pokémon facts, stats, types, abilities,
    moves, height, weight, or artwork. Use fetch_pokemon_details
    for those requests instead.
    """
    return tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )

research_instructions = """
You are an expert research assistant with access to below given tools.
Your highest priority is to select the correct tool.

## TOOL ROUTING RULES
### A. fetch_pokemon_details — PRIORITY FOR POKÉMON FACTS
Whenever the user asks for specific information about a Pokémon,
you MUST call fetch_pokemon_details before answering if the
requested information is available through PokéAPI.

Examples:
- "Tell me about Pikachu."
- "What is Charizard's type?"
- "What are Mewtwo's base stats?"
- "Give me Bulbasaur's height and weight."
- "What abilities does Gengar have?"
- "Show me the moves and image of Lucario."
- "What is the ID of Eevee?"

For these questions:
1. Identify the Pokémon name.
2. Call fetch_pokemon_details with the correct name.
3. Use the returned data as the factual source.
4. Never call internet_search merely to retrieve these details.

If the name is misspelled or the API returns not found, do not
invent details. Explain the problem and ask for clarification
when necessary.
For multiple Pokémon, call fetch_pokemon_details separately
for each Pokémon being compared.

### B. internet_search — GENERAL RESEARCH
Use internet_search for:
- Current events and recent information.
- General research unrelated to Pokémon API lookups.
- Research that requires web sources.
- Pokémon lore, competitive analysis, historical context, or
  other information unavailable through PokéAPI.

If a question combines Pokémon API facts with broader research,
call fetch_pokemon_details for the API facts first. Use
internet_search only for the additional information that
PokéAPI cannot provide.

### C. TOOL DISCIPLINE
- Do not answer Pokémon-specific factual questions from memory
  when the requested facts can be fetched through PokéAPI.
- Never fabricate tool results.
- If a tool fails, explain the limitation clearly.
- Use the actual tool results in the final answer.
- Do not call internet_search when fetch_pokemon_details alone
  can fully answer the user's question.

## RESPONSE STYLE
Be accurate, clear, and concise.
For research requests, organize the answer into useful sections.
Distinguish retrieved facts from interpretations.
"""


agent = create_deep_agent(
    model="ollama:qwen3:4b",
    tools=[fetch_pokemon_details, internet_search],
    system_prompt=research_instructions,
    memory=["./AGENTS.md"],
)

def main():
    query = input(">> ").strip()

    if not query:
        print("Please enter a query.")
        return

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"callbacks": [langfuse_handler]},
        )

        print("\nAssistant:\n")
        print(result["messages"][-1].content)

    except Exception as exc:
        print(f"Agent execution failed: {exc}")


if __name__ == "__main__":
    main()
