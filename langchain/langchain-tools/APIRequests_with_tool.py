from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
import requests


from dotenv import load_dotenv
load_dotenv()

@tool
def fetch_pokemon_details(name: str):
    """
        This tool is helpful to fetch the pokemon details of given pokemon name
    """
    url = f"https://pokeapi.co/api/v2/pokemon/{name}"

    response = requests.get(url)
    data = response.json()

    parsed_data = {
        "id": data["id"],
        "name":data["name"],
        "height":data["height"],
        'weight':data["weight"],
        'type': data["types"][0]["type"]["name"],
        "image_url":data["sprites"]["other"]["dream_world"]["front_default"]
    }

    return parsed_data

tools = [fetch_pokemon_details]
tools_by_name = {tool.name: tool for tool in tools}
# print(tools_by_name)

llm = ChatOllama(
    model="qwen3:4b",
    base_url="http://localhost:11434",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)

langfuse = get_client()
langfuse_handler = CallbackHandler()

query=HumanMessage("Give me details of Mew, MewTwo, Pikachu, bulbasaur also, compare them in tabular format!!")

messages=[query]



result = llm_with_tools.invoke(
    messages,
    config={"callbacks": [langfuse_handler]},
)

messages.append(result)

for tool_call in result.tool_calls:
    selected_tool = tools_by_name[tool_call["name"]]

    tool_result = selected_tool.invoke(tool_call["args"])

    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

response = llm_with_tools.invoke(
    messages,
    config={"callbacks": [langfuse_handler]}
)

print(response.content)

# print(fetch_pokemon_details.invoke({"name":"pikachu"}))