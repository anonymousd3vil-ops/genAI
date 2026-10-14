from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from dotenv import load_dotenv
load_dotenv()

@tool
def addition(a: int, b: int) -> int:
    """This Adds given two number and returns the value"""
    return a+b

@tool
def multiplication(a: int, b: int) -> int:
    """This multiplies given two number and returns the value"""
    return a*b

@tool
def subtraction(a: int, b: int) -> int:
    """This subtracts given two number and returns the value"""
    return a-b

@tool
def division(a: int, b: int) -> int:
    """This divides given two number and returns the value"""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    
    return a/b

tools = [addition, multiplication, subtraction, division]
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

query=HumanMessage("Do the given calculations, first add 3 with 10, after addition divide the result with 13 and finally subtract 1 from the division result.")

messages=[query]

result = llm_with_tools.invoke(
    messages,
    config={"callbacks": [langfuse_handler]},
)

messages.append(result)

# print(multiplication.invoke({'name': 'multiplication', 'args': {'a': 13, 'b': 3}, 'id': '36b777af-f497-45ce-90d9-968fcef1def9', 'type': 'tool_call'}))

for tool_call in result.tool_calls:
    selected_tool = tools_by_name[tool_call["name"]]

    tool_result = selected_tool.invoke(tool_call["args"])

    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

# print("AI Reply: ", result.content)
# print("Called Tools: ", result.tool_calls)
# print(messages)

response = llm_with_tools.invoke(messages)
print(response.content)

# response = multiplication.invoke(result.tool_calls[0])
# print(response.content)
