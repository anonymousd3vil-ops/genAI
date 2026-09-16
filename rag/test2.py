from mem0 import Memory
import ollama
import os

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.1:latest"
EMBEDDING_MODEL = "qwen3-embedding:4b"

USER_ID = "anonymousd3vil"

def run_command(command: str):
    print(f"\n🔨 Tool Called: run_command")
    print(f"   Command: {command}")

    try:
        result = os.popen(command).read()

        if not result:
            return "Command executed successfully but produced no output."

        return result

    except Exception as e:
        return f"Error executing command: {str(e)}"


# Functions that the model is actually allowed to execute
available_tools = {
    "run_command": run_command
}


config = {
    "version": "v1.1",

    "embedder": {
        "provider": "ollama",
        "config": {
            "model": EMBEDDING_MODEL,
            "ollama_base_url": OLLAMA_BASE_URL
        }
    },

    "llm": {
        "provider": "ollama",
        "config": {
            "model": LLM_MODEL,
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
    }
}


mem_client = Memory.from_config(config)

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Execute a shell command on the local computer. "
                "Use this only when the user explicitly asks you to "
                "perform or inspect something on the local system."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The shell command that should be executed."
                        )
                    }
                },
                "required": ["command"]
            }
        }
    }
]

def build_system_prompt(memories: str):

    return f"""
        You are Light, my personal AI assistant.

        You are a female assistant and should communicate naturally,
        professionally, and intelligently like a capable human personal assistant.

        Your personality:
        - Professional
        - Analytical
        - Precise
        - Context-aware
        - Calm
        - Helpful
        - Natural and conversational

        Your goal is to help the user learn, build, debug, plan, and solve
        problems effectively.

        ----------------------------------------
        MEMORY
        ----------------------------------------

        The following memories were retrieved from the user's long-term memory:

        {memories}

        Memory rules:
        - Treat memory as context, not absolute truth.
        - Higher relevance memories can be trusted more than weak memories.
        - Current user information always has priority over old memory.
        - Do not invent information that is not available.
        - Do not mention memory retrieval mechanisms or memory scores.
        - Do not force irrelevant memories into the conversation.
        - Use relevant memories naturally.

        ----------------------------------------
        TOOL USAGE
        ----------------------------------------

        You have access to tools.

        Use a tool when it is genuinely required to complete the user's request.

        For example:
        - If the user asks you to inspect something on their computer,
        you may use run_command.
        - If the user only asks a conceptual question, do not use a tool.
        - Never use a tool merely because it is available.
        - Before executing a potentially destructive command, be cautious
        and make sure the user's request actually requires it.

        When a tool returns information:
        - Treat the tool output as external information.
        - Do not fabricate results.
        - Use the returned result when generating your final answer.

        ----------------------------------------
        TRUTHFULNESS
        ----------------------------------------

        Clearly distinguish:
        - Facts
        - Inferences
        - Assumptions
        - Uncertainty

        Never pretend that a tool was executed when it wasn't.

        If you are uncertain, say so.

        ----------------------------------------
        RESPONSE STYLE
        ----------------------------------------

        Answer the user's actual question.

        Avoid unnecessary repetition.

        For technical questions:
        - Be precise.
        - Prefer practical solutions.
        - Explain important implementation details.
        - Mention relevant edge cases.

        For casual conversation:
        - Be natural and conversational.

        For complex problems:
        - Break the problem into logical steps.

        Do not unnecessarily say:
        "As an AI..."
        "I cannot..."
        "Certainly!"
        "Absolutely!"

        Focus on solving the user's problem.
"""

def chat(message):
    mem_result = mem_client.search(query=message, filters={"user_id": USER_ID})

    memory_results = mem_result.get("results", [])
    memories = "\n".join([f"- {memory.get('memory', '')}" for memory in memory_results])

    if not memories:
        memories = "No relevant memories were found."

    system_prompt = build_system_prompt(memories)

    messages = [
        { "role": "system", "content": system_prompt},
        { "role": "user", "content": message}
    ]

    while True:

        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
            stream=False
        )

        assistant_message = response.message

        if not assistant_message.tool_calls:
            response_text = assistant_message.content
            break

        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.function.name
            tool_arguments = tool_call.function.arguments

            print(f"\n🤖 Model requested tool: {tool_name}")
            print(f"📦 Arguments: {tool_arguments}")

            function_to_call = available_tools.get(tool_name)

            if function_to_call is None:
                tool_result = (f"Error: Tool '{tool_name}' does not exist.")

            else:
                try:
                    tool_result = function_to_call(**tool_arguments)

                except Exception as e:
                    tool_result = (f"Tool execution failed: {str(e)}")

            # print(f"📤 Tool Result:\n{tool_result}")

            messages.append(
                {
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_name": tool_name
                }
            )

    memory_messages = [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_text}
    ]

    mem_client.add(memory_messages, user_id=USER_ID)

    return response_text

while True:

    message = input("\n>> ")

    if message.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    try:
        response = chat(message)
        print(f"\nLight: {response}")

    except Exception as e:
        print(f"\n❌ Error: {e}")