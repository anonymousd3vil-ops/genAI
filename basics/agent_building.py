import json
import requests
from dotenv import load_dotenv
# from langfuse import observe
from google import genai
import os

load_dotenv()

# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# --------------------------------------------------
# TOOLS
# --------------------------------------------------

# @observe()
def run_command(command):
    print("🔨 Tool Called: run_command", command)

    result = os.system(command)

    return result


# @observe()
def get_weather(city: str):
    print("🔨 Tool Called: get_weather", city)

    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"

    return "Something went wrong"


# @observe()
def add(x, y):
    print("🔨 Tool Called: add", x, y)
    return x + y

# --------------------------------------------------
# AVAILABLE TOOLS
# --------------------------------------------------

available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description":
            "Takes a city name as input and returns the current weather for the city"
    },
    "run_command": {
        "fn": run_command,
        "description":
            "Takes a command as input, executes it on the system and returns the result"
    },
    "add": {
        "fn": add,
        "description":
            "Takes two numbers as input and returns their sum"
    }
}


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

system_prompt = """
    You are a helpful AI Assistant specialized in resolving user queries.
    You work using the following process:
    START → PLAN → ACTION → OBSERVE → OUTPUT

    For every user query:
        1. Understand the user's query.
        2. Plan what needs to be done.
        3. Decide whether a tool is required.
        4. If a tool is required, call the appropriate tool.
        5. Wait for the tool observation.
        6. Based on the observation, continue reasoning.
        7. Finally provide the answer.

    RULES:
        - Always return valid JSON.
        - Perform only ONE step at a time.
        - After an ACTION, wait for the OBSERVATION.
        - Do not directly provide the final answer when a tool is required.
        - Carefully analyse the user query.
        - Use only the available tools.
        - The "function" field is required only when step = action.
        - The "input" field is required only when step = action.
        - "input" in action step when we got function name as add then use input schema as "input": {"x": first number, "y": second}

    OUTPUT JSON FORMAT:
    {
        "step": "plan | action | observe | output",
        "content": "string",
        "function": "function name",
        "input": "function input"
    }

    AVAILABLE TOOLS:
    1. get_weather
    Description: Takes a city name as input and returns the current weather.

    2. run_command
    Description: Takes a command as input, executes it on the system and returns the result.

    3. add
    Description: Takes two numbers and returns their sum.

    EXAMPLE:
    User: What is the weather in New York?
    Assistant: { "step": "plan", "content": "The user wants the current weather information for New York." }
    Assistant: { "step": "plan", "content": "I should use the get_weather tool."}
    Assistant: { "step": "action", "function": "get_weather", "input": "New York"}

    After receiving the tool result:
    Assistant: {"step": "observe", "content": "The weather in New York is Clear +25°C"}

    Finally:
    Assistant:{ "step": "output", "content": "The weather in New York is currently clear and 25°C."}

    EXAMPLE 2:
    User: What is the the sum of 12 and 23?
    Assistant: { "step": "plan", "content": "The user wants the sum of 12 and 23." }
    Assistant: { "step": "plan", "content": "I should use the add tool."}
    Assistant: { "step": "action", "function": "add", "input": {"x": 12, "y": 23}}

    After receiving the tool result:
    Assistant: {"step": "observe", "content": "The sum of given two numbers is 35"}

    Finally:
    Assistant:{ "step": "output", "content": "After Observing i got to know that the sum of given two number is correct and is 35"}


"""

# --------------------------------------------------
# JSON SCHEMA
# --------------------------------------------------

response_schema = {
    "type": "object",
    "properties": {
        "step": {
            "type": "string",
            "enum": [
                "plan",
                "action",
                "observe",
                "output"
            ]
        },
        "content": {
            "type": "string"
        },
        "function": {
            "type": ["string", "null"]
        },
        "input": {
            "type": ["string", "object", "number", "null"]
        }
    },
    "required": [
        "step",
        "content",
        "function",
        "input"
    ]
}

# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def ask_gemini(user_input, previous_id=None):
    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=user_input,
        previous_interaction_id=previous_id,
        system_instruction=system_prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": response_schema
        }
    )
    return interaction

# --------------------------------------------------
# MAIN AGENT LOOP
# --------------------------------------------------
previous_id = None

while True:
    user_query = input("> ")

    # ----------------------------------------------
    # NEW USER QUERY
    # ----------------------------------------------
    interaction = ask_gemini(user_query, previous_id)
    previous_id = interaction.id

    while True:

        # Get Gemini's JSON response
        parsed_output = json.loads(interaction.output_text)
        print("this is the parsed output", parsed_output, "ending of parsed output")
        # ------------------------------------------
        # PLAN
        # ------------------------------------------

        if parsed_output.get("step") == "plan":
            # print(f"🧠: {parsed_output.get('content')}")
            interaction = ask_gemini("Continue to the next step.", previous_id)

            previous_id = interaction.id

            continue


        # ------------------------------------------
        # ACTION
        # ------------------------------------------

        if parsed_output.get("step") == "action":

            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")

            if tool_name not in available_tools:
                # print(f"❌ Unknown tool: {tool_name}")
                interaction = ask_gemini(
                    f"""
                    The requested tool '{tool_name}' does not exist.
                    Available tools are:
                    {list(available_tools.keys())}
                    Choose an available tool.
                    """,
                    previous_id
                )

                previous_id = interaction.id
                continue

            # Get tool function
            tool_function = available_tools[tool_name]["fn"]


            # --------------------------------------
            # CALL TOOL
            # --------------------------------------

            if tool_name == "add":

                # add expects two arguments
                x = tool_input["x"]
                y = tool_input["y"]

                output = tool_function(x, y)

            else:
                output = tool_function(tool_input)


            # --------------------------------------
            # OBSERVATION
            # --------------------------------------

            observation = {"step": "observe", "content": str(output)}

            print(f"👀: {output}")


            # --------------------------------------
            # CONTINUE USING previous_interaction_id
            # --------------------------------------

            interaction = ask_gemini(json.dumps(observation), previous_id)
            previous_id = interaction.id
            continue

        # ------------------------------------------
        # OBSERVE
        # ------------------------------------------
        if parsed_output.get("step") == "observe":
            # print(f"👀: {parsed_output.get('content')}")
            interaction = ask_gemini("Use the observation and continue to the next step.", previous_id)
            previous_id = interaction.id
            continue

        # ------------------------------------------
        # FINAL OUTPUT
        # ------------------------------------------

        if parsed_output.get("step") == "output":
            print(f"🤖: {parsed_output.get('content')}")
            break