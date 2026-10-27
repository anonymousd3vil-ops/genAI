from deepagents import (
    create_deep_agent,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import FilesystemBackend
from deepagents_custom_toos import fetch_pokemon_details, internet_search, research_instructions, langfuse_handler
from langchain_core.messages import HumanMessage

register_harness_profile(
    "ollama:qwen3:4b",
    HarnessProfile(
        base_system_prompt="""
            You are an expert research orchestrator.

            Use fetch_pokemon_details for Pokémon facts available
            through PokéAPI. Use internet_search for broader research.

            Use filesystem tools to manage research notes, read relevant
            project files, and save reports when requested.
        """,
        general_purpose_subagent=GeneralPurposeSubagentProfile(
            system_prompt="""
                You are a research subagent.
                Investigate assigned tasks and cite web sources.
                Use filesystem tools when needed.
                Return concise, evidence-based findings.
            """,
        ),
        system_prompt_suffix="""
            Verify factual claims against tool results.
            Never invent facts or sources.
        """,
    ),
)

filesystem = FilesystemMiddleware(\
    backend=FilesystemBackend(
        root_dir="./agent_workspace",
        virtual_mode=True,
    ),
    tools=[
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
    ]
)

agent = create_deep_agent(
    model="ollama:qwen3:4b",
    tools=[
        internet_search,
        fetch_pokemon_details,
    ],
    system_prompt=research_instructions,
    middleware=[filesystem],
    memory=["./AGENTS.md"],
    backend=FilesystemBackend(
        root_dir='./agent_workspace',
        virtual_mode=True,
    ),
)

def main():
    query = input(">> ").strip()

    if not query:
        print("Please enter a query.")
        return

    
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"callbacks": [langfuse_handler]}, #for langfuse monitoring
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
