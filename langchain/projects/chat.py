
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import MultiQueryRetriever


DB_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "google_classroom"

EMBEDDING_MODEL = "qwen3-embedding:4b"
CHAT_MODEL = "qwen3:4b"


def main():
    print("Connecting to your classroom knowledge base...")

    llm = ChatOllama(
        model=CHAT_MODEL,
        base_url="http://localhost:11434",
        temperature=0,
    )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434",
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIRECTORY,
        embedding_function=embeddings,
    )

    if vector_store._collection.count() == 0:
        print("Your knowledge base is empty. Run: python ingest.py")
        return

    similarity_retriever = vector_store.as_retriever(
        search_kwargs={"k": 5},
        search_type="similarity"
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=similarity_retriever,
        llm=llm,
        include_original=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful Google Classroom assistant.
Answer questions using the supplied classroom context.
Rules:
- Treat the context as the source of truth.
- Do not invent assignment deadlines, marks, or instructions.
- If the context does not contain the answer, say so clearly.
- Explain technical and academic topics in simple language.
- Distinguish facts from reasonable inferences.
- When useful, mention the assignment or course title.

Classroom context:
{context}"""),
        ("human", "Question: {question}"),
])

    chain = prompt | llm

    print("\nGoogle Classroom RAG Assistant is ready!")
    print("Type 'exit' or 'quit' to close it.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            print("\nSearching classroom data...")

            relevant_docs = retriever.invoke(question)

            if not relevant_docs:
                print("Assistant: I couldn't find relevant classroom information.\n")
                continue

            context_parts = []

            for index, doc in enumerate(relevant_docs, start=1):
                title = doc.metadata.get("title", "Untitled")
                course = doc.metadata.get("course_name", "Unknown course")
                content_type = doc.metadata.get("content_type", "unknown")
                url = doc.metadata.get("alternate_link", "")

                context_parts.append(
                    f"[Source {index}]\n"
                    f"Course: {course}\n"
                    f"Title: {title}\n"
                    f"Type: {content_type}\n"
                    f"URL: {url}\n"
                    f"Content:\n{doc.page_content}"
                )

            context = "\n\n".join(context_parts)

            response = chain.invoke({
                "context": context,
                "question": question,
            })

            print(f"\nAssistant:\n{response.content}")

            # print("\nSources retrieved:")
            # seen = set()

            # for doc in relevant_docs:
            #     title = doc.metadata.get("title", "Untitled")
            #     url = doc.metadata.get("alternate_link", "")
            #     source = (title, url)

            #     if source in seen:
            #         continue

            #     seen.add(source)
            #     print(f"- {title}")

            #     if url:
            #         print(f"  {url}")

            print()

        except Exception as error:
            print(f"\nAn error occurred: {error}\n")
            print(
                "Check that Ollama is running, both models are installed, "
                "and ingestion completed successfully.\n"
            )


if __name__ == "__main__":
    main()
