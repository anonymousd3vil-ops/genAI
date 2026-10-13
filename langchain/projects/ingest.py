import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from googleClassrommLoader import get_classroom_documents


# Configuration
DB_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "google_classroom"

EMBEDDING_MODEL = "qwen3-embedding:4b"


def main():
    print("Fetching Google Classroom documents...")

    documents = get_classroom_documents()

    # Remove empty documents and normalize metadata for Chroma.
    documents = [
        doc
        for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]

    if not documents:
        print("No classroom documents were found.")
        return

    print(f"Loaded {len(documents)} documents.")

    # Split long documents into smaller pieces.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434",
    )

    # Rebuild the collection so rerunning ingestion doesn't
    # leave duplicate or outdated classroom chunks.
    client = chromadb.PersistentClient(path=DB_DIRECTORY)

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Removed the previous collection.")
    except Exception:
        pass

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Keep batches manageable for local hardware.
    batch_size = 32

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vector_store.add_documents(documents=batch)

        print(
            f"Indexed {min(start + len(batch), len(chunks))}"
            f"/{len(chunks)} chunks."
        )

    print("\nIngestion completed successfully.")
    print(f"Database location: {DB_DIRECTORY}")


if __name__ == "__main__":
    main()
