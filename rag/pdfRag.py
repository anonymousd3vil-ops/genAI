import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from pathlib import Path

pdfPath = Path(__file__).parent / "The Ultimate Python Handbook.pdf"

loader = PyPDFLoader(file_path=pdfPath)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

split_docs = text_splitter.split_documents(documents=docs)

embedder = OllamaEmbeddings(
    model="qwen3-embedding:4b",
    dimensions=1024,
)

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     embedding=embedder,
#     collection_name="learing_langchain",
#     url="http://localhost:6333",
# )

# vector_store.add_documents(documents=split_docs)
print("INJECTION DONE")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    collection_name="learing_langchain",
    url="http://localhost:6333",
)

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)

query = input("Query >")

# Retrieve relevant documents from Qdrant
search_result = vector_store.similarity_search(
    query,
    k=4
)

SYSTEM_PRMPT = """
You are a Smart AI Chat Bot.
Analyze the context given below and answer the user's question
based only on the provided context.

If the question is not releted to the given context, Just say "Problem is not availabe in the given PDF or context."

CONTEXT:

"""

# Add retrieved chunks to context
for doc in search_result:
    SYSTEM_PRMPT += "\n\n" + doc.page_content


messages = [
    (
        "system",
        SYSTEM_PRMPT,
    ),
    (
        "human",
        query,
    ),
]

ai_msg = llm.invoke(messages)

print(ai_msg.content)

# print("Releveant Chunks: ", search_result)

# print("DOCS Length: ", len(docs))
# print("Split DOCS Length: ", len(split_docs))