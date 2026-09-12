import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
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

retriver = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    collection_name="learing_langchain",
    url="http://localhost:6333",
)

search_result = retriver.similarity_search(
    query="What is loops in python?"
)

print("Releveant Chunks: ", search_result)

# print("DOCS Length: ", len(docs))
# print("Split DOCS Length: ", len(split_docs))