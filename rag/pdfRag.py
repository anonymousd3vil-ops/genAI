import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from pathlib import Path

pdfPath = Path(__file__).parent / "The Ultimate Python Handbook.pdf"

loader = PyPDFLoader(file_path=pdfPath) #load the respective file

docs = loader.load()

#make splitter client this will spilit the complete docs in given chunk size with the overlapping
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

#split docs are stored in split_docs
split_docs = text_splitter.split_documents(documents=docs)

#made an embedded client
embedder = OllamaEmbeddings(
    model="qwen3-embedding:4b",
    dimensions=1024,
)

#store complete split_docs in qdrant database running on docker
# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     embedding=embedder,
#     collection_name="learing_langchain",
#     url="http://localhost:6333",
# )

#add the vectors in database
# vector_store.add_documents(documents=split_docs)
print("INJECTION DONE")

#giveing datatabse bonter from where we have to take the data
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    collection_name="learing_langchain",
    url="http://localhost:6333",
)

#created a chat interface with using OLlama with qwen3 model
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)

#ask for user quesy
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

# Add retrieved chunks to context (SYSTEM_PRIOMPT)
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