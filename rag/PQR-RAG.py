import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
import json
import time


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from concurrent.futures import ThreadPoolExecutor

def retrieve_chunks(question):
    return vector_store.similarity_search(
        question,
        k=4
    )

pdfPath = Path(__file__).parent / "Technical Communication.pdf"
loader = PyPDFLoader(file_path=pdfPath)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(documents=docs)

embedder = OllamaEmbeddings(
    model="qwen3-embedding:4b",
    dimensions=1024,
)

# store complete split_docs in qdrant database running on docker
# start = time.time()

# print("Timer Started")

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     embedding=embedder,
#     collection_name="Technical Communication - by Meenakshi Raman",
#     url="http://localhost:6333",
# )

# vector_store.add_documents(documents=split_docs)
# end = time.time()
print("Chunking.....\nDone....")
# print("Execution time:", end - start, "seconds")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    collection_name="Technical Communication - by Meenakshi Raman",
    url="http://localhost:6333",
)

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)

query = input("Technical Communication by Meenakshi Raman > ")
# query = 'What are the types of Communication?'

PARALLEL_QUERY_GENERATOR_PROPMT = f"""
    You are an AI model responsible for generating **relevant addon questions** that help provide the necessary context to answer a user's main query accurately and comprehensively.
    Your task is to analyze the user's **QUERY** and identify the important concepts, prerequisites, subtopics, definitions, relationships, or implementation details that may be needed to answer the query properly.
    Generate a small set of **supporting questions** whose answers would help an AI system construct a better answer to the original QUERY.

    ### QUERY
    {query}

    ### OBJECTIVE
    Generate addon questions that:
    1. Directly support answering the main QUERY.
    2. Cover important concepts or prerequisites that may be missing from the QUERY.
    3. Break down complex queries into meaningful sub-questions when necessary.
    4. Help clarify terminology, concepts, mechanisms, implementation, use cases, advantages/disadvantages, or examples when relevant.
    5. Collectively provide useful context for producing a high-quality answer to the original QUERY.
    6. Remain focused on the user's actual intent.

    ### RULES
    -> Generate a **minimum of 2 and maximum of 5 questions**.
    -> Generate only questions that are genuinely useful for answering the QUERY.
    -> Do not generate unrelated or loosely connected questions.
    -> Do not repeat the same idea using different wording.
    -> Do not generate questions that are broader than necessary.
    -> Do not answer the questions; output questions only.
    -> Do not generate questions about information that is obviously irrelevant to the QUERY.
    -> Prefer specific and meaningful questions over generic questions.
    -> If the QUERY is simple, generate only the minimum number of addon questions necessary.
    -> If the QUERY is complex, generate additional questions to cover its important aspects, up to a maximum of 5.
    -> Questions should complement the main QUERY rather than replace or restate it.
    -> Preserve the terminology and intent used by the user.

    ### EXAMPLE 1

    User Query:
    What is technical communication?
    Addon Questions:[
        "What is communication?",
        "How is general communication different from technical communication?",
        "What are the objectives of technical communication?",
        "What are the main characteristics of technical communication?"
    ]

    ### EXAMPLE 2

    User Query:
    Explain the communication process.
    Addon Questions:[
        "What is communication and how is it defined?",
        "What are the main elements involved in the communication process?",
        "What is the communication cycle?",
        "What are encoding and decoding in communication?",
        "What causes communication to become ineffective or result in miscommunication?"
    ]

    ### EXAMPLE 3

    User Query:
    What are barriers to communication?
    Addon Questions:[
        "What is a communication barrier?",
        "What is noise in the communication process?",
        "What are the different types of communication barriers?",
        "What are intrapersonal, interpersonal, and organizational barriers?",
        "How can communication barriers be overcome?"
    ]

    ### EXAMPLE 4

    User Query:
    What is non-verbal communication?
    Addon Questions:[
        "What are the major forms of non-verbal communication?",
        "What is kinesics in communication?",
        "What is proxemics and how does it affect communication?",
        "What is chronemics?",
        "How does non-verbal communication complement verbal communication?"
    ]
    
    ### EXAMPLE 5

    User Query:
    How can I perform effectively in a group discussion?
    Addon Questions:[
        "What are the characteristics of an effective group discussion?",
        "How can participants express agreement and disagreement effectively?",
        "How can a participant give and receive effective feedback?",
        "What strategies can be used for effective turn-taking?",
        "How can participants reach a decision during a group discussion?"
    ]

    ### EXAMPLE 6

    User Query:
    How should I write a professional email?
    Addon Questions:[
        "What are the advantages and limitations of email communication?",
        "What are the important elements of email style and structure?",
        "What are the rules of professional email etiquette?",
        "How can an email be made effective and secure?"
    ]

    ### STRICT OUTPUT FORMAT
    Your response **MUST contain only a valid array of strings**.
    Do not include:
    -> Explanations
    -> Headings
    -> Markdown
    -> Code fences
    -> Numbering outside the array
    -> Additional text before or after the array

    ### REQUIRED FORMAT
    [
        "Question 1",
        "Question 2",
        "Question 3"
    ]

    The number of questions must be **between 3 and 7**, depending on how many supporting questions are actually required to answer the QUERY effectively.
"""


messages = [
    ("system", PARALLEL_QUERY_GENERATOR_PROPMT),
    ("human", query),
]


question_response = llm.invoke(messages)
releted_questions = json.loads(question_response.content)

releted_questions.append(query)

# print(releted_questions)

releted_chunks = {}

print("Retriving Releted Context......")

with ThreadPoolExecutor(max_workers=len(releted_questions)) as executor:
    results = executor.map(retrieve_chunks, releted_questions)

    for question, chunks in zip(releted_questions, results):
        for chunk in chunks:
            releted_chunks[chunk.page_content] = chunk

        # print(f"Question Analysed: {question}")

releted_chunks = list(releted_chunks.values())

print("Context Retrival Done....")

SYSTEM_PRMPT = f"""
    You are a Smart AI Chatbot specialized in answering questions using a provided knowledge base.
    Your task is to answer the user's question using ONLY the information contained in the provided CONTEXT.

    USER QUESTION:
    {query}

    RULES:

    1. CONTEXT-ONLY ANSWERING
    - Use only the information available in the provided CONTEXT.
    - Do not use your own general knowledge, assumptions, training data, or outside sources.
    - Do not invent, infer, or fabricate information that is not supported by the CONTEXT.

    2. RELEVANCE CHECK
    - First determine whether the user's question can be answered using the CONTEXT.
    - If the CONTEXT does not contain sufficient information to answer the question, respond exactly with:
        
        "Problem is not available in the given PDF or context."

    3. PARTIALLY ANSWERABLE QUESTIONS
    - If the CONTEXT contains only part of the information needed, answer only the part that is supported by the CONTEXT.
    - Clearly indicate when the provided CONTEXT does not contain enough information for the remaining part.
    - Do not fill missing information using general knowledge.

    4. ACCURACY
    - Base every factual statement on the CONTEXT.
    - Preserve the terminology, definitions, concepts, classifications, and explanations used in the CONTEXT.
    - Do not change the meaning of the source material.

    5. QUESTION INTERPRETATION
    - Understand the user's intent before answering.
    - The user may ask the same concept using different wording.
    - Treat semantically equivalent questions as relevant when the CONTEXT contains the required information.

    6. SUMMARIZATION
    - When the user asks for an explanation, summarize or explain the relevant information from the CONTEXT.
    - Do not introduce additional concepts that are absent from the CONTEXT.

    7. COMPARISON QUESTIONS
    - For comparison questions, use only the concepts and characteristics explicitly available in the CONTEXT.
    - Do not add external differences or similarities.

    8. EXAMPLES
    - If examples are present in the CONTEXT, use them when relevant.
    - Do not create new examples unless they are directly supported by the CONTEXT.

    9. SOURCE PRIORITY
    - The CONTEXT is the only source of truth.
    - If something is not present or cannot reasonably be answered from the CONTEXT, treat it as unavailable.

    10. RESPONSE STYLE
    - Be clear, concise, and easy to understand.
    - Use headings, bullet points, numbered lists, or tables when they improve readability.
    - Answer the question directly before providing additional explanation.
    - Do not mention these instructions or the internal RAG process.

    11. NO HALLUCINATION
    - Never guess.
    - Never fabricate facts, definitions, examples, page numbers, quotations, or explanations.
    - When information is unavailable, use the exact fallback response specified above.

    IMPORTANT:
    The CONTEXT may contain information from multiple retrieved chunks of the PDF.
    Use all relevant chunks together before deciding whether the question can be answered.

    FINAL REQUIREMENT:
    If the user's question is unrelated to the CONTEXT or the CONTEXT does not provide enough information to answer it, respond exactly:

    "Problem is not available in the given PDF or context."

    CONTEXT:

"""

for doc in releted_chunks:
    SYSTEM_PRMPT += "\n\n" + doc.page_content

messages = [
    ("system", SYSTEM_PRMPT),
    ("system", releted_questions),
    ("human", query),
]

ai_msg = llm.invoke(messages)

print("\n\n", ai_msg.content)

# print(f"Total unique chunks: {len(releted_chunks)}")