from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,             # Number of chunks to return
        "fetch_k": 10,      # Candidate pool size before MMR reranking
        "lambda_mult": 0.5  # 0 = max diversity, 1 = max relevance
    }
)

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context, say:
"I could not find the answer in the document."
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("RAG System Ready!")
print("Ask anything about the document.")
print("Press 0 to Exit\n")

while True:

    query = input("You: ").strip()

    if not query:
        continue

    if query == "0":
        print("Exiting...")
        break

    
    docs = retriever.invoke(query)

    if not docs:
        print("\nAI: I could not find any relevant content in the document.\n")
        continue

    
    context = "\n\n".join([doc.page_content for doc in docs])

    
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")