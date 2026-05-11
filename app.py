# app.py

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------
# Streamlit Page Config
# ---------------------------------

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.markdown("Ask questions from your PDF document")


# ---------------------------------
# Load Embedding Model
# ---------------------------------

@st.cache_resource
def load_vectorstore():

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

    return vectorstore


vectorstore = load_vectorstore()


# ---------------------------------
# Retriever
# ---------------------------------

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# ---------------------------------
# Load LLM
# ---------------------------------

@st.cache_resource
def load_llm():

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.3
    )

    return llm


llm = load_llm()


# ---------------------------------
# Prompt Template
# ---------------------------------

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


# ---------------------------------
# Chat History
# ---------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------
# User Input
# ---------------------------------

user_query = st.chat_input("Ask a question...")


if user_query:

    # Show user message
    st.chat_message("user").markdown(user_query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    # Retrieve documents
    docs = retriever.invoke(user_query)

    if not docs:

        response = "I could not find any relevant content in the document."

    else:

        # Create context
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # Final Prompt
        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": user_query
            }
        )

        # Generate response
        ai_response = llm.invoke(final_prompt)

        response = ai_response.content

    # Show AI response
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )