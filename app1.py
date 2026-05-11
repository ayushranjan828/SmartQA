# app.py

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import tempfile
import os

# -----------------------------------
# Streamlit Page Config
# -----------------------------------
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PDF Chatbot using RAG")
st.write("Upload a PDF and ask questions from it.")

# -----------------------------------
# Upload PDF
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

# -----------------------------------
# Initialize Session State
# -----------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------------
# Process PDF
# -----------------------------------
if uploaded_file is not None:

    with st.spinner("Processing PDF..."):

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        # Embedding Model
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Create Chroma Vector DB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        st.session_state.vectorstore = vectorstore

        # Remove temp file
        os.remove(temp_pdf_path)

    st.success("✅ PDF Processed Successfully!")

# -----------------------------------
# Chat Section
# -----------------------------------
if st.session_state.vectorstore is not None:

    query = st.chat_input("Ask a question from the PDF...")

    if query:

        # Store User Message
        st.session_state.chat_history.append(
            {"role": "user", "content": query}
        )

        # Retriever
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        docs = retriever.invoke(query)

        if docs:

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            # LLM
            llm = ChatMistralAI(
                model="mistral-small-latest",
                temperature=0.3
            )

            # Prompt
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

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

            response = llm.invoke(final_prompt)

            answer = response.content

        else:
            answer = "I could not find relevant content in the document."

        # Store AI Message
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

# -----------------------------------
# Display Chat History
# -----------------------------------
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.write(message["content"])