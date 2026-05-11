from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


load_dotenv()


print("Loading PDF...")
loader = PyPDFLoader(r"E:\Assingment\Q & A RAG Project\Document Loader\GRU.pdf")
docs = loader.load()
print(f"Loaded {len(docs)} pages.")


print("Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=200     # Overlap to preserve context between chunks
)

chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks.")


print("Loading embedding model (this may take a moment on first run)...")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"   # Free, lightweight, runs on CPU
)


print("Storing chunks in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("\nDatabase created successfully!")
print(f"Total chunks stored: {len(chunks)}")