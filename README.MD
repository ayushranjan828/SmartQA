# PDF Q&A RAG System

A Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask questions about their content. The system uses vector embeddings to retrieve relevant information from the documents and generates answers using a large language model.

## Features

- **PDF Processing**: Load and process PDF documents into searchable chunks
- **Vector Search**: Uses ChromaDB for efficient similarity search
- **Multiple Interfaces**:
  - Web interface via Streamlit (app1.py)
  - Command-line interface (main1.py)
- **Embedding Model**: HuggingFace's all-MiniLM-L6-v2 for text embeddings
- **LLM Integration**: MistralAI's mistral-small-latest model for answer generation
- **Persistent Storage**: ChromaDB persists vector data for reuse

## Project Structure

```
├── app1.py                 # Streamlit web application
├── main1.py                # Command-line interface
├── DB.py                   # Database creation script
├── requirements.txt        # Python dependencies
├── chroma_db/              # ChromaDB vector database storage
├── Document Loader/        # Directory for PDF documents
├── Vector DB/              # Vector database utilities
├── myenv/                  # Python virtual environment
└── README.MD               # This file
```

## Installation

1. **Clone or download the project** to your local machine.

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv myenv
   # On Windows:
   myenv\Scripts\activate
   # On macOS/Linux:
   source myenv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root with your API keys:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

## Usage

### Web Interface (Streamlit)

Run the Streamlit application for an interactive web interface:

```bash
streamlit run app1.py
```

- Upload a PDF document
- Ask questions about the content
- View chat history

### Command-Line Interface

Run the CLI version for terminal-based interaction:

```bash
python main1.py
```

- The system will load the existing vector database
- Enter questions interactively
- Type "0" to exit

### Creating the Vector Database

To create or update the vector database with a specific PDF:

```bash
python Vector DB/DB.py
```

This script loads a PDF from `Document Loader/GRU.pdf`, processes it, and stores the embeddings in ChromaDB.

## How It Works

1. **Document Processing**: PDFs are loaded and split into smaller text chunks with overlap for context preservation.

2. **Embedding Generation**: Each chunk is converted to a vector embedding using HuggingFace's transformer model.

3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search.

4. **Query Processing**:
   - User query is embedded
   - Similar chunks are retrieved using MMR (Maximal Marginal Relevance)
   - Retrieved context is passed to the LLM with the query

5. **Answer Generation**: MistralAI generates answers based only on the retrieved context.

## Dependencies

Key libraries used:
- **LangChain**: Framework for building LLM applications
- **ChromaDB**: Vector database for embeddings
- **Streamlit**: Web application framework
- **HuggingFace Transformers**: Embedding model
- **MistralAI**: Large language model
- **PyPDFLoader**: PDF document loading

See `requirements.txt` for the complete list.

## Configuration

- **Chunk Size**: 1000 characters with 200 character overlap
- **Retriever Settings**: MMR with k=3, fetch_k=10, lambda_mult=0.5
- **LLM Temperature**: 0.3 for consistent answers
- **Embedding Model**: all-MiniLM-L6-v2 (lightweight, CPU-compatible)

## Troubleshooting

- Ensure your MistralAI API key is set in the `.env` file
- Check that the virtual environment is activated
- Verify PDF files are not corrupted
- For large PDFs, processing may take time on first run

## License

This project is for educational purposes. Please ensure compliance with API terms of service and data usage policies.