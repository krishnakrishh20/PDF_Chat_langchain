import os
import faiss
import tiktoken
import logging
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables (to manage configurations like base_url and model)
load_dotenv()

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PDF_DIR = "data"  # Directory where PDFs are stored
VECTOR_DB_PATH = r'C:\Users\krish\OneDrive\Documents\codes\Langchain-ollama\ML_db'
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
QUERY_MODEL = os.getenv("QUERY_MODEL", "llama2")

# Function to load PDFs from the directory and split them into chunks
def load_and_process_pdfs(pdf_directory):
    logger.info(f"Loading PDF files from directory: {pdf_directory}")
    pdf_files = [os.path.join(root, file) 
                 for root, dirs, files in os.walk(pdf_directory) 
                 for file in files if file.endswith(".pdf")]

    docs = []
    for pdf in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf)
            temp_docs = loader.load()
            docs.extend(temp_docs)
            logger.info(f"Loaded PDF: {pdf}")
        except Exception as e:
            logger.error(f"Error loading PDF {pdf}: {e}")
    
    logger.info(f"Loaded {len(docs)} documents.")
    return docs

# Function to split documents into chunks
def split_documents(docs, chunk_size=1000, chunk_overlap=100):
    logger.info("Splitting documents into chunks.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

# Function to initialize and load vector store
def initialize_vector_store(embeddings, chunks, vector_db_path=None):
    logger.info("Initializing vector store.")
    try:
        if vector_db_path and os.path.exists(vector_db_path):
            logger.info(f"Loading vector store from: {vector_db_path}")
            vector_store = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        else:
            logger.info("Creating a new vector store.")
            # Create a FAISS index and store
            index = faiss.IndexFlatL2(len(embeddings.embed_query(chunks[0].page_content)))  # Create FAISS index
            vector_store = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore({}), index_to_docstore_id={})
            vector_store.add_documents(chunks)  # Add documents to the vector store
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise

# Function to perform a similarity search
def perform_similarity_search(vector_store, query, k=1):
    logger.info(f"Performing similarity search for query: {query}")
    try:
        docs = vector_store.search(query=query, k=k, search_type='similarity')
        return docs
    except Exception as e:
        logger.error(f"Error performing similarity search: {e}")
        return []

# Streamlit UI Elements
st.title("PDF-based Query Answering System")
st.subheader("Upload PDFs for Document Processing")

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Load and process PDFs
    docs = []
    for uploaded_file in uploaded_files:
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyMuPDFLoader(f"temp_{uploaded_file.name}")
        docs.extend(loader.load())

    # Split documents into chunks
    chunks = split_documents(docs)

    # Initialize embeddings and vector store
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    vector_store = initialize_vector_store(embeddings, chunks, VECTOR_DB_PATH)

    st.success(f"Successfully loaded {len(docs)} documents and created vector store.")

    # Query input section
    query = st.text_input("Enter your query:")

    if query:
        # Perform similarity search
        search_results = perform_similarity_search(vector_store, query, k=1)

        if search_results:
            st.write(f"Top search result: {search_results[0].page_content[:500]}...")  # Display first 500 chars of result
        else:
            st.warning("No results found for the query.")
else:
    st.info("Upload PDF files to get started.")
