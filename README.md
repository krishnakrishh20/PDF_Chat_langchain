
PDF-based Query Answering System
This project demonstrates how to create a PDF-based query answering system using Langchain, FAISS, and Streamlit. The system allows users to upload PDFs, processes them into chunks, stores the processed data in a FAISS vector store, and answers queries based on the contents of the PDFs.
Requirements
Before running the project, make sure to install the following dependencies:
pip install langchain langchain-community streamlit faiss-cpu tiktoken python-dotenv
Additionally, you will need to have the following:
•	FAISS for vector search.
•	Langchain for document processing and embedding handling.
•	Streamlit for the user interface.
•	Ollama for embeddings.


Setup
1.	Clone the Repository
git clone https://github.com/krishnakrishh20/PDF_Chat_langchain.git
cd PDF_Chat_langchain
2.	Environment Variables
Create a .env file in the root directory and define the following environment variables:
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
QUERY_MODEL=llama2



Project Structure
•	app.py: Main script for loading PDFs, splitting them into chunks, initializing the vector store, and performing the query search.
•	data/: Directory for storing PDF files.
•	ML_db/: Directory where the vector store is saved.
•	.env: Configuration file for managing environment variables.

Usage

1.	Running the Application
To run the Streamlit application, use the following command:
streamlit run app.py
2.	Upload PDF Files
•	Upon running the application, you'll be prompted to upload PDF files.
•	After uploading, the documents are loaded, processed, and split into chunks.
3.	Querying the Documents
•	Enter your query in the input box, and the system will search for the most relevant content based on the uploaded PDFs.
•	The top search result will be displayed.


Functions
load_and_process_pdfs(pdf_directory)
Loads PDF files from the specified directory and returns the documents.
split_documents(docs, chunk_size=1000, chunk_overlap=100)
Splits the documents into smaller chunks for processing.
initialize_vector_store(embeddings, chunks, vector_db_path=None)
Initializes and loads the FAISS vector store for similarity search.
perform_similarity_search(vector_store, query, k=1)
Performs a similarity search in the vector store for the given query.


Notes
•	This project assumes you have the necessary environment variables set up.
•	Ensure you have the correct FAISS and embedding models available for use.


