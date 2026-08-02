 import os
import sys

# Ensure Python can find modules inside 'src'
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from langchain_chroma import Chroma

# 1. Load PDFs from data directory
print("Loading PDF files...")
extracted_data = load_pdf_file(data='Data/')

# 2. Filter and split into text chunks
print("Splitting text into chunks...")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# 3. Load Hugging Face local embeddings (384 dimensions)
print("Downloading/Loading Hugging Face embeddings...")
embeddings = download_hugging_face_embeddings()

# 4. Create and persist local Chroma vector database
print("Creating local Chroma vector store in ./chroma_db ...")
vectorstore = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✅ Success! Local Chroma database created in ./chroma_db")