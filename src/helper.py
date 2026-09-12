import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf_file(data):
    loader = PyPDFDirectoryLoader(data)
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs):
    return [doc for doc in docs if doc.page_content and doc.page_content.strip()]

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    clean_chunks = [chunk for chunk in text_chunks if chunk.page_content and chunk.page_content.strip()]
    return clean_chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings