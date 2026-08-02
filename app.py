import sys
import os

# Ensure Python can locate the 'src' package from the root directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, render_template, request
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Import system prompt from src, or fall back to default if src/prompt.py is missing
try:
    from src.prompt import system_prompt
except ModuleNotFoundError:
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )

app = Flask(__name__)

# 1. Local Embeddings (No API Key needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Local Vector Store
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 3. Local Ollama Model
chat_model = ChatOllama(model="qwen2.5:3b")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Input: {msg}")
    
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    
    return str(response["answer"])


if __name__ == '__main__':
    # Localhost server on port 5000
    app.run(host="127.0.0.1", port=5000, debug=True)