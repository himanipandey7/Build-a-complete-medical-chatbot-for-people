from __future__ import annotations

import os
import sys

# Ensure Python can locate the 'src' package from the root directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, render_template, request
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import system prompt from src, or fall back to default
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

# Verify API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("WARNING: GROQ_API_KEY environment variable is not set!")

# 1. Initialize Embeddings & Vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chroma_dir = os.path.abspath("./chroma_db")
if not os.path.exists(chroma_dir):
    os.makedirs(chroma_dir, exist_ok=True)

vectorstore = Chroma(
    persist_directory=chroma_dir,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 2. Initialize Groq Model
chat_model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# 3. Conversational Memory Pipeline
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    chat_model, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []

# --- ROUTES ---

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg") or request.args.get("msg")
    if not msg:
        return "Please enter a valid message."

    print(f"User Input: {msg}")

    try:
        response = rag_chain.invoke({
            "input": msg,
            "chat_history": chat_history
        })
        answer = response["answer"]
    except Exception as e:
        print(f"Error generating response: {e}")
        answer = "Sorry, an error occurred while processing your request."

    chat_history.append(("human", msg))
    chat_history.append(("ai", answer))

    return str(answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)