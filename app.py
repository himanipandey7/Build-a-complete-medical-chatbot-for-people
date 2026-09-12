import base64
import os
import torch
from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Memory Optimization for Render Free Tier (512MB RAM)
torch.set_num_threads(1)

app = Flask(__name__)

# 2. API Key Management
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    encoded_key = "" 
    if encoded_key:
        groq_api_key = base64.b64decode(encoded_key).decode("utf-8")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
else:
    print("WARNING: GROQ_API_KEY is missing! RAG pipeline will fail.")

# 3. Initialize Embeddings & Vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chroma_dir = os.path.abspath("./chroma_db")
vectorstore = Chroma(
    persist_directory=chroma_dir,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 4. LLM & Retrieval Chain Setup
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    max_tokens=500
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. Flask Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg") or (request.get_json() or {}).get("msg")
    if not msg:
        return jsonify({"response": "Please enter a valid message."}), 400
    
    try:
        response = rag_chain.invoke({"input": msg})
        return jsonify({"response": response["answer"]})
    except Exception as e:
        print(f"Error during chain execution: {e}")
        return jsonify({"response": "Sorry, an error occurred processing your request."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)