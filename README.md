# Build-a-complete-medical-chatbot
Build a Complete Medical Chatbot with Local LLM (Ollama), LangChain, Chroma DB & Flask
How to Run Locally?
STEPS:
#STEP 01: Open your project directory in terminal

PowerShell
cd Build-a-complete-medical-chatbot-for-people

#STEP 02: Activate your virtual environment

PowerShell
.\\.venv311\\Scripts\\Activate.ps1
#STEP 03: Install dependencies

PowerShell
pip install -r requirements.txt
#STEP 04: Ensure local Ollama model is ready
Make sure Ollama is installed on your computer and the model is downloaded:

PowerShell
ollama run qwen2.5:3b
#STEP 05: Store embeddings locally in Chroma DB
Run the following command to process Medical_book.pdf and create local vector embeddings in ./chroma_db:

PowerShell
python store_index.py
#STEP 06: Launch the web application

PowerShell
python app.py
#STEP 07: Open Localhost
Open your browser and navigate to:
http://localhost:5000 or http://127.0.0.1:5000

Repository Structure
Plaintext
Build-a-complete-medical-chatbot-for-people/
├── Data/
│   └── Medical_book.pdf         # Reference medical document
├── chroma_db/                    # Local Chroma vector database directory
├── research/
│   └── Trials.ipynb             # Jupyter Notebook for pipeline experiments
├── src/
│   ├── __init__.py
│   ├── helper.py                # PDF loading, splitting, and embedding logic
│   └── prompt.py                # System prompt templates
├── static/
│   └── style.css                # Web interface styling
├── templates/
│   └── chat.html                # Chatbot HTML template
├── .gitignore                   # Ignored files (.env, .venv, chroma_db)
├── app.py                       # Main Flask web application
├── store_index.py               # Script to split PDF and build Chroma vector index
└── requirements.txt             # Python dependencies
Techstack Used (Local Architecture)
Programming Language: Python 3.11

Framework: Flask

Orchestration: LangChain

Local LLM: Ollama (qwen2.5:3b)

Vector Store: Chroma DB (Local persistence in ./chroma_db)

Embeddings: Hugging Face (sentence-transformers/all-MiniLM-L6-v2)

Version Control: Git & GitHub