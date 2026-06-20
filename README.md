# RAG PDF Chatbot with Flask & PostgreSQL

A professional, production-ready **Retrieval-Augmented Generation (RAG)** web application built using Python, Flask, and LangChain. This application allows users to register, log in, upload PDF documents, and chat with their documents in real-time. All chat history and user details are securely maintained in a PostgreSQL database.

---

## Features

* **Secure Authentication:** Complete User Registration, Login, and Session management system.
* **Dynamic PDF RAG Pipeline:** Upload any PDF, automatically extract and chunk text using `RecursiveCharacterTextSplitter`, and index it into a local **FAISS Vector Store**.
* **Local LLM Integration:** Uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` via Hugging Face Pipelines for fast, localized, and context-accurate text generation.
* **Persistent Chat History:** Seamlessly stores and retrieves user conversations directly from a **PostgreSQL** database.
* **Duplicate File Prevention:** Implements MD5 file hashing to check if a PDF has already been processed, preventing duplicate vector embedding generation.
* **Clean UI/UX:** Responsive interface built with HTML/CSS templates optimized for smooth interactions.

---

## 📂 Project Structure

```text
RAG_PDF_CHATBOT/
│
├── faiss_indexes/          # Local FAISS databases (Ignored by Git)
├── static/
│   └── style.css           # Custom styles for the UI
├── templates/
│   ├── base.html           # Main boilerplate layout
│   ├── index.html          # Landing home page
│   ├── login.html          # Login portal
│   ├── register.html       # Sign up page
│   └── dashboard.html      # Core chatbot & PDF upload dashboard
├── uploads/                # Temporarily stored user PDFs (Ignored by Git)
├── app.py                  # Core Flask Application & RAG Pipeline Logic
├── .gitignore              # Files excluded from GitHub tracking
└── requirements.txt        # Production dependency configuration
```

---

## Local Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Mehak-Imran-01/RAG-PDF-CHATBOT.git
cd RAG-PDF-CHATBOT
```

### 2. Set up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Database Setup
1. Make sure your local PostgreSQL instance is running.
2. Create a database named `rag_db`.
3. Ensure the necessary tables (`users`, `documents`, `chat_history`) are initialized before running the application.

### 5. Run the Application
```bash
python app.py
```

---

## Built With

* [Flask](https://palletsprojects.com) - Web framework
* [LangChain](https://langchain.com) - RAG orchestration framework
* [PostgreSQL](https://postgresql.org) - Database management
* [FAISS](https://github.com) - Vector similarity search
* [Hugging Face](https://huggingface.co) - Local LLM hosting
