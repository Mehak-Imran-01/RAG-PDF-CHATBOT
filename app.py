from flask import Flask, render_template, request, redirect, session
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
import psycopg2
import hashlib
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "Mehak123"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("faiss_indexes", exist_ok=True)

# =========================
# DATABASE CONNECTION
# =========================
conn = psycopg2.connect(
    dbname="rag_db",
    user="postgres",
    password="1234",
    host="172.19.32.182",
    port="5432"
)
cur = conn.cursor()

# =========================
# GLOBALS
# =========================
retriever = None

# =========================
# MODELS
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1,
    max_new_tokens=200,
    temperature=0.3,
    repetition_penalty=1.2,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant.

Use ONLY the given context.

Context:
{context}

Question:
{question}

Answer:
"""
)

# =========================
# HASH FUNCTION
# =========================
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# =========================
# PROCESS PDF
# =========================
def process_pdf(file_path):
    global retriever

    file_hash = get_file_hash(file_path)

    cur.execute("SELECT faiss_path FROM documents WHERE file_hash=%s", (file_hash,))
    result = cur.fetchone()

    if result:
        faiss_path = result[0]
        vectorstore = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(docs)

        faiss_path = f"faiss_indexes/{file_hash}"

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(faiss_path)

        cur.execute("""
            INSERT INTO documents
            (file_name, file_hash, file_path, faiss_path, status)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            os.path.basename(file_path),
            file_hash,
            file_path,
            faiss_path,
            "ready"
        ))
        conn.commit()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# =========================
# DASHBOARD
# =========================
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    print("DASHBOARD SESSION:", dict(session))

    # Check if user is logged in
    if "user_id" not in session:
        return redirect("/login")

    print("SESSION DEBUG:", dict(session))

    # =========================
    # HANDLE FORM SUBMISSION
    # =========================
    if request.method == "POST":

        # ---------- PDF Upload ----------
        if "pdf_file" in request.files:
            file = request.files["pdf_file"]

            if file and file.filename:
                path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(path)
                process_pdf(path)

        # ---------- User Query ----------
    # ---------- User Query ----------
    query = request.form.get("query")

    # If user asks a question without uploading a PDF
    if query and not retriever:
        answer = "Please upload a PDF first."

        # Save this message to database (optional but recommended)
        cur.execute("""
            INSERT INTO chat_history (user_id, question, answer)
            VALUES (%s, %s, %s)
        """, (
            session["user_id"],
            query,
            answer
        ))
        conn.commit()

    # If PDF is loaded, process normally
    elif query and retriever:
        # Retrieve relevant chunks
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])

        # Build prompt
        final_prompt = prompt.format(
            context=context,
            question=query
        )

        # Generate answer
        answer = llm.invoke(final_prompt).strip()

        # Save chat history to PostgreSQL
        cur.execute("""
            INSERT INTO chat_history (user_id, question, answer)
            VALUES (%s, %s, %s)
        """, (
            session["user_id"],
            query,
            answer
        ))
        conn.commit()

    # =========================
    # LOAD CHAT HISTORY FROM DATABASE
    # =========================
    cur.execute("""
        SELECT question, answer
        FROM chat_history
        WHERE user_id = %s
        ORDER BY created_at ASC
    """, (session["user_id"],))

    rows = cur.fetchall()

    # Convert database rows into the format expected by dashboard.html
    chat_history = []
    for row in rows:
        chat_history.append({
            "q": row[0],
            "a": row[1]
        })

    # =========================
    # RENDER DASHBOARD
    # =========================
    return render_template(
        "dashboard.html",
        chat_history=chat_history
    )

# =========================
# Clear Chat
# =========================
@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    if "user_id" not in session:
        return redirect("/login")

    cur.execute("""
        DELETE FROM chat_history
        WHERE user_id = %s
    """, (session["user_id"],))
    conn.commit()

    return redirect("/dashboard")
# =========================
# REGISTER
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        try:
            cur.execute("""
                INSERT INTO users (username, email, password)
                VALUES (%s, %s, %s)
            """, (username, email, password))
            conn.commit()

        except Exception as e:
            conn.rollback()
            print("Register Error:", e)

        return redirect("/login")

    return render_template("register.html")

# =========================
# LOGIN
# =========================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        cur.execute(
            "SELECT user_id, username FROM users WHERE email=%s AND password=%s",
            (email, password)
        )
        user = cur.fetchone()

        print("LOGIN USER:", user)

        if not user:
            return "Invalid credentials"

        global retriever
        retriever = None

        session.clear()
        session["user_id"] = user[0]
        session["username"] = str(user[1])

        session.permanent = True
        session.modified = True

        return redirect("/dashboard")

    return render_template("login.html")
# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():
    global retriever
    retriever = None

    session.clear()
    return redirect("/")

# =========================
# HOME
# =========================
@app.route("/")
def home():
    if "user_id" in session:
        return redirect("/dashboard")
    return render_template("index.html")
# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)