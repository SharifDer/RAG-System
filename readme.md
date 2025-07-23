# 🤖 Personal RAG System: Local AI Assistant with ChromaDB, LLM & Telegram Bot

This is a fully local Retrieval-Augmented Generation (RAG) system that combines:

* 🧠 **LLM for generation** (e.g. Phi-3, LLaMA, Mistral, etc., running locally via `llama.cpp`)
* 📚 **Embedding model** for semantic document retrieval
* 💾 **ChromaDB** for vector storage
* 🌐 **Modern Web Frontend** for querying
* 💬 **Telegram Bot** interface

> 🔐 Entirely private and offline — no OpenAI/Anthropic APIs. Your data stays with you.

---

## 📽️ Demo Video (Frontend Interaction)

🚧 *Placeholder: Demo video will be embedded here.*

```markdown
[![Watch the demo](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)
```

---

## 💬 Demo Video (Telegram Bot)

🚧 *Placeholder: Screen recording of the Telegram bot in action.*

---

## 🧩 Features

* 🔍 Smart document retrieval via embeddings
* ✨ Local LLM-based answer generation
* 📥 Easy file ingestion (PDF, text, etc.)
* 🌐 Web-based UI for real-time Q\&A
* 🤖 Telegram bot interface
* ⚡ Fast, streaming responses via FastAPI

---

## 🏗️ Architecture

```text
         +-----------------+
         | Telegram Bot UI |
         +--------+--------+
                  |
+---------+       v       +--------------+
| Web UI  | ---> FastAPI <-> Retrieval   |
+---------+       |       +--------------+
                  v
        +--------------------+
        | ChromaDB (VectorDB)|
        +--------------------+
                  |
                  v
        +--------------------+
        | Embedding Model    |
        +--------------------+
                  |
                  v
        +--------------------+
        | Local LLM (GGUF)   |
        +--------------------+
```

---

## 🚀 Setup Instructions (Local)

> ✅ Tested on **Linux/macOS**. For Windows, **WSL is required** due to `llama.cpp` dependencies like CMake & g++.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/local-rag-system.git
cd local-rag-system
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies (\~2GB)

```bash
pip install -r requirements.txt
```

> ⚠️ This installs models like `sentence-transformers` and dependencies for llama.cpp (if needed).

### 4. Ingest Your Documents

```bash
python app/ingest.py  # or your custom ingestion script
```

### 5. Run the Backend API

```bash
python run.py  # Or use uvicorn: uvicorn app.main:app --reload
```

### 6. Start the Telegram Bot (optional)

```bash
python telegram_bot.py
```

---

## 🐳 Docker Support (For Users)

> 💡 If you're running this on a different OS or want zero setup: Docker is the easiest way.

### 🛠️ Prerequisites

Make sure you have [Docker installed](https://www.docker.com/products/docker-desktop/).

### 📦 Build Docker Image

```bash
docker build -t local-rag .
```

### ▶️ Run Docker Container

```bash
docker run -p 8000:8000 local-rag
```

> 📝 The container runs the FastAPI backend. You can expose additional ports (e.g., for frontend or Telegram bot) as needed.

---

## 📁 Folder Structure

```bash
local-rag-system/
├── app/
│   ├── retrieval.py            # Embedding search
│   ├── generation.py           # LLM generation logic
│   ├── ingest.py               # Ingest docs into ChromaDB
│   └── main.py                 # FastAPI routes
├── frontend/                   # Web frontend code (React/Vite)
├── telegram_bot.py             # Telegram bot entrypoint
├── run.py                      # Main runner
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## 🔎 FAQs

### Why not run it using Docker myself?

This repo is **developed using a local Python environment (in WSL)** for simplicity. Docker is provided for:

* Other users/recruiters
* Cross-platform compatibility

No need for the developer to use Docker unless desired.

### What about model downloads?

Model download automation is out-of-scope (due to licenses & size). You can:

* Place `.gguf` model in a `models/` folder
* Adjust the code to point to your local path

---

## 📫 Contact

Built with ❤️ by \Sharif.

If you're a recruiter or engineer interested in building real-world AI tools, feel free to reach out:

* 📧 Email: [you@example.com](mailto:you@example.com)
* 💼 LinkedIn: [linkedin.com/in/yourname](https://linkedin.com/in/yourname)

---

## ⭐️ Bonus (If You Want to Extend)

* 📊 Add tracing with Langfuse or Prometheus + Grafana
* 🔐 Secure the API using OAuth2 or JWT
* 🌍 Deploy with Docker Compose, Traefik, or on Hugging Face Spaces
* 📥 Upload files via frontend drag-n-drop
* 🧠 Switch to better embeddings (e.g. `bge-m3` or `E5`)
