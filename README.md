# Avika Chat API

A FastAPI-based chatbot API that provides mental wellness support by recommending mental health resources based on user input.

---

## ⚙️ Setup and Configuration

### 1. Environment Variables

Before running the application, you need to set up the following environment variables. You can create a `.env` file in the root directory:

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (with defaults)
AVIKA_TITLES_PATH=./Avika_Titles.docx
CHROMA_DB_PATH=./chroma_storage
AVIKA_DOCS_PATH=./Avika_Docs/
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Populate the Vector Database

```bash
python scripts/populate_db.py
```

## 🚀 How to Run

### Option 1: Local Development (Recommended)

**Simple and fast for development/testing:**

```bash
# 1. Set environment variables
export GEMINI_API_KEY="your_api_key"

# 2. Install and run
pip install -r requirements.txt
python scripts/populate_db.py
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Docker (For Production/Isolation)

**Use Docker if you need:**
- Consistent deployment across different environments
- Dependency isolation
- Easy cloud deployment

```bash
# Build and run
docker build -t avika-chat-api .
docker run -p 8000:8000 \
  -e GEMINI_API_KEY="your_api_key" \
  -v $(pwd)/chroma_storage:/app/chroma_storage \
  -v $(pwd)/Avika_Titles.docx:/app/Avika_Titles.docx \
  avika-chat-api
```

**Note:** Docker requires volume mounting for data persistence.

---

## 📡 API Endpoints

### 🔍 1. Health Check

`GET /health`

Returns the API status.

Response:

```json
{
  "status": "healthy"
}
```

### 2. Chat with Avika

`POST /chat`

Send a message to the chatbot and get a relevant resource recommendation.

Request Body:

```json
{
  "message": "string",
  "session_id": "string (optional)"
}
```

Response:

```json
{
  "response": "string",
  "session_id": "string"
}
```

---

## Example Usage

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "I feel anxious lately"}'
```

### Using Python requests

```python
import requests

# Health check
print(requests.get("http://localhost:8000/health").json())

# Chat
res = requests.post("http://localhost:8000/chat", json={"message": "I feel low"})
print(res.json())
```

---

## 🛠️ Features

- Session-aware conversations
- RAG-based mental health recommendations
- Safety guardrails for crisis situations
- Only recommends from official Avika app resources
- Interactive docs at: http://localhost:8000/docs

---

## Tech Stack

- FastAPI + Uvicorn
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- Google Gemini API (LLM)
- Python DOCX processing

---

## File Structure

```
.
├── api_server.py         # FastAPI application server
├── avika_chat.py        # Core chatbot logic with AvikaChat class
├── scripts/
│   └── populate_db.py   # Script to populate the vector database
├── Avika_Docs/          # Folder for DOCX source documents
├── Avika_Titles.docx    # Document containing titles and categories
├── chroma_storage/      # (Generated) ChromaDB data directory
├── requirements.txt     # Python dependencies
├── Dockerfile          # Optional: Container setup
├── .dockerignore       # Docker build exclusions
├── .gitignore          # Git version control exclusions
└── README.md           # This documentation

# Development Files (excluded from production)
├── Mistral_Avika.ipynb    # Jupyter notebook for experimentation
```

---

## Development vs Production

### For Local Development
```bash
# Quick start - no Docker needed
pip install -r requirements.txt
python scripts/populate_db.py
uvicorn api_server:app --reload
```

### For Production Deployment
- **Simple servers**: Use the local approach with a process manager (PM2, systemd)
- **Cloud platforms**: Consider Docker for consistent deployments
- **Complex environments**: Docker provides better isolation

---

## Crisis Safety

The chatbot includes built-in responses to guide users to mental health helplines in case of crisis situations.
