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

## Recent Enhancements (Post-Initial Commit)

This section details significant improvements made after the initial codebase setup, focusing on safety, conversational quality, and ethical alignment based on recent feedback.

### 1. Chatbot Safety and Ethics

*   **Disclaimer Implementation**: 
    *   The chatbot now introduces itself with a clear disclaimer: `"Hello! I'm Avika, a supportive chatbot here to listen and help you find helpful resources. Please know that I'm not a licensed therapist or a replacement for professional care. If you're in crisis or experiencing severe distress, please reach out to a mental health professional or a crisis hotline. What's on your mind today?"`
    *   This initial greeting is added to the chat history, making the LLM contextually aware of its role and limitations.
    *   System messages within prompts also reiterate that Avika is a chatbot, not a human or a substitute for professional help.
*   **Contextual Safety Monitoring**:
    *   The `check_safety_concerns` method in `avika_chat.py` has been enhanced. It now analyzes the last 3 user messages along with the current input, providing a broader context for detecting potential self-harm or harm-to-others indicators.
    *   This moves beyond simple keyword matching on the latest message to better understand nuanced or gradually revealed concerning statements.
*   **Simulated Escalation Logging**:
    *   When a potential safety concern is flagged, a message is now printed to the console: `SAFETY ALERT: Potential crisis detected. User history hint: [last 100 chars of context]. Flag for human review.`
    *   This simulates a basic human-in-the-loop mechanism, indicating where a real system would trigger alerts for human moderation.
*   **Refined Crisis Response Language**:
    *   The `_get_crisis_response` method has been rewritten to be more empathetic, shorter, and context-aware.
    *   Instead of a long, static message, it now offers to share resources in a more conversational tone, for example: `"It sounds like you're going through something incredibly painful, and I want you to know you're not alone. For the kind of support you need right now, it's best to talk with a professional. Can I share some resources that can help immediately?"` (It then provides key helplines).

### 2. Conversational Quality

*   **More Natural Empathy Prompts**:
    *   The rules in `_construct_empathy_prompt` were relaxed to make the LLM's responses less formulaic.
    *   The bot is now encouraged to reflect on the user's feelings, offer supportive statements, or ask a gentle follow-up question *if it feels natural*, rather than always being forced to end with a question.
*   **Smoother Transition to Recommendations**:
    *   The `chat` method now generates a `reflection_summary` (e.g., `"From what you've shared, it sounds like you're dealing with: [detected emotional theme]. I'd like to offer something that might be helpful."`) before suggesting resources.
    *   This aims to make the shift from empathetic listening to resource recommendation feel less abrupt and more congruent with the conversation's flow.
*   **Reduced Empathy-Only Turns**:
    *   The initial phase of pure empathy (before any recommendations) was shortened from 3 turns to 2 turns (`self.turn_number < 2`). This allows the conversation to progress to actionable suggestions more quickly if appropriate, while still ensuring initial emotional validation.

### 3. Resource Recommendation Handling

*   **Improved "No Titles Found" Flow**:
    *   The `_construct_recommendation_prompt` now includes specific "NO TITLES GUIDANCE."
    *   If no suitable titles are found matching the user's current needs, the bot will now respond with: `"I wasn't able to find a specific resource that perfectly matches what you've described right now. Sometimes, telling me a bit more about what you're looking for or how you're feeling can help me find something more suitable. Would you like to try describing your needs differently, or perhaps explore a general topic?"`
    *   This is more interactive and helpful than a dead-end message.

### 4. LLM Change

*   The application was migrated from using the Mistral model via OpenRouter to **Google's Gemini API** (specifically `gemini-1.5-flash`).
*   This involved updating API call logic in `avika_chat.py`, changing relevant environment variables (from `OPENROUTER_API_KEY` to `GEMINI_API_KEY`), and adjusting `requirements.txt`.

### 5. Dependency Management

*   Reviewed and cleaned up `requirements.txt` to ensure only necessary packages are included. `langchain-text-splitters` is retained for its `RecursiveCharacterTextSplitter` used in `scripts/populate_db.py`, but the full `langchain` package was removed as it was not essential for current functionality.

These enhancements aim to create a more responsible, effective, and user-friendly chatbot experience.
