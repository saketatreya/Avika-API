# Avika Chat API

A FastAPI-based chatbot API that provides mental wellness support by recommending mental health resources based on user input.

---

## ⚙️ Setup and Configuration

### 1. Environment Variables

Create a `.env` file in the root project directory by copying the example below. The application will automatically load these variables on startup.

```dotenv
# .env

# Required API Key for Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Paths for data files (defaults are shown)
# AVIKA_TITLES_PATH=./Avika_Titles.docx
# CHROMA_DB_PATH=./chroma_storage
# AVIKA_DOCS_PATH=./Avika_Docs/
```

*   `GEMINI_API_KEY`: **Required.** Your API key for Google Gemini.
*   `AVIKA_TITLES_PATH`: Optional. Path to the `Avika_Titles.docx` file. Defaults to `./Avika_Titles.docx`.
*   `CHROMA_DB_PATH`: Optional. Path to the directory where ChromaDB will store its data. Defaults to `./chroma_storage`.
*   `AVIKA_DOCS_PATH`: Optional. Path to the folder containing DOCX documents for the knowledge base. Defaults to `./Avika_Docs/`.

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

1.  **Create `.env` file**: Copy the template from "Environment Variables" section above and fill in your `GEMINI_API_KEY`.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Populate the vector database** (if not already done or if documents changed):
    ```bash
    python scripts/populate_db.py
    ```
4.  **Run the API server**:
    ```bash
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

### 6. Granular Crisis Detection

*   **Hybrid Approach**: The `check_safety_concerns` method in `avika_chat.py` now uses a hybrid approach for detecting safety concerns:
    *   It retains the existing keyword-based matching for specific self-harm and harm-to-others phrases.
    *   It incorporates a pre-trained Hugging Face Transformer model (`cardiffnlp/twitter-roberta-base-offensive`) to provide an additional layer of analysis. This model helps identify generally offensive or potentially concerning language that might not be caught by keywords alone.
*   **Classifier Integration**:
    *   The RoBERTa model is loaded during `AvikaChat` initialization.
    *   User input (including recent conversational history) is tokenized and passed to the classifier.
    *   If the classifier's "offensive" score for the input exceeds a configurable threshold (`SAFETY_CLASSIFIER_THRESHOLD`), it contributes to flagging a potential safety concern.
*   **Combined Logic**: A concern is raised if *either* the keyword matching *or* the classifier (above threshold) indicates a potential issue.
*   **Nuanced Logging**: The console log for safety alerts now specifies whether the alert was triggered by keywords, the classifier, or both, providing more insight into the detection.
*   **Fallback Mechanism**: If the Hugging Face model fails to load for any reason, the system gracefully falls back to using only the keyword-based detection.
*   **Limitations Note**: A note has been added acknowledging that distinguishing nuanced states like self-deprecating humor from genuine distress is a complex NLP challenge and the current classifier provides a general layer of safety rather than a perfect solution for such subtleties.

### 7. Advanced Conversational Flow & Interaction Logic (Reduced Keyword Reliance)

*   **LLM-Powered Intent Detection**:
    *   Replaced keyword-based lists (`RECOMMENDATION_KEYWORDS`, `RESISTANCE_KEYWORDS`) with direct LLM calls (`_llm_is_requesting_recommendation`, `_llm_is_user_resistant` in `avika_chat.py`).
    *   These methods prompt the Gemini model to analyze user input in conversational context to determine if they are asking for a resource or expressing resistance/hopelessness.
    *   This allows for more natural language understanding and makes the system less brittle to variations in user phrasing.
*   **State-Based Flow Control for Transitions**:
    *   Introduced boolean state flags within the `AvikaChat` class: `recommendation_offered_in_last_turn`, `avika_just_asked_for_clarification`, and `avika_just_said_no_titles_found`.
    *   These flags are set based on Avika's previous actions and are used in the `chat` method to make more intelligent decisions about when to transition to empathy or attempt a recommendation.
    *   This replaces previous logic that relied on string-matching Avika's last response, making the flow control more robust and preventing awkward immediate re-recommendations after clarification requests or if no titles were found.
*   **Nuanced Resistance Handling in Empathy Prompts**:
    *   The `_construct_empathy_prompt` method now dynamically adjusts its guidance to the LLM if the user is flagged as resistant.
    *   It further differentiates if the resistance is a direct response to a recommendation Avika just made (e.g., user says "why should I read that?"). In such cases, the LLM is prompted to acknowledge the specific skepticism about the resource and explore the user's reservations directly.
    *   If resistance is more general, broader empathetic validation is prompted.
*   **Conversation Reset Functionality**:
    *   Added a `reset()` method to the `AvikaChat` class to clear conversation history and reset all state flags.
    *   Exposed this via a new `/reset` POST endpoint in `api_server.py`.
    *   The frontend (`frontend/index.html`) now includes a "Reset Conversation" button that calls this endpoint.
    *   The frontend also now initializes the chat by calling `/reset` on page load to fetch the initial greeting and session ID, ensuring a clean start.

These enhancements aim to create a more responsible, effective, and user-friendly chatbot experience.
