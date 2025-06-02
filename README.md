# Avika - Your Supportive Chat Companion

A fully Streamlit-based chatbot application that provides mental wellness support by recommending mental health resources based on user input. It uses Qdrant for vector storage, Google Gemini for generative AI capabilities, and Streamlit's caching for optimized performance.

---

## ⚙️ Setup and Configuration

### 1. Environment Variables

Create a `.env` file in the root project directory by copying the example below. The application will automatically load these variables on startup.

```dotenv
# .env

# Required API Key for Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Required: Qdrant Connection Details
QDRANT_URL=http://localhost:6333 # Or your Qdrant Cloud URL
# QDRANT_API_KEY=your_qdrant_cloud_api_key_if_applicable # Uncomment and set if using Qdrant Cloud with API key

# Optional: Path for Avika Titles (defaults to ./Avika_Titles.docx in the project root)
# Used by the Streamlit app to load recommendation titles.
# AVIKA_TITLES_PATH=Avika_Titles.docx

# Optional: Path to the folder containing DOCX documents for the knowledge base (defaults to ./Avika_Docs/)
# Used by scripts/populate_db.py to populate the Qdrant database.
# AVIKA_DOCS_PATH=Avika_Docs/

# Optional: Name for the Qdrant collection for document chunks (defaults to 'avika_doc_chunks')
# Used by both the Streamlit app and scripts/populate_db.py.
# QDRANT_DOC_COLLECTION_NAME=avika_doc_chunks
```

*   `GEMINI_API_KEY`: **Required.** Your API key for Google Gemini.
*   `QDRANT_URL`: **Required.** The URL of your Qdrant instance (e.g., `http://localhost:6333` for a local Docker setup, or your Qdrant Cloud endpoint).
*   `QDRANT_API_KEY`: Optional. Your API key for Qdrant Cloud, if authentication is enabled.
*   `AVIKA_TITLES_PATH`: Optional. Path to the `Avika_Titles.docx` file. Defaults to `Avika_Titles.docx` in the project root.
*   `AVIKA_DOCS_PATH`: Optional. Path to the folder for DOCX documents, used by `scripts/populate_db.py`. Defaults to `Avika_Docs/` in the project root.
*   `QDRANT_DOC_COLLECTION_NAME`: Optional. Name of the Qdrant collection for document chunks. Defaults to `avika_doc_chunks`.

### 2. Qdrant Vector Database Setup

You need a running Qdrant instance for Avika to store and search document embeddings.

**Option A: Local Qdrant with Docker (Recommended for Development)**
   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
       -v $(pwd)/qdrant_storage:/qdrant/storage:z \
       qdrant/qdrant
   ```
   This creates a `qdrant_storage` directory in your project root for persistent data. Ensure `QDRANT_URL` in `.env` is `http://localhost:6333`.

**Option B: Qdrant Cloud**
   Use [Qdrant Cloud](https://cloud.qdrant.io/). Set `QDRANT_URL` and `QDRANT_API_KEY` (if applicable) in your `.env` file.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
Ensure you have Python 3.8+ installed. Consider using a virtual environment.

### 4. Populate the Vector Database (Qdrant)
This step is crucial for Avika to have a knowledge base for providing context and recommendations.
Ensure your Qdrant instance is running and configured in `.env`. Then, run:
```bash
python scripts/populate_db.py
```
This script processes documents from the folder specified by `AVIKA_DOCS_PATH` (default: `Avika_Docs/`) and titles from the file specified by `AVIKA_TITLES_PATH` (default: `Avika_Titles.docx`). It then embeds this content and stores it in your Qdrant database in the collection specified by `QDRANT_DOC_COLLECTION_NAME` (default: `avika_doc_chunks`).

## 🚀 How to Run the Streamlit Application

1.  Ensure your `.env` file is correctly configured with all required variables (see Setup).
2.  Make sure your Qdrant instance is running and accessible, and that it has been populated using `scripts/populate_db.py`.
3.  Install dependencies: `pip install -r requirements.txt`.
4.  Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```
    This will typically open the app in your web browser (e.g., at `http://localhost:8501`). The app directly uses the `AvikaChat` logic (now within `streamlit_app.py`) and connects to Qdrant as per your `.env` settings. Heavy components like models and data are cached for better performance after the initial load.

---

## 🛠️ Features

-   Session-aware conversations via Streamlit session state.
-   RAG-based mental health recommendations using Qdrant for document retrieval.
-   LLM-powered (Google Gemini) intent detection, empathetic responses, and resource recommendations.
-   Safety guardrails and crisis response guidance.
-   Resources sourced from `Avika_Titles.docx`.
-   Knowledge base context from DOCX files (via Qdrant, populated by `scripts/populate_db.py`).
-   Interactive and responsive chat interface built entirely with Streamlit.
-   Optimized performance through Streamlit's caching mechanisms (`@st.cache_resource` and `@st.cache_data`) for models and data.

---

## Tech Stack

-   Streamlit (Web application framework, including caching features)
-   Qdrant (Vector database)
-   Sentence Transformers (Embeddings generation)
-   Google Gemini API (LLM for chat and intent detection)
-   Hugging Face Transformers (RoBERTa model for offensive language detection)
-   `python-docx` (DOCX processing for titles and knowledge base documents)
-   `langchain-text-splitters` (Text splitting for document chunking - used in `scripts/populate_db.py`)
-   Python-dotenv (Environment variable management)
-   Numpy, Torch

---

## File Structure

```
.
├── streamlit_app.py     # Main Streamlit application including AvikaChat logic
├── scripts/
│   └── populate_db.py   # Script to populate Qdrant with document and title embeddings
├── Avika_Docs/          # Default folder for DOCX source documents for knowledge base
├── Avika_Titles.docx    # Default document with titles, categories, and embedding text for recommendations
├── qdrant_storage/      # (Generated by local Qdrant Docker if used)
├── requirements.txt     # Python dependencies
├── .env                 # (You create this from .env.example) Local environment variables (GITIGNORED)
├── .env.example         # Example environment file
├── .gitignore           # Git exclusions
└── README.md            # This documentation

# Deprecated / Integrated:
# ├── avika_chat.py      # Core chat logic now integrated into streamlit_app.py

# Deleted (if previously existed):
# ├── api_server.py
# ├── frontend/
# ├── render.yaml
```

---

## Deploying the Streamlit App

The easiest way to deploy a public-facing Streamlit app is using **Streamlit Community Cloud**:
1.  Push your project to a GitHub repository.
2.  Sign up/log in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
3.  Deploy your app, selecting the repository, branch, and `streamlit_app.py` as the main file.
4.  **Crucially, configure the necessary secrets (Environment Variables) in the Streamlit Cloud settings for your app.** This includes `GEMINI_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` (if your Qdrant instance requires it), `AVIKA_TITLES_PATH` (if not default), `AVIKA_DOCS_PATH` (if used by a startup script, though `populate_db.py` is typically run pre-deployment), and `QDRANT_DOC_COLLECTION_NAME` (if not default).

For other deployment options (like Dockerizing the Streamlit app and hosting it on platforms like Render, Fly.io, or Google Cloud Run), you would create a `Dockerfile` for the Streamlit app and follow the platform's deployment guides. Ensure your Qdrant database is accessible from your deployment environment.

---

## Next Steps & Potential Enhancements

*   **Refine Prompts:** Continuously test and refine the LLM prompts in `streamlit_app.py` (within `AvikaChat`) for better conversation quality and recommendation accuracy.
*   **Error Handling:** Enhance error handling and user feedback within the Streamlit app for edge cases or API issues.
*   **Advanced Monitoring:** If deployed for wider use, integrate more comprehensive logging and monitoring.
*   **CI/CD:** For regular updates, especially if using self-hosting, set up a CI/CD pipeline.
*   **Streamlit Caching Review:** Ensure caching strategies remain optimal as the app evolves.
*   **Knowledge Base Management:** Consider more sophisticated ways to manage and update the knowledge base in `Avika_Docs/` and `Avika_Titles.docx`, perhaps with versioning or a simpler update mechanism for non-technical users if needed.
