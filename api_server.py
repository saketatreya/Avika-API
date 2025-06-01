from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Added for CORS
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
import os # For environment variables
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import traceback # Added traceback import here

load_dotenv() # Load environment variables from .env file

# Import the refactored AvikaChat class and loader
from avika_chat import AvikaChat, load_avika_titles

app = FastAPI(
    title="Avika Chat API",
    description="Mental health recommendation chatbot API",
    version="1.0.1" # Updated version
)

# --- CORS Middleware ---
# Allow all origins for simplicity in local development.
# For production, you should restrict this to your frontend's actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Global Initialization of Avika Components ---
# Ensure API key is set before initializing anything that might use it indirectly
if not os.getenv("GEMINI_API_KEY"):
    print("CRITICAL ERROR: GEMINI_API_KEY environment variable not set. API cannot start.")
    exit(1)

S_MODEL = None
AVIKA_TITLES_DATA = []
TITLE_EMBEDDINGS_DATA = {}
CHROMA_COLLECTION_INSTANCE = None
INITIALIZATION_ERROR = None

try:
    print("Initializing SentenceTransformer model...")
    S_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("SentenceTransformer model initialized.")

    print("Loading Avika titles...")
    AVIKA_TITLES_DATA = load_avika_titles() # This now checks for file existence
    print(f"Loaded {len(AVIKA_TITLES_DATA)} Avika titles.")

    if AVIKA_TITLES_DATA:
        print("Generating title embeddings...")
        TITLE_EMBEDDINGS_DATA = {
            idx: S_MODEL.encode(title["embedding_text"]) 
            for idx, title in enumerate(AVIKA_TITLES_DATA)
        }
        print("Title embeddings generated.")
    else:
        print("Warning: No Avika titles loaded, so no embeddings will be generated.")

    print("Initializing ChromaDB client...")
    chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_storage")
    persistent_client = chromadb.PersistentClient(path=chroma_db_path)
    CHROMA_COLLECTION_INSTANCE = persistent_client.get_or_create_collection(name="docx_chunks")
    print("ChromaDB client initialized and collection retrieved.")
    print("--- Avika Components Initialized Successfully ---")
except Exception as e:
    INITIALIZATION_ERROR = f"Error during global initialization: {str(e)}"
    print(f"CRITICAL ERROR: {INITIALIZATION_ERROR}")
    # In a real app, you might want to prevent startup or have a degraded mode.

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    error: Optional[str] = None # To send back initialization errors

# In-memory session storage: session_id -> AvikaChat instance
chat_sessions: Dict[str, AvikaChat] = {}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if INITIALIZATION_ERROR:
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {INITIALIZATION_ERROR}")
    # Simplified error checking: Relies on INITIALIZATION_ERROR to be comprehensive
    # if not S_MODEL or not AVIKA_TITLES_DATA or not CHROMA_COLLECTION_INSTANCE:
    #     missing_components = []
    #     if not S_MODEL: missing_components.append("Sentence Model")
    #     if not AVIKA_TITLES_DATA: missing_components.append("Avika Titles")
    #     if not CHROMA_COLLECTION_INSTANCE: missing_components.append("Chroma Collection")
    #     detail_msg = f"Service Unavailable: Core components ({', '.join(missing_components)}) not initialized."
    #     raise HTTPException(status_code=503, detail=detail_msg)

    try:
        session_id = request.session_id
        
        if not session_id or session_id not in chat_sessions:
            new_session_id = str(os.urandom(16).hex()) # Generate a more robust session ID
            session_id = new_session_id
            print(f"Creating new chat session: {session_id}")
            chat_sessions[session_id] = AvikaChat(
                model=S_MODEL,
                avika_titles=AVIKA_TITLES_DATA,
                title_embeddings=TITLE_EMBEDDINGS_DATA,
                chroma_collection=CHROMA_COLLECTION_INSTANCE
            )
        
        current_chat_instance = chat_sessions[session_id]
        
        # Get response from AvikaChat instance
        # The `chat` method in AvikaChat now manages its own turn count and history
        avika_response = current_chat_instance.chat(request.message)
        
        return ChatResponse(
            response=avika_response,
            session_id=session_id
        )
    except Exception as e:
        # import traceback # Removed from here
        print(f"Error during chat processing for session {session_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    if INITIALIZATION_ERROR:
        return {"status": "unhealthy", "reason": INITIALIZATION_ERROR}
    if not S_MODEL or not AVIKA_TITLES_DATA or not CHROMA_COLLECTION_INSTANCE:
        return {"status": "unhealthy", "reason": "Core components not initialized"}
    return {"status": "healthy", "sessions_active": len(chat_sessions)}

class ResetResponse(BaseModel):
    message: str
    initial_greeting: str
    session_id: str

@app.post("/reset", response_model=ResetResponse)
async def reset_chat_session(request: ChatRequest): # Reusing ChatRequest for session_id, message is ignored
    if INITIALIZATION_ERROR:
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {INITIALIZATION_ERROR}")
    # Simplified error checking: Relies on INITIALIZATION_ERROR to be comprehensive
    # if not S_MODEL or not AVIKA_TITLES_DATA or not CHROMA_COLLECTION_INSTANCE:
    #     missing_components = []
    #     if not S_MODEL: missing_components.append("Sentence Model")
    #     if not AVIKA_TITLES_DATA: missing_components.append("Avika Titles")
    #     if not CHROMA_COLLECTION_INSTANCE: missing_components.append("Chroma Collection")
    #     detail_msg = f"Service Unavailable: Core components ({', '.join(missing_components)}) not initialized."
    #     raise HTTPException(status_code=503, detail=detail_msg)

    session_id = request.session_id
    if not session_id or session_id not in chat_sessions:
        # If session doesn't exist, we can create a new one and reset it, 
        # or return an error. For simplicity, let's create one.
        new_session_id = str(os.urandom(16).hex())
        session_id = new_session_id
        print(f"Creating and resetting new chat session: {session_id}")
        chat_sessions[session_id] = AvikaChat(
            model=S_MODEL,
            avika_titles=AVIKA_TITLES_DATA,
            title_embeddings=TITLE_EMBEDDINGS_DATA,
            chroma_collection=CHROMA_COLLECTION_INSTANCE
        )
        # The AvikaChat __init__ already sets up the initial greeting.
        initial_greeting = chat_sessions[session_id].INITIAL_GREETING
    else:
        print(f"Resetting existing chat session: {session_id}")
        current_chat_instance = chat_sessions[session_id]
        initial_greeting = current_chat_instance.reset() # reset() returns the initial greeting

    return ResetResponse(
        message="Chat session has been reset.",
        initial_greeting=initial_greeting,
        session_id=session_id
    )

if __name__ == "__main__":
    if INITIALIZATION_ERROR:
        print(f"Cannot start server due to initialization error: {INITIALIZATION_ERROR}")
    else:
        print("Starting FastAPI server...")
        uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 