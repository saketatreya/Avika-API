import os
# import chromadb # Replaced with qdrant_client
from qdrant_client import QdrantClient, models # Qdrant imports
from docx import Document
from sentence_transformers import SentenceTransformer # Keep for embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv
import traceback
import uuid # For generating point IDs

load_dotenv()

# Environment variables
AVIKA_DOCS_PATH_ENV = "AVIKA_DOCS_PATH"
# CHROMA_DB_PATH_ENV = "CHROMA_DB_PATH" # No longer needed for Qdrant population script
QDRANT_URL_ENV = "QDRANT_URL" # For Qdrant server
QDRANT_API_KEY_ENV = "QDRANT_API_KEY" # Optional, for Qdrant Cloud

# Qdrant settings (should match AvikaChat)
QDRANT_COLLECTION_NAME = "avika_doc_chunks"
VECTOR_SIZE = 384  # For all-MiniLM-L6-v2

# Text splitting settings
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

def load_docx_from_folder(folder_path):
    """Load all .docx files from a specified folder."""
    documents = []
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return documents

    print(f"Loading .docx files from: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx") and not filename.startswith(".~"):
            path = os.path.join(folder_path, filename)
            try:
                doc = Document(path)
                full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])
                if full_text:
                    # Store the raw text content directly for payload
                    documents.append({"text_content": full_text, "source": filename})
                    print(f"  Loaded and parsed: {filename}")
                else:
                    print(f"  Warning: No text content found in {filename}")
            except Exception as e:
                print(f"  Error loading or parsing {filename}: {e}")
    return documents

def populate_vector_db():
    """Load documents, split them, embed, and store in QdrantDB."""
    docs_folder = os.getenv(AVIKA_DOCS_PATH_ENV)
    qdrant_url = os.getenv(QDRANT_URL_ENV, "http://localhost:6333") # Default to local Qdrant
    qdrant_api_key = os.getenv(QDRANT_API_KEY_ENV) # Will be None if not set

    if not docs_folder:
        print(f"Error: Environment variable {AVIKA_DOCS_PATH_ENV} not set. Please set it to your Avika_Docs folder path.")
        return

    print("Initializing components for Qdrant database population...")
    try:
        # 1. Load documents
        raw_docs = load_docx_from_folder(docs_folder)
        if not raw_docs:
            print("No documents loaded. Exiting population process.")
            return
        print(f"Successfully loaded {len(raw_docs)} documents.")

        # 2. Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
            length_function=len
        )
        print("Text splitter initialized.")

        # 3. Split documents into chunks for payload
        payload_chunks = []
        print("Splitting documents into chunks for payload...")
        for doc in raw_docs:
            splits = text_splitter.split_text(doc["text_content"])
            for i, chunk_text in enumerate(splits):
                payload_chunks.append({
                    "text_chunk": chunk_text, # The actual text snippet
                    "source": doc["source"],
                    "chunk_id": i
                })
        print(f"Generated {len(payload_chunks)} text chunks from {len(raw_docs)} documents.")
        if not payload_chunks:
            print("No chunks generated. Exiting population process.")
            return

        # 4. Initialize Sentence Transformer model for embedding
        print("Initializing SentenceTransformer model...")
        s_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("SentenceTransformer model initialized.")

        # 5. Initialize Qdrant client
        print(f"Initializing Qdrant client. URL: {qdrant_url}")
        if qdrant_api_key:
            qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            print("Connected to Qdrant Cloud with API key.")
        else:
            qdrant_client = QdrantClient(url=qdrant_url)
            print("Connected to Qdrant (likely local) without API key.")

        # Ensure the collection exists, create if not
        try:
            collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' already exists with {collection_info.points_count} points.")
            # Optional: You might want to offer to clear it or check vector_size/distance compatibility
            # For now, we assume if it exists, it's compatible or we're adding to it.
        except Exception as e: # Catching a more specific exception if possible (e.g., from qdrant_client.http.exceptions)
            print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' not found or error: {e}. Creating it...")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
            )
            print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created.")

        # 6. Embed documents and upsert to Qdrant in batches
        print("Embedding documents and upserting to Qdrant...")
        batch_size = 100 
        added_count = 0
        points_to_upsert = []

        for i in tqdm(range(0, len(payload_chunks), batch_size), desc="Processing batches for Qdrant"):
            batch_payloads = payload_chunks[i:i+batch_size]
            
            # Prepare points for Qdrant
            current_batch_points = []
            for payload_item in batch_payloads:
                # Embed the text chunk
                vector = s_model.encode(payload_item["text_chunk"]).tolist()
                # Create a unique ID for the point
                point_id = str(uuid.uuid4())
                current_batch_points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload_item # Store the text_chunk, source, chunk_id in payload
                ))
            
            if current_batch_points:
                try:
                    qdrant_client.upsert(
                        collection_name=QDRANT_COLLECTION_NAME,
                        points=current_batch_points,
                        wait=True # Wait for operation to complete
                    )
                    added_count += len(current_batch_points)
                except Exception as e:
                    print(f"Error upserting batch to Qdrant: {e}")
        
        print(f"Successfully upserted {added_count} points to Qdrant collection '{QDRANT_COLLECTION_NAME}'.")
        collection_info_after = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Total points in collection: {collection_info_after.points_count}")

    except Exception as e:
        print(f"An error occurred during the Qdrant database population process: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting Qdrant database population script...")
    populate_vector_db()
    print("Qdrant database population script finished.") 