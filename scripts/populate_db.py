import os
import chromadb
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Environment variables
AVIKA_DOCS_PATH_ENV = "AVIKA_DOCS_PATH"
CHROMA_DB_PATH_ENV = "CHROMA_DB_PATH"

# ChromaDB settings
CHROMA_COLLECTION_NAME = "docx_chunks"

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
                    documents.append({"content": full_text, "source": filename})
                    print(f"  Loaded and parsed: {filename}")
                else:
                    print(f"  Warning: No text content found in {filename}")
            except Exception as e:
                print(f"  Error loading or parsing {filename}: {e}")
    return documents

def populate_vector_db():
    """Load documents, split them, embed, and store in ChromaDB."""
    docs_folder = os.getenv(AVIKA_DOCS_PATH_ENV)
    chroma_path = os.getenv(CHROMA_DB_PATH_ENV, "./chroma_storage") # Default if not set

    if not docs_folder:
        print(f"Error: Environment variable {AVIKA_DOCS_PATH_ENV} not set. Please set it to your Avika_Docs folder path.")
        return

    print("Initializing components for database population...")
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

        # 3. Split documents into chunks
        chunked_docs = []
        print("Splitting documents into chunks...")
        for doc in raw_docs:
            splits = text_splitter.split_text(doc["content"])
            for i, chunk_content in enumerate(splits):
                chunked_docs.append({
                    "content": chunk_content,
                    "metadata": {"source": doc["source"], "chunk_id": i}
                })
        print(f"Generated {len(chunked_docs)} chunks from {len(raw_docs)} documents.")
        if not chunked_docs:
            print("No chunks generated. Exiting population process.")
            return

        # 4. Initialize Sentence Transformer model (for embedding by ChromaDB)
        # ChromaDB handles the embedding if an embedding function is not directly provided to collection.add with pre-embedded data.
        # However, our avika_chat.py uses the model directly for title search. For consistency in how embeddings are made,
        # we ensure Chroma is configured to use it or we pre-embed.
        # The current avika_chat.py relies on Chroma collection doing its own embeddings based on its default or creation-time model.
        # For this script, we will let ChromaDB handle embeddings for simplicity, assuming it is set up with a compatible model like MiniLM.
        print("Initializing ChromaDB client...")
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        # This assumes SentenceTransformer model used by ChromaDB is 'sentence-transformers/all-MiniLM-L6-v2'
        # or compatible, which is often a default or can be specified at collection creation if not.
        # For ensuring the same model is used, one could pass an EmbeddingFunction to get_or_create_collection
        # from chromadb.utils import embedding_functions
        # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        # collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=sentence_transformer_ef)
        collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        print(f"Retrieved/Created ChromaDB collection: '{CHROMA_COLLECTION_NAME}' at {chroma_path}")

        # 5. Add documents to ChromaDB in batches
        # Check if already populated to avoid duplicates if script is rerun.
        # A more robust check might involve checking IDs, but count is simple for now.
        # Consider clearing the collection if a full re-population is desired: 
        # if collection.count() > 0: 
        #     print("Collection already has data. Consider clearing it if you want a fresh population.")
        # For now, we'll just add, Chroma handles ID conflicts by ignoring if ID exists.

        print("Adding documents to ChromaDB...")
        batch_size = 100 # ChromaDB recommends batches of up to ~40k docs, much smaller here.
        added_count = 0
        for i in tqdm(range(0, len(chunked_docs), batch_size), desc="Adding batches to ChromaDB"):
            batch = chunked_docs[i:i+batch_size]
            
            documents_to_add = [doc["content"] for doc in batch]
            metadatas_to_add = [doc["metadata"] for doc in batch]
            ids_to_add = [f'{meta["source"]}_chunk_{meta["chunk_id"]}' for meta in metadatas_to_add]
            
            try:
                collection.add(
                    documents=documents_to_add,
                    metadatas=metadatas_to_add,
                    ids=ids_to_add
                )
                added_count += len(documents_to_add)
            except Exception as e:
                print(f"Error adding batch {i//batch_size + 1} to ChromaDB: {e}")
        
        print(f"Successfully added/updated {added_count} document chunks to ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")
        print(f"Total chunks in collection: {collection.count()}")

    except Exception as e:
        import traceback
        print(f"An error occurred during the database population process: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting database population script...")
    populate_vector_db()
    print("Database population script finished.") 