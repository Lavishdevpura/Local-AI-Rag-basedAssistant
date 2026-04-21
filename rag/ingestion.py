import os
import json
import hashlib
import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

from config.settings import (
    DOCUMENTS_DIR,
    CHROMA_DB_DIR,
    TRACKER_FILE,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    EMBED_BATCH_SIZE,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


# ------------------------------------------------
# Hash Utility
# ------------------------------------------------

def compute_file_hash(file_path):
    """Compute SHA256 hash of a file"""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


# ------------------------------------------------
# Tracker Utilities
# ------------------------------------------------

def load_tracker():
    """Load document tracker"""

    if not os.path.exists(TRACKER_FILE):
        return {}

    if os.path.getsize(TRACKER_FILE) == 0:
        return {}

    with open(TRACKER_FILE, "r") as f:
        return json.load(f)


def save_tracker(tracker):
    """Save tracker file"""

    os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)

    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=4)


# ------------------------------------------------
# Detect New / Updated Files
# ------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
SKIP_FILES = {".DS_Store", ".gitkeep", "Thumbs.db", "desktop.ini"}

def detect_changed_files(tracker):
    changed_files = []
    for file in os.listdir(DOCUMENTS_DIR):
        # Skip hidden files and system files
        if file.startswith("."):
            continue
        if file in SKIP_FILES:
            continue
        ext = os.path.splitext(file)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        file_path = os.path.join(DOCUMENTS_DIR, file)
        if not os.path.isfile(file_path):
            continue
        file_hash = compute_file_hash(file_path)
        if file not in tracker or tracker[file]["hash"] != file_hash:
            changed_files.append((file, file_hash))
    return changed_files



# ------------------------------------------------
# Configure LlamaIndex Settings
# ------------------------------------------------

def configure_llamaindex():

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device=EMBEDDING_DEVICE,
        embed_batch_size=EMBED_BATCH_SIZE
    )

    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )


# ------------------------------------------------
# Build / Update Vector DB
# ------------------------------------------------

def build_vector_store(files, tracker):

    if not files:
        print("No new or modified documents found.")
        return

    print("New or modified documents detected:")
    print([f[0] for f in files])

    file_paths = [os.path.join(DOCUMENTS_DIR, f[0]) for f in files]

    documents = SimpleDirectoryReader(
        input_files=file_paths
    ).load_data()

    # Add metadata
    for doc in documents:
        doc.metadata["source"] = doc.metadata.get("file_name", "unknown")

    print("Connecting to ChromaDB...")

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR)
    )

    collection = chroma_client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    print("Generating embeddings and storing vectors...")

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    # Update tracker
    for file, file_hash in files:

        tracker[file] = {
            "hash": file_hash
        }

    save_tracker(tracker)

    print("Vector database successfully updated.")


# ------------------------------------------------
# Ingestion Pipeline
# ------------------------------------------------

def run_ingestion():

    print("Configuring embedding + chunking settings...")
    configure_llamaindex()

    print("Loading document tracker...")
    tracker = load_tracker()

    print("Scanning for changed documents...")
    changed_files = detect_changed_files(tracker)

    build_vector_store(changed_files, tracker)


# ------------------------------------------------
# Run Script
# ------------------------------------------------

if __name__ == "__main__":
    run_ingestion()

