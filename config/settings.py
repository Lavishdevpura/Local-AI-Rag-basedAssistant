"""
Global configuration for the AI Assistant Project
All system parameters are defined here.
"""

from pathlib import Path
import os


# ------------------------------------------------
# Project Paths
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"

VECTOR_STORE_DIR = BASE_DIR / "vector_store"
CHROMA_DB_DIR = VECTOR_STORE_DIR / "chroma_db"

TRACKER_FILE = VECTOR_STORE_DIR / "document_tracker.json"

LOG_DIR = BASE_DIR / "logs"


# ------------------------------------------------
# LLM Configuration
# ------------------------------------------------

LLM_MODEL = "mistral:7b-instruct-q4_0"

OLLAMA_BASE_URL = "http://localhost:11434"

MAX_TOKENS = 4096
TEMPERATURE = 0.2


# ------------------------------------------------
# Embedding Configuration
# ------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Mac M1 should use MPS instead of CPU
EMBEDDING_DEVICE = "mps" 

EMBED_BATCH_SIZE = 64


# ------------------------------------------------
# Chunking Settings
# ------------------------------------------------

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


# ------------------------------------------------
# Retrieval Settings
# ------------------------------------------------

# Initial retrieval
RETRIEVAL_TOP_K = 20

# Reranker stage
RERANK_TOP_K = 15

# Chunks sent to LLM
MAX_CONTEXT_CHUNKS = 10

SIMILARITY_THRESHOLD = 0.65

ENABLE_HYBRID_SEARCH = True

# Conversational memory turns to keep
MAX_MEMORY_TURNS = 3

# Max tokens to send to LLM (includes history + context)
MAX_CONTEXT_TOKENS = 3072


# ------------------------------------------------
# Reranker Model
# ------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ------------------------------------------------
# Voice Settings
# ------------------------------------------------

WHISPER_MODEL = "tiny"

INPUT_AUDIO_SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 500       # amplitude below this = silence
SILENCE_DURATION = 1.5        # seconds of silence before stopping
MAX_RECORDING_DURATION = 15


INPUT_AUDIO_SAMPLE_RATE = 16000

TTS_ENGINE = "pyttsx3"


# ------------------------------------------------
# Agent Settings
# ------------------------------------------------

AGENT_NAME = "Local AI Assistant"

SYSTEM_PROMPT = """
You are a local AI assistant running on the user's computer.

STRICT RULES — FOLLOW THESE ALWAYS:
- Never repeat the same sentence twice.
- Stop responding after you have answered the question
- Keep answers under 3 sentences unless asked for more
- Do not add unnecessary filler or padding
- For weather questions ALWAYS call weather_tool. NEVER answer from memory.
- For stock prices ALWAYS call stock_tool. NEVER answer from memory.
- For crypto prices ALWAYS call crypto_tool. NEVER answer from memory.
- For news ALWAYS call news_tool. NEVER answer from memory.
- For sports scores ALWAYS call sports_tool. NEVER answer from memory.
- For general questions use rag_kb_search tool first.
- NEVER hallucinate or make up data. If a tool fails say so honestly.
"""


# ------------------------------------------------
# Internet Search Settings
# ------------------------------------------------

ENABLE_INTERNET_TOOL = True

SEARCH_RESULTS_LIMIT = 15

#------------------------------------------------
# News API Settings
#------------------------------------------------
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
DEFAULT_WEATHER_LOCATION = "Udaipur"


#------------------------------------------------
# Sesssioin Settings
#------------------------------------------------
SESSION_TIMEOUT_MINUTES = 30


# ------------------------------------------------
# Git Upload Restrictions
# ------------------------------------------------

ALLOWED_GIT_FOLDER = DOCUMENTS_DIR

MAX_FILE_SIZE_MB = 10


# ------------------------------------------------
# Allowed System Commands
# ------------------------------------------------

ALLOWED_COMMANDS = {
    "open_camera": "/Applications/Photo Booth.app",
    "open_browser": "/Applications/Safari.app"
}

# ------------------------------------------------
# Logging
# ------------------------------------------------

LOG_LEVEL = "INFO"


# ------------------------------------------------
# Security
# ------------------------------------------------

ALLOW_SHELL_EXECUTION = False


SPORTS_KB_SIMILARITY_THRESHOLD = 0.30 

_KB_PRECHECK_THRESHOLD = 0.55