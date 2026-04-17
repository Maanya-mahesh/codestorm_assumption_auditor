import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "groq")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "huggingface")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b")

OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL  = os.getenv("OLLAMA_LLM_MODEL", "mistral:7b")

HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHROMA_PATH       = os.getenv("CHROMA_PATH", "./chromadb_store")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "assumption_auditor")

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
TOP_K_CHUNKS  = int(os.getenv("TOP_K_CHUNKS", 8))