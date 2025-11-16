# nlp_pipeline/config.py

# CPU/GPU selection
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Local embedding model (small, CPU-friendly)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Local LLM (Mistral via Ollama)
LLM_MODEL = "mistral:instruct"  # name of local Ollama model

# Chunking settings
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50
TOP_K = 3
