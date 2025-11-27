# nlp_pipeline/config.py

# CPU/GPU selection
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


LLM_MODEL = "mistral:instruct"  # name of local Ollama model

# Chunking settings
# CHUNK_SIZE = 500  # characters per chunk
# CHUNK_OVERLAP = 50
TOP_K = 3
