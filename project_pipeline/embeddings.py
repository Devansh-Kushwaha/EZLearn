# nlp_pipeline/embeddings.py
from sentence_transformers import SentenceTransformer
import faiss
from typing import List
from .config import EMBEDDING_MODEL
import numpy as np
import os
import pickle
import hashlib

class LocalVectorStore:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.texts = []
        
    def _hash_file(self, file_path: str) -> str:
        """
        Generate a short hash for the given file based on its contents.
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]  # short hash
    
    def _get_cache_paths(self, file_hash: str):
        """
        Get the cache paths for FAISS index and metadata.
        """
        os.makedirs("vectorstores", exist_ok=True)
        index_path = os.path.join("vectorstores", f"{file_hash}.faiss")
        meta_path = os.path.join("vectorstores", f"{file_hash}.pkl")
        return index_path, meta_path
    
    def save_index(self, file_hash: str):
        index_path, meta_path = self._get_cache_paths(file_hash)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.texts, f)
            
    def load_index(self, file_hash: str):
        index_path, meta_path = self._get_cache_paths(file_hash)
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.texts = pickle.load(f)
            return True
        return False

    def build_index(self, texts: List[str]):
        self.texts = texts
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product similarity
        self.index.add(embeddings)
        return self.index

    def retrieve(self, query: str, top_k: int = 3):
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, ids = self.index.search(q_emb, top_k)
        return [self.texts[i] for i in ids[0]]
