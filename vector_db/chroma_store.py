import chromadb
from typing import List

class ChromaStore:
    def __init__(self, persist_dir: str = "./storage"):
        self.client = chromadb.PersistentClient()

        self.collection = self.client.get_or_create_collection(name="code_embeddings")

    def add(self, ids: List[str], texts: List[str], embeddings: List[List[float]], metadata):
        if not ids or not texts or not embeddings:
            return
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadata
        )
    
    def get_all(self):
        return self.collection.get(include=["documents", "embeddings"])

    def search(self, query_embedding: List[float], k: int = 2):
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return result
