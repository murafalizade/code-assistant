from pathlib import Path
import chromadb
from typing import List
from sentence_transformers import SentenceTransformer

class ChromaStore:
    def __init__(self, persist_dir: str = "./storage", collection_name: str = "code_embeddings"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        model = SentenceTransformer('jinaai/jina-embeddings-v2-base-code', trust_remote_code=True)
        self.embedding_fn = lambda texts: model.encode(texts, show_progress_bar=False).tolist()


    def add(self, ids: List[str], texts: List[str], metadata):
        if not ids or not texts:
            return
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=self.embedding_fn(texts=texts),
            metadatas=metadata
        )
    
    def get_all(self):
        return self.collection.get(include=["documents", "embeddings", "metadatas", "ids"])

    def search(self, query: str, k: int = 2):
        query_embedding = self.embedding_fn(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        return result
