import os
import torch
from typing import List
from code_assistant.utils.code_chunk_extractor import CodeChunkExtractor
from code_assistant.vector_db.chroma_store import ChromaStore

# ---------------------------
# Helper functions
# ---------------------------
def sanitize_metadata(meta: dict) -> dict:
    """Ensure all metadata values are str, int, float, or bool."""
    sanitized = {}
    for k, v in meta.items():
        if v is None:
            sanitized[k] = ""  # Use empty string for None
        else:
            sanitized[k] = v
    return sanitized

def get_processed_ids(db) -> set:
    """Retrieve IDs already stored in the DB to allow resumable embedding."""
    try:
        all_docs = db.get_all() or []
        return set([doc["id"] for doc in all_docs])
    except Exception:
        return set()

# ---------------------------
# Main embedding function
# ---------------------------
def embed_project(folder_path: str, batch_size: int = 4, use_mps: bool = True):
    """
    Embed all TypeScript files in a folder and store in ChromaDB safely.

    - batch_size: number of chunks processed at once
    - use_mps: whether to use MPS (GPU) or CPU
    """
    if not folder_path or not os.path.exists(folder_path):
        raise ValueError("Invalid folder path")

    print(f"Scanning project folder: {folder_path}")

    # 1️⃣ Collect all TS code chunks
    all_chunks = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".ts"):
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf8") as f:
                    code = f.read()
                extractor = CodeChunkExtractor(code)
                chunks = extractor.get_chunks()
                for ch in chunks:
                    ch["file_path"] = full_path
                    all_chunks.append(ch)

    if not all_chunks:
        print("No TypeScript files found.")
        return

    print(f"Found {len(all_chunks)} code chunks.")

    # 2️⃣ Initialize embedding model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3️⃣ Initialize ChromaDB
    db = ChromaStore()
    processed_ids = get_processed_ids(db)
    print(f"{len(processed_ids)} chunks already embedded. Resuming...")

    # 4️⃣ Process chunks batch by batch
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]

        # Skip already embedded chunks
        batch = [ch for ch in batch if f"{ch['file_path']}:{ch['start_line']}-{ch['end_line']}" not in processed_ids]
        if not batch:
            continue

        # Prepare inputs
        texts = [ch["text"] for ch in batch]
        ids = [f"{ch['file_path']}:{ch['start_line']}-{ch['end_line']}" for ch in batch]
        metadata = [sanitize_metadata({
            "file_path": ch["file_path"],
            "name": ch["name"],
            "type": ch["node_type"],
            "start_line": ch["start_line"],
            "end_line": ch["end_line"],
        }) for ch in batch]

        # Add to DB
        db.add(ids=ids, texts=texts, metadata=metadata)

        # Free MPS cache if using GPU
        if device == "mps":
            torch.mps.empty_cache()

        print(f"Processed batch {i} → {i + len(batch)}")

    print("✅ All code chunks embedded and stored successfully!")

if __name__ == "__main__":
    embed_project('/Users/muradeliyev/Desktop/Web Development/bizden-api')