import os

import torch

from code_assistant.utils.code_chunk_extractor import CodeChunkExtractor
from code_assistant.vector_db.chroma_store import ChromaStore


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


def embed_project(folder_path: str, batch_size: int = 4, use_mps: bool = True):
    """
    Embed all TypeScript files in a folder and store in ChromaDB safely.

    - batch_size: number of chunks processed at once
    """
    if not folder_path or not os.path.exists(folder_path):
        raise ValueError("Invalid folder path")

    print(f"Scanning project folder: {folder_path}")

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

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    db = ChromaStore()
    processed_ids = get_processed_ids(db)
    print(f"{len(processed_ids)} chunks already embedded. Resuming...")

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]

        batch = [
            ch
            for ch in batch
            if f"{ch['file_path']}:{ch['start_line']}-{ch['end_line']}" not in processed_ids
        ]
        if not batch:
            continue

        texts = [ch["text"] for ch in batch]
        ids = [f"{ch['file_path']}:{ch['start_line']}-{ch['end_line']}" for ch in batch]
        metadata = [
            sanitize_metadata(
                {
                    "file_path": ch["file_path"],
                    "name": ch["name"],
                    "type": ch["node_type"],
                    "start_line": ch["start_line"],
                    "end_line": ch["end_line"],
                }
            )
            for ch in batch
        ]

        db.add(ids=ids, texts=texts, metadata=metadata)

        if device == "mps":
            torch.mps.empty_cache()

        print(f"Processed batch {i} → {i + len(batch)}")

    print("✅ All code chunks embedded and stored successfully!")


if __name__ == "__main__":
    path = input("Enter your project folder(e.g. '/Users/your_name/Desktop/project_folder'): ")
    embed_project(path)
