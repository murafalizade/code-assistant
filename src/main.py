from code_assistant.utils.code_chunk_extractor import CodeChunkExtractor
from code_assistant.vector_db.chroma_store import ChromaStore


if __name__ == "__main__":
    with open("../sample_data/test.ts") as f:
        code = f.read()

    db = ChromaStore()
    extractor = CodeChunkExtractor(code)
    chunks = extractor.get_chunks()

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadata = [{
        "node_type": chunk["node_type"],
        "name": chunk["name"],
        "start_line": chunk["start_line"],
        "end_line": chunk["end_line"],
    } for chunk in chunks]
    texts = [chunk['text'] for chunk in chunks]
    db.add(ids=ids, texts=texts, metadata=metadata)

    results = db.search("How login is working", k=2)
    print("Search Results:", results)
    
