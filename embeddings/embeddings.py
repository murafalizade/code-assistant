from sentence_transformers import SentenceTransformer
from vector_db.chroma_store import ChromaStore
from helpers.code_chunk_extractor import CodeChunkExtractor


model = SentenceTransformer('jinaai/jina-embeddings-v2-base-code', trust_remote_code=True)
with open("./sample_data/test.ts", "r") as f:
    code = f.read()

chunker = CodeChunkExtractor(code)

chunks = chunker.get_chunks()
db = ChromaStore()

for i, ch in enumerate(chunks):
  embedding = model.encode(ch['text'])
  db.add(
      ids=[f"chunk_{i}"],
      texts=[ch['text']],
      embeddings=[embedding.tolist()],
      metadata={
          "node_type": ch["node_type"],
          "name": ch["name"],
          "start_line": ch["start_line"],
          "end_line": ch["end_line"],
      }
  )

