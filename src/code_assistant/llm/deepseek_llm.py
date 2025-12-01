from typing import List
from code_assistant.vector_db.chroma_store import ChromaStore
from llama_cpp import Llama
import os
import torch

SYSTEM_PROMPT = """
    You are Bizden Code Assistant, a helpful AI assistant specialized in software development and code understanding. 
    Guidelines for you:
    1. Always provide clear, concise, and accurate explanations.
    2. Focus on code, algorithms, debugging, and best practices.
    3. Never provide harmful instructions, exploits, or personal advice.
    4. Use examples when helpful.
    5. Keep responses professional and beginner-friendly if requested.
    6. If you don’t know something, admit it instead of guessing.
    7. Give code in the language requested or present in the context.
    8. If the provided context does not match the question, ignore the context and answer based on the question only.
    9. Only answer questions related to coding, code explanation, code generation, software projects, and technology. For any other type of question, respond: "I am specialized in coding and technology questions, and cannot provide advice on this topic."
"""

class DeepSeekLLM:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "../../../models/deepseek-coder-7b-instruct-v1.5-Q5_K_M.gguf"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        torch.mps.empty_cache()
        self.model = Llama(model_path=model_path, 
                           n_gpu_layers=0, 
                           mps=False, 
                           n_ctx=4096, 
                           n_threads=8)

    def _generate_answer(self, prompt: str, chunks) -> str:
        """
        Generate answer using all available chunks without truncating.
        """
        chunks = self._normalize_results(chunks)
        context = self._make_llm_context(chunks)

        messages = [
            # {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": f"Here is the context:\n{context}\n\nUse it ONLY if it is relevant."},
            {"role": "user", "content": prompt},
        ]

        result = self.model.create_chat_completion(
            messages=messages,
            temperature=0.2
        )

        return result["choices"][0]["message"]["content"].strip()

    def _make_llm_context(self, chunks):
        parts = []
        for c in chunks:
            print(c)
            parts.append(f"###\n```ts\n{c['code']}\n```")
        return "\n\n".join(parts)
    
    def _normalize_results(self, r):
        out = []
        ids = r["ids"][0]
        dists = r["distances"][0]
        metas = r["metadatas"][0]
        docs = r["documents"][0]

        for i in range(len(ids)):
            out.append({
                "id": ids[i],
                "distance": dists[i],
                "meta": metas[i],
                "code": docs[i],
            })

        return out
    
    def generate_from_chunks(self, prompt: str, chunks) -> str:
        """
        Generate answer from chunks without truncation or token limits.
        """
        return self._generate_answer(prompt, chunks)

    def close(self):
        self.model.close()