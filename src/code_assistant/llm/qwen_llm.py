import os

from llama_cpp import Llama

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


class QwenLLM:
    def __init__(self, model_path: str = None):
        """
        Initialize the LLM with a local GGUF model using llama_cpp.
        """
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "../../../models/qwen2.5-coder-1.5b-q8_0.gguf"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = Llama(model_path=model_path)
        self.max_context_chars = 3000

    def _generate_answer(self, prompt: str, chunks, max_tokens: int = 256) -> str:
        chunks = self._normalize_results(chunks)
        chunks = self._truncate_chunks_by_context(chunks, max_tokens=350)
        context = self._make_llm_context(chunks)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "assistant",
                "content": f"Here is the context:\n{context}\n\nUse it ONLY if it is relevant.",
            },
            {"role": "user", "content": prompt},
        ]

        result = self.model.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=0.7
        )

        return result["choices"][0]["message"]["content"].strip()

    def _make_llm_context(self, chunk):
        parts = []
        for c in chunk:
            block = f"""
                ### 
                ```ts
                {c["code"]}
                """
            parts.append(block.strip())
        return "\n\n".join(parts)

    def _normalize_results(self, r):
        out = []
        ids = r["ids"][0]
        dists = r["distances"][0]
        metas = r["metadatas"][0]
        docs = r["documents"][0]

        for i in range(len(ids)):
            out.append(
                {
                    "id": ids[i],
                    "distance": dists[i],
                    "meta": metas[i],
                    "code": docs[i],
                }
            )

        return out

    def _truncate_chunks_by_context(self, chunks, max_tokens=500):
        """
        Keep chunks until total estimated tokens < max_tokens.
        Roughly assume 1 token ≈ 4 chars.
        """
        selected = []
        total_tokens = 0
        for ch in chunks:
            est_tokens = len(ch["code"]) // 4
            if total_tokens + est_tokens > max_tokens:
                break
            selected.append(ch)
            total_tokens += est_tokens
        return selected

    def generate_from_chunks(self, prompt: str, chunks, max_tokens: int = 256) -> str:
        """
        Combine multiple retrieved code chunks into context and generate answer.
        Truncates context if too long.
        """
        return self._generate_answer(prompt, chunks=chunks, max_tokens=max_tokens)

    def close(self):
        """
        Explicitly close the model to avoid destructor warnings.
        """
        self.model.close()
