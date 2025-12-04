import os

from dotenv import load_dotenv
from groq import Groq

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


class GroqQwenLLM:
    def __init__(self):
        load_dotenv()
        api_key = os.environ["GROQ_API_KEY"]
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY environment variable")
        
        self.client = Groq(api_key=api_key)

    def __generate_answer(self, prompt: str, chunks) -> str:
        """
        Generate answer using all available chunks without truncating.
        """
        context = self._normalize_results(chunks)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "assistant",
                "content": f"Here is the context:\n{context}\n\nUse it ONLY if it is relevant.",
            },
            {"role": "user", "content": prompt},
        ]

        result = self.client.chat.completions.create(
            model="qwen/qwen3-32b", messages=messages, temperature=0.2
        )

        return result.choices[0].message.content

    def _normalize_results(self, r):
        parts = []
        ids = r["ids"][0]
        metas = r["metadatas"][0]
        docs = r["documents"][0]

        for i in range(len(ids)):
            meta = metas[i]
            meta_block = (
                f"// file: {meta.get('file_path')}\n"
                f"// name: {meta.get('name')}\n"
                f"// type: {meta.get('type')}\n"
                f"// lines: {meta.get('start_line')}–{meta.get('end_line')}"
            )
            parts.append(f"###\n{meta_block}\n```ts\n{docs[i]}\n```")

        return "\n\n".join(parts)

    def generate_from_chunks(self, prompt: str, chunks) -> str:
        """
        Generate answer from chunks without truncation or token limits.
        """
        return self.__generate_answer(prompt, chunks)

    def close(self):
        self.model.close()
