from typing import List
from code_assistant.vector_db.chroma_store import ChromaStore
from ctransformers import AutoModelForCausalLM
import os


class QwenLLM:
    def __init__(self):
        """
        Initialize LLM using ctransformers and a GGUF model.
        """
        model_path = os.path.join(os.path.dirname(__file__), "../../../models/qwen2.5-coder-1.5b-q8_0.gguf")
        print(os.path.exists(model_path))
        self.model = AutoModelForCausalLM.from_pretrained(model_path, model_type="qwen")

    def generate_answer(self, prompt: str, context: str = "", max_tokens: int = 256) -> str:
        """
        Generate answer using prompt and optional retrieved context.
        """
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\nAnswer:"
        output = self.model.generate(
            full_prompt,
            max_new_tokens=max_tokens,
            temperature=0.2,
        )
        return output

    def generate_from_chunks(self, prompt: str, chunks: List[str], max_tokens: int = 256) -> str:
        """
        Combine multiple retrieved code chunks into context and generate answer.
        """
        context = "\n".join(chunks)
        return self.generate_answer(prompt, context=context, max_tokens=max_tokens)


# Example usage
if __name__ == "__main__":
    llm = QwenLLM()
    prompt = "How does the AuthService login method work?"
    db = ChromaStore()
    vector = db.search(prompt)
    answer = llm.generate_from_chunks(prompt, vector['documents'])
    print("LLM Answer:\n", answer)
