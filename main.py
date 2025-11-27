from typing import List
from sentence_transformers import SentenceTransformer
import torch
from vector_db.chroma_store import ChromaStore


# Example: using a local Qwen model
from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenLLM:
    def __init__(self, model_name: str = "Qwen/Qwen-3B-Code", device: str = "cpu"):
        """
        Initialize LLM with tokenizer and model.
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        ).to(self.device)

    def generate_answer(self, prompt: str, context: str = "", max_tokens: int = 256) -> str:
        """
        Generate answer using prompt and optional retrieved context.
        """
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\nAnswer:"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.2
        )

        return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    def generate_from_chunks(self, prompt: str, chunks: List[str], max_tokens: int = 256) -> str:
        """
        Combine multiple retrieved code chunks into context and generate answer.
        """
        context = "\n".join(chunks)
        return self.generate_answer(prompt, context=context, max_tokens=max_tokens)


# Example usage
if __name__ == "__main__":
    llm = QwenLLM()
    context_chunks = [
        "class AuthService { constructor() {} async login() { return 'ok'; } }",
        "function sum(a, b) { return a + b; }"
    ]
    model = SentenceTransformer('jinaai/jina-embeddings-v2-base-code', trust_remote_code=True)
    prompt = "How does the AuthService login method work?"
    db = ChromaStore()
    embedding = model.encode(prompt)
    vector = db.search(embedding)
    print(vector)
    answer = llm.generate_from_chunks(prompt, vector['documents'])
    print("LLM Answer:\n", answer)
