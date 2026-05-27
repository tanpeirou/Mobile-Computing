from llama_cpp import Llama

class RealLocalLLM:
    def __init__(self, model_path="qwen2.5-1.5b-instruct-q4_k_m.gguf"):
        print(f"[System Setup] Loading real local 4-bit LLM from {model_path} into RAM...")
        # n_ctx=512 is plenty for short audio utterances and keeps processing ultra-fast
        self.llm = Llama(model_path=model_path, n_ctx=512, verbose=False)
        print("[System Setup] Local LLM ready for on-device semantic refinement.")

    def generate(self, prompt: str) -> str:
        """Runs local 4-bit INT4 inference using a speech error correction prompt."""
        
        # Build an explicit instructional prompt structure
        system_instruction = (
            "You are an on-device ASR post-processing assistant. "
            "Your task is to fix grammar, context, and homophone errors in the raw transcript. "
            "Respond ONLY with the corrected sentence. Do not add explanations."
        )
        
        full_prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\nCorrect this: {prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Run local CPU inference
        response = self.llm(
            full_prompt,
            max_tokens=64,
            temperature=0.1,  # Low temperature prevents hallucinations and over-correction
            stop=["<|im_end|>", "\n"]
        )
        
        # Extract response text safely
        corrected_text = response["choices"][0]["text"].strip()
        return corrected_text
