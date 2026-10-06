import re

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from config import LLM_MODEL


class LocalLLM:

    def __init__(self):
        print(f"Loading LLM: {LLM_MODEL}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def _fallback_answer(self, question, context):
        cleaned_context = (context or "").strip()
        if not cleaned_context:
            return "The resume does not contain enough information to answer this question."

        context_lines = [line.strip() for line in cleaned_context.splitlines() if line.strip()]
        if not context_lines:
            return "The resume does not contain enough information to answer this question."

        snippet = " ".join(context_lines[:3])
        keywords = re.findall(r"[A-Za-z][A-Za-z+/.-]{2,}", question) or ["skills", "experience"]
        keywords_text = ", ".join(keywords[:6])

        return (
            f"Based on the available resume content, the most relevant sections mention: {snippet[:400]}. "
            f"For the question about {keywords_text}, the resume provides evidence in the matching excerpts above. "
            "If you need a stricter or more detailed assessment, please review the relevant resume sections manually."
        )

    def generate(self, question, context):
        if not question or not str(question).strip():
            return "Please provide a question to analyze."

        prompt = f"""
You are an AI Resume Assistant.

Answer the user's question using ONLY
the resume information provided below.

If the information is not available,
say that it is not available in the resume.

Resume context:
{context}

User question:
{question}

Give a clear and useful answer.
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful resume analysis assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=250,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_tokens = output[0][inputs["input_ids"].shape[1]:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            cleaned_answer = answer.strip()
            if cleaned_answer:
                return cleaned_answer
        except Exception:
            pass

        return self._fallback_answer(question, context)