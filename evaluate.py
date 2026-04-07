"""LLM-based paper relevance evaluation using a local HuggingFace model."""

from __future__ import annotations

import json
import sys

from search import Paper

PROMPT = """You are a strict academic reviewer evaluating whether a candidate paper should be cited.
You must distribute your verdicts — not everything is "should_cite".

RESEARCH PAPER:
{context}

CANDIDATE PAPER:
Title: {title}
Authors: {authors}
Year: {year}
Abstract: {abstract}

EXAMPLES:

must_cite (9/10): "using VLMs to generate counterfactual explanations for classifiers"
→ Nearly identical to our method.

should_cite (7/10): "diffusion-based counterfactual image generation"
→ A method component we use, but doesn't address bias discovery.

maybe_cite (4/10): "data augmentation with CutMix for robustness"
→ Tangentially related to our augmentation contribution.

skip (2/10): "adversarial robustness certification for neural networks"
→ Different problem from spurious correlations.

DISTRIBUTION: ~10-15% must_cite, ~30-40% should_cite, ~30-40% maybe_cite, ~20-30% skip.

Respond in EXACTLY this JSON (no other text):
{{
  "verdict": "must_cite" | "should_cite" | "maybe_cite" | "skip",
  "relevance_score": <1-10>,
  "relationship": "<similar_method|builds_upon|we_build_upon|same_problem|competing_approach|foundational|supports_claim|provides_benchmark|tangentially_related>",
  "cite_in_sections": ["<section1>", "<section2>"],
  "reasoning": "<2-3 sentences>",
  "differentiation": "<1 sentence if similar, else empty>"
}}"""


class LLMEvaluator:

    def __init__(self, model_id: str = "google/gemma-4-E4B-it"):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model:
            return
        print(f"Loading {self.model_id}…", file=sys.stderr)
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="auto",
        )
        print(f"Loaded on {self._model.device}", file=sys.stderr)

    def evaluate_batch(self, papers: list[Paper], context: str) -> None:
        for i, paper in enumerate(papers):
            print(f"  [{i+1}/{len(papers)}] {paper.title[:60]}…", file=sys.stderr)
            self._evaluate_single(paper, context)

    def unload(self) -> None:
        if not self._model:
            return
        del self._model, self._processor
        self._model = self._processor = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _evaluate_single(self, paper: Paper, context: str) -> None:
        try:
            response = self._generate(paper, context)
            self._parse_response(paper, response)
        except Exception as e:
            print(f"    error: {e}", file=sys.stderr)
            paper.llm_verdict = "error"
            paper.llm_reasoning = str(e)

    def _generate(self, paper: Paper, context: str) -> str:
        self.load()
        prompt = PROMPT.format(
            context=context, title=paper.title, authors=paper.authors,
            year=paper.year, abstract=paper.abstract[:1500],
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        outputs = self._model.generate(
            **inputs, max_new_tokens=512, temperature=0.2,
            top_p=0.9, do_sample=True,
        )
        return self._processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    def _parse_response(self, paper: Paper, response: str) -> None:
        data = _extract_json(response)
        if not data:
            paper.llm_verdict = "error"
            paper.llm_reasoning = f"Parse failed: {response[:200]}"
            return
        paper.llm_verdict = data.get("verdict", "skip")
        paper.llm_score = int(data.get("relevance_score", 0))
        paper.llm_relationship = data.get("relationship", "")
        paper.llm_sections = data.get("cite_in_sections", [])
        paper.llm_reasoning = data.get("reasoning", "")
        paper.llm_differentiation = data.get("differentiation", "")


def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= 0:
        return None
    try:
        return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        return None
