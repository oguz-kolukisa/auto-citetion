"""Local LLM evaluator for paper relevance scoring using HuggingFace transformers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from ..paper import Paper

EVAL_PROMPT = """You are a research assistant evaluating whether a candidate paper should be cited in a given research paper.

RESEARCH PAPER:
{context}

CANDIDATE PAPER:
Title: {title}
Authors: {authors}
Year: {year}
Abstract: {abstract}

Evaluate and respond in EXACTLY this JSON format (no other text):
{{
  "verdict": "must_cite" | "should_cite" | "maybe_cite" | "skip",
  "relevance_score": <1-10>,
  "relationship": "<one of: similar_method, builds_upon, we_build_upon, same_problem, competing_approach, foundational, supports_claim, provides_benchmark, provides_method_component, tangentially_related>",
  "cite_in_sections": ["<section1>", "<section2>"],
  "reasoning": "<2-3 sentences explaining why to cite or skip>",
  "differentiation": "<if similar method: 1 sentence on how the research paper differs, else empty string>"
}}

GUIDELINES:
- "must_cite": Very similar method, foundational work, or directly addresses same problem
- "should_cite": Relevant related work that reviewers would expect to see
- "maybe_cite": Tangentially related, cite only if space permits
- "skip": Not relevant enough
- Be strict: limited space, only recommend genuinely useful citations"""


@dataclass
class EvalResult:
    verdict: str = "skip"
    relevance_score: int = 0
    relationship: str = ""
    cite_in_sections: list[str] = field(default_factory=list)
    reasoning: str = ""
    differentiation: str = ""


class LLMEvaluator:
    """Evaluates paper relevance using a local LLM."""

    def __init__(self, model_id: str = "google/gemma-4-E4B-it"):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        print(f"Loading {self.model_id}...", file=sys.stderr)
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Model loaded on {self._model.device}", file=sys.stderr)

    def evaluate(self, paper: Paper, context: str) -> EvalResult:
        self.load()

        prompt = EVAL_PROMPT.format(
            context=context,
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            abstract=paper.abstract[:1500],
        )

        messages = [{"role": "user", "content": prompt}]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )
        response = self._processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        return self._parse(response)

    def evaluate_batch(self, papers: list[Paper], context: str,
                       on_progress=None) -> None:
        for i, paper in enumerate(papers):
            if on_progress:
                on_progress(i, len(papers), paper.title)
            try:
                result = self.evaluate(paper, context)
                paper.llm_verdict = result.verdict
                paper.llm_score = result.relevance_score
                paper.llm_relationship = result.relationship
                paper.llm_sections = result.cite_in_sections
                paper.llm_reasoning = result.reasoning
                paper.llm_differentiation = result.differentiation
            except Exception as e:
                print(f"  LLM error: {e}", file=sys.stderr)
                paper.llm_verdict = "error"
                paper.llm_reasoning = str(e)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _parse(self, response: str) -> EvalResult:
        response = response.strip()
        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1 or end == 0:
            return EvalResult(reasoning=f"Parse failed: {response[:200]}")
        try:
            data = json.loads(response[start:end])
            return EvalResult(
                verdict=data.get("verdict", "skip"),
                relevance_score=int(data.get("relevance_score", 0)),
                relationship=data.get("relationship", ""),
                cite_in_sections=data.get("cite_in_sections", []),
                reasoning=data.get("reasoning", ""),
                differentiation=data.get("differentiation", ""),
            )
        except json.JSONDecodeError:
            return EvalResult(reasoning=f"JSON error: {response[:200]}")
