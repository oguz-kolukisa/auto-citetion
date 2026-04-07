"""Relevance scoring and categorization for discovered papers."""

from __future__ import annotations

from .paper import Paper

HIGH_KEYWORDS = [
    "spurious correlation", "shortcut learning", "counterfactual explanation",
    "counterfactual image", "bias discovery", "feature discovery", "model diagnosis",
    "vision-language model", "attention map", "grad-cam", "score-cam",
    "counterfactual generation", "semantic feature", "intrinsic", "contextual",
    "spurious feature", "bias auditing", "concept discovery",
]

MED_KEYWORDS = [
    "explainability", "interpretability", "debiasing", "group robustness",
    "diffusion", "image editing", "saliency", "concept-based", "attribution",
    "vlm", "multimodal", "generative explanation",
]

LOW_KEYWORDS = [
    "robustness", "distribution shift", "imagenet", "augmentation", "causal",
    "classifier", "background bias",
]

TOPIC_CATEGORIES = {
    "similar_method": [
        "counterfactual", "feature discovery", "bias discovery",
        "model diagnosis", "spurious", "attention map", "automated pipeline",
    ],
    "counterfactual_xai": [
        "counterfactual explanation", "counterfactual image",
        "counterfactual generation", "visual counterfactual",
    ],
    "shortcut_spurious": [
        "spurious correlation", "shortcut learning", "shortcut",
        "group robustness", "debiasing", "spurious feature",
    ],
    "vlm_multimodal": [
        "vision-language", "vlm", "multimodal", "clip",
        "large language model", "vision language",
    ],
    "diffusion_editing": [
        "diffusion", "image editing", "text-guided",
        "instruction-guided", "generative model",
    ],
    "explainability": [
        "explainability", "interpretability", "attribution",
        "grad-cam", "score-cam", "saliency", "concept-based",
        "concept discovery",
    ],
    "augmentation": [
        "augmentation", "data augmentation", "synthetic data",
        "generative augmentation", "training data",
    ],
}

OFF_TOPIC = ["robot", "speech", "drug", "protein", "molecule", "reinforcement learning"]

TOP_VENUES = [
    "neurips", "nips", "icml", "iclr", "cvpr", "eccv",
    "iccv", "aaai", "tmlr", "tpami", "wacv",
]


def score_paper(paper: Paper) -> float:
    text = f"{paper.title} {paper.abstract}".lower()

    s = 0.0
    s += sum(3.0 for kw in HIGH_KEYWORDS if kw in text)
    s += sum(2.0 for kw in MED_KEYWORDS if kw in text)
    s += sum(1.0 for kw in LOW_KEYWORDS if kw in text)
    s += min(paper.citation_count / 40, 8.0)

    if paper.year.isdigit():
        y = int(paper.year)
        if y >= 2023:
            s += 3.0
        if y >= 2024:
            s += 2.0
        if y >= 2025:
            s += 2.0

    if paper.venue:
        vl = paper.venue.lower()
        if any(v in vl for v in TOP_VENUES):
            s += 4.0

    if any(kw in text for kw in OFF_TOPIC) and "image" not in text and "vision" not in text:
        s -= 10.0

    # Cross-reference bonus
    s += paper.source_count * 2.0
    strategy_types = set(src.split(":")[0] for src in paper.sources)
    s += len(strategy_types) * 3.0

    return round(max(s, 0), 1)


def categorize_paper(paper: Paper) -> str:
    text = f"{paper.title} {paper.abstract}".lower()
    best, best_score = "other", 0
    for cat, kws in TOPIC_CATEGORIES.items():
        hits = sum(1 for kw in kws if kw in text)
        if hits > best_score:
            best, best_score = cat, hits
    return best
