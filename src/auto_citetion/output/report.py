"""Generate the final markdown reading list."""

from __future__ import annotations

from ..paper import Paper

CAT_META = {
    "similar_method": (
        "MOST SIMILAR (Must Differentiate)",
        "These do something close to your method. Read to position contributions.",
        "Related Work + Introduction",
    ),
    "counterfactual_xai": (
        "COUNTERFACTUAL VISUAL EXPLANATIONS",
        "Methods generating counterfactual images to explain classifiers.",
        "Counterfactual Explanations section + Method",
    ),
    "shortcut_spurious": (
        "SHORTCUT LEARNING & SPURIOUS CORRELATIONS",
        "The core problem space. These define and measure spurious correlations.",
        "Shortcut Learning section + Introduction",
    ),
    "vlm_multimodal": (
        "VLM / MULTIMODAL FOR ANALYSIS",
        "Using VLMs to discover or mitigate bias.",
        "Explainability section + Method",
    ),
    "diffusion_editing": (
        "DIFFUSION-BASED EDITING",
        "Using diffusion models for image editing and classifier testing.",
        "Counterfactual section + Method",
    ),
    "explainability": (
        "EXPLAINABILITY & CONCEPT DISCOVERY",
        "Attribution and concept-based explanation methods.",
        "Explainability section",
    ),
    "augmentation": (
        "GENERATIVE AUGMENTATION",
        "Using generated images to retrain and debias classifiers.",
        "Data Augmentation section + Experiments",
    ),
    "other": (
        "OTHER RELEVANT",
        "Broadly relevant papers.",
        "Various sections",
    ),
}

CAT_ORDER = [
    "similar_method", "counterfactual_xai", "shortcut_spurious",
    "vlm_multimodal", "diffusion_editing", "explainability",
    "augmentation", "other",
]


def _format_paper(idx: int, p: Paper, brief: bool = False) -> list[str]:
    lines = [f"### {idx}. {p.title}", "", f"- **Authors:** {p.authors}", f"- **Year:** {p.year}"]
    if p.venue:
        lines.append(f"- **Venue:** {p.venue}")
    if p.arxiv_id:
        lines.append(f"- **arXiv:** [{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id})")
    if p.llm_verdict:
        lines.append(f"- **LLM Verdict:** {p.llm_verdict} ({p.llm_score}/10)")
        lines.append(f"- **Relationship:** {p.llm_relationship}")
        lines.append(f"- **Cite in:** {', '.join(p.llm_sections)}")
        lines.append(f"- **Why:** {p.llm_reasoning}")
        if p.llm_differentiation:
            lines.append(f"- **How we differ:** {p.llm_differentiation}")
    else:
        lines.append(f"- **Relevance score:** {p.score}")
        stypes = set(s.split(":")[0] for s in p.sources)
        lines.append(f"- **Found by:** {', '.join(sorted(stypes))} ({p.source_count} hits)")
    if not brief and p.abstract:
        short = p.abstract[:400] + "..." if len(p.abstract) > 400 else p.abstract
        lines.append(f"- **Abstract:** {short}")
    lines.append("")
    return lines


def generate_report(papers: list[Paper], max_per_category: int = 15) -> str:
    has_llm = any(p.llm_verdict for p in papers)

    if has_llm:
        must = [p for p in papers if p.llm_verdict == "must_cite"]
        should = [p for p in papers if p.llm_verdict == "should_cite"]
        maybe = [p for p in papers if p.llm_verdict == "maybe_cite"]
        skip = [p for p in papers if p.llm_verdict in ("skip", "error", "")]
    else:
        must, should, maybe, skip = [], [], [], []

    # Group by category
    grouped: dict[str, list[Paper]] = {}
    for p in papers:
        cat = p.category or "other"
        if cat not in grouped:
            grouped[cat] = []
        if len(grouped[cat]) < max_per_category:
            grouped[cat].append(p)

    total = sum(len(v) for v in grouped.values())

    lines = ["# Final Reading List", ""]

    if has_llm:
        lines.extend([
            "## Verdict Summary", "",
            "| Verdict | Count |", "|---------|-------|",
            f"| **Must Cite** | {len(must)} |",
            f"| **Should Cite** | {len(should)} |",
            f"| **Maybe Cite** | {len(maybe)} |",
            f"| Skip | {len(skip)} |",
            f"| **Total** | {len(papers)} |", "", "---", "",
            "## MUST CITE", "",
            "Essential papers — reviewers will expect these.", "",
        ])
        for i, p in enumerate(must, 1):
            lines.extend(_format_paper(i, p))
        lines.extend(["---", "", "## SHOULD CITE", "",
                      "Strong related work that strengthens the paper.", ""])
        for i, p in enumerate(should, 1):
            lines.extend(_format_paper(i, p))
        lines.extend(["---", "", "## MAYBE CITE (if space permits)", ""])
        for i, p in enumerate(maybe, 1):
            lines.extend(_format_paper(i, p, brief=True))

        # By section
        section_map: dict[str, list[Paper]] = {}
        for p in must + should:
            for sec in p.llm_sections:
                section_map.setdefault(sec, []).append(p)
        if section_map:
            lines.extend(["---", "", "## Papers by Section", ""])
            for sec in sorted(section_map):
                lines.append(f"### {sec}")
                lines.append("")
                for p in section_map[sec]:
                    tag = "**MUST**" if p.llm_verdict == "must_cite" else "SHOULD"
                    lines.append(f"- [{tag}] {p.title} ({p.year})")
                lines.append("")
    else:
        lines.extend([f"**Total candidates: {total}**", "", "---", ""])

    # By category
    lines.extend(["---", "", "## Papers by Topic", ""])
    for cat in CAT_ORDER:
        if cat not in grouped:
            continue
        title, why, where = CAT_META.get(cat, (cat, "", ""))
        lines.extend([
            f"### {title}", "",
            f"**Why read:** {why}",
            f"**Cite in:** {where}",
            f"**Count:** {len(grouped[cat])}", "",
        ])
        for i, p in enumerate(grouped[cat], 1):
            lines.extend(_format_paper(i, p))

    # Summary table
    lines.extend(["---", "", "## Summary", "",
                  "| Category | Count |", "|----------|-------|"])
    for cat in CAT_ORDER:
        if cat not in grouped:
            continue
        title = CAT_META.get(cat, (cat,))[0]
        lines.append(f"| {title} | {len(grouped[cat])} |")
    lines.extend([f"| **TOTAL** | **{total}** |", ""])

    return "\n".join(lines)
