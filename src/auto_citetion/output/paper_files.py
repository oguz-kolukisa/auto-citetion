"""Generate individual markdown files for each paper."""

from __future__ import annotations

import re
from pathlib import Path

from ..paper import Paper


def generate_paper_files(papers: list[Paper], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in papers:
        tag = (p.llm_verdict or p.category or "OTHER").upper().replace("_", "-")
        slug = re.sub(r"[^a-z0-9\s-]", "", p.title.lower().strip())
        slug = re.sub(r"[\s]+", "_", slug)[:70]
        fname = f"{tag}_{slug}.md"
        filepath = output_dir / fname
        if filepath.exists():
            continue

        arxiv_line = ""
        if p.arxiv_id:
            arxiv_line = f"**arXiv:** [{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id})\n"

        llm_section = ""
        if p.llm_verdict:
            llm_section = f"""
## LLM Evaluation

- **Verdict:** {p.llm_verdict}
- **Score:** {p.llm_score}/10
- **Relationship:** {p.llm_relationship}
- **Cite in:** {', '.join(p.llm_sections)}
- **Reasoning:** {p.llm_reasoning}
- **Differentiation:** {p.llm_differentiation}
"""

        content = f"""# {p.title}

**Authors:** {p.authors}
**Year:** {p.year}
**Venue:** {p.venue}
{arxiv_line}**Relevance score:** {p.score}
{llm_section}
## Abstract

{p.abstract or 'Abstract not available.'}
"""
        filepath.write_text(content)
        count += 1
    return count
