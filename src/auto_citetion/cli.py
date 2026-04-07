"""CLI entry point for auto-citetion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .paper import Paper, PaperPool
from .scoring import score_paper, categorize_paper
from .search import SemanticScholarSearch, ScholarInboxSearch, ArxivSearch
from .evaluator import LLMEvaluator
from .output import generate_report, generate_paper_files


def load_config(config_path: Path) -> dict:
    """Load search configuration from JSON file."""
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        return json.load(f)


def load_known_titles(references_path: Path) -> set[str]:
    titles = set()
    if not references_path.exists():
        return titles
    for line in references_path.read_text().splitlines():
        parts = line.split("|")
        if len(parts) >= 4:
            t = parts[2].strip().lower()
            if t and t != "title":
                titles.add(t)
    return titles


def is_known(title: str, known: set[str]) -> bool:
    tl = title.lower().strip()
    for k in known:
        if tl in k or k in tl:
            return True
        w1, w2 = set(tl.split()), set(k.split())
        if len(w1 & w2) / max(len(w1 | w2), 1) > 0.5:
            return True
    return False


def run_search(config: dict, output_dir: Path, skip_si: bool = False) -> list[Paper]:
    pool = PaperPool()

    si_cookie_path = output_dir / ".scholar_inbox_cookie"
    si_cookie = ""
    if si_cookie_path.exists():
        si_cookie = si_cookie_path.read_text().strip()

    # Scholar Inbox
    if not skip_si and si_cookie:
        si = ScholarInboxSearch(si_cookie)
        queries = config.get("scholar_inbox_queries", [])
        if queries:
            si.semantic_search(pool, queries, pages_per_query=3)
        abstract = config.get("paper_abstract", "")
        if abstract:
            ids = si.collect_paper_ids(abstract, limit=30)
            if ids:
                si.similar_papers(pool, ids[:25])
                si.paper_detail(pool, ids[:15])
    elif not skip_si:
        print("No Scholar Inbox cookie found, skipping.", file=sys.stderr)

    # Semantic Scholar
    ss = SemanticScholarSearch()
    kw_queries = config.get("semantic_scholar_queries", [])
    if kw_queries:
        ss.keyword_search(pool, kw_queries)
    seed_ids = config.get("seed_arxiv_ids", [])
    if seed_ids:
        ss.citation_chains(pool, seed_ids)
    authors = config.get("key_authors", [])
    if authors:
        ss.author_tracking(pool, authors)

    # arXiv
    arxiv_queries = config.get("arxiv_queries", [])
    if arxiv_queries:
        arxiv = ArxivSearch()
        arxiv.search(pool, arxiv_queries)

    print(f"\nTotal pool: {pool.size} papers", file=sys.stderr)
    return pool.all()


def run_score_and_filter(papers: list[Paper], known: set[str],
                          min_score: float = 4.0) -> list[Paper]:
    for p in papers:
        p.score = score_paper(p)
        p.category = categorize_paper(p)
    papers.sort(key=lambda p: p.score, reverse=True)
    filtered = [p for p in papers if not is_known(p.title, known) and p.score >= min_score]
    print(f"After scoring/filtering: {len(filtered)} papers (min_score={min_score})", file=sys.stderr)
    return filtered


def run_llm_eval(papers: list[Paper], config: dict,
                 model_id: str = "google/gemma-4-E4B-it",
                 top_n: int = 80) -> list[Paper]:
    subset = papers[:top_n]
    context = config.get("paper_context", config.get("paper_abstract", ""))
    if not context:
        print("WARNING: No paper_context in config, LLM eval may be poor", file=sys.stderr)

    evaluator = LLMEvaluator(model_id)

    def progress(i, total, title):
        print(f"  [{i+1}/{total}] {title[:60]}...", file=sys.stderr)

    evaluator.evaluate_batch(subset, context, on_progress=progress)
    evaluator.unload()

    # Re-sort by LLM score
    subset.sort(key=lambda p: p.llm_score, reverse=True)
    return subset


def main():
    parser = argparse.ArgumentParser(
        prog="auto-citetion",
        description="Automated paper discovery and relevance evaluation for academic research.",
    )
    parser.add_argument("config", help="Path to search config JSON file")
    parser.add_argument("-o", "--output-dir", default=".", help="Output directory")
    parser.add_argument("--references", default=None,
                       help="Path to references.md for dedup (default: <output_dir>/references.md)")
    parser.add_argument("--skip-search", action="store_true",
                       help="Skip search, use existing all_candidates_raw.json")
    parser.add_argument("--skip-si", action="store_true", help="Skip Scholar Inbox")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM evaluation")
    parser.add_argument("--model", default="google/gemma-4-E4B-it",
                       help="HuggingFace model for LLM evaluation")
    parser.add_argument("--top", type=int, default=80,
                       help="Number of top papers to evaluate with LLM")
    parser.add_argument("--min-score", type=float, default=4.0,
                       help="Minimum keyword score threshold")
    parser.add_argument("--fast", action="store_true", help="Quick run with reduced queries")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))

    ref_path = Path(args.references) if args.references else output_dir / "references.md"
    known = load_known_titles(ref_path)

    raw_path = output_dir / "all_candidates_raw.json"

    # ── Stage 1: Search ───────────────────────────────────────────────
    if not args.skip_search:
        print("=" * 60, file=sys.stderr)
        print("STAGE 1: Paper Search", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        papers = run_search(config, output_dir, skip_si=args.skip_si)
        papers = run_score_and_filter(papers, known, args.min_score)
        # Save raw
        with open(raw_path, "w") as f:
            json.dump([p.to_dict() for p in papers], f, indent=2)
        print(f"Saved {len(papers)} papers to {raw_path}", file=sys.stderr)
    else:
        if not raw_path.exists():
            print(f"ERROR: {raw_path} not found", file=sys.stderr)
            sys.exit(1)
        with open(raw_path) as f:
            data = json.load(f)
        papers = [Paper.from_dict(d) for d in data]
        papers = [p for p in papers if not is_known(p.title, known) and p.score >= args.min_score]
        print(f"Loaded {len(papers)} papers from {raw_path}", file=sys.stderr)

    # ── Stage 2: LLM Evaluation ──────────────────────────────────────
    if not args.skip_llm:
        print("\n" + "=" * 60, file=sys.stderr)
        print("STAGE 2: LLM Evaluation", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        top = min(args.top, 30) if args.fast else args.top
        papers = run_llm_eval(papers, config, model_id=args.model, top_n=top)
        # Save evaluated
        eval_path = output_dir / "llm_evaluated.json"
        with open(eval_path, "w") as f:
            json.dump([p.to_dict() for p in papers], f, indent=2)

    # ── Stage 3: Report ──────────────────────────────────────────────
    print("\n" + "=" * 60, file=sys.stderr)
    print("STAGE 3: Report Generation", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    report = generate_report(papers)
    report_path = output_dir / "final_reading_list.md"
    report_path.write_text(report)
    print(f"Report: {report_path}", file=sys.stderr)

    papers_dir = output_dir / "papers"
    count = generate_paper_files(papers, papers_dir)
    print(f"Paper files: {count} new in {papers_dir}/", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print("DONE", file=sys.stderr)
    print(f"  Report: {report_path}", file=sys.stderr)
    print(f"  Papers: {papers_dir}/", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
