#!/usr/bin/env python3
"""Auto-Citetion: find and evaluate papers for your research."""

import argparse
import json
import re
import sys
from pathlib import Path

from auto_citetion.search import (
    Paper, PaperPool, score_and_categorize, si_collect_ids,
    run_api_threads,
    job_ss_keyword, job_ss_citations, job_ss_author,
    job_si_semantic, job_si_similar, job_si_detail,
    job_arxiv, job_oalex_search, job_oalex_cited_by,
    job_dblp_search, job_dblp_venue,
    job_gs_search, job_gs_cited_by, job_gs_author,
)
from auto_citetion.evaluate import LLMEvaluator


# ── Dedup helpers ─────────────────────────────────────────────────────────

def load_known_titles(path: Path) -> set[str]:
    if not path.exists():
        return set()
    titles = set()
    for line in path.read_text().splitlines():
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


def filter_novel(papers: list[Paper], known: set[str], min_score: float) -> list[Paper]:
    filtered = [p for p in papers if not is_known(p.title, known) and p.score >= min_score]
    filtered.sort(key=lambda p: p.score, reverse=True)
    return filtered


# ── Cookie loading ────────────────────────────────────────────────────────

def load_cookie(output_dir: Path) -> str:
    for path in [output_dir / ".scholar_inbox_cookie",
                 Path(__file__).parent.parent.parent / ".scholar_inbox_cookie"]:
        if path.exists():
            return path.read_text().strip()
    return ""


# ── Job building ──────────────────────────────────────────────────────────

def build_ss_jobs(cfg: dict) -> list:
    jobs = []
    for i, q in enumerate(cfg.get("semantic_scholar_queries", [])):
        jobs.append(lambda q=q, i=i: job_ss_keyword(q, i))
    for aid in cfg.get("seed_arxiv_ids", []):
        jobs.append(lambda a=aid: job_ss_citations(a))
    for name in cfg.get("key_authors", []):
        jobs.append(lambda n=name: job_ss_author(n))
    return jobs


def build_si_jobs(cfg: dict, cookie: str) -> list:
    jobs = []
    for i, q in enumerate(cfg.get("scholar_inbox_queries", [])):
        jobs.append(lambda q=q, i=i: job_si_semantic(q, i, cookie))
    abstract = cfg.get("paper_abstract", "")
    if abstract:
        paper_ids = si_collect_ids(abstract, cookie)
        for pid in paper_ids[:25]:
            jobs.append(lambda pid=pid: job_si_similar(pid, cookie))
        for pid in paper_ids[:15]:
            jobs.append(lambda pid=pid: job_si_detail(pid, cookie))
    return jobs


def build_oalex_jobs(cfg: dict) -> list:
    jobs = []
    for i, q in enumerate(cfg.get("semantic_scholar_queries", [])):
        jobs.append(lambda q=q, i=i: job_oalex_search(q, i))
    for aid in cfg.get("seed_arxiv_ids", []):
        jobs.append(lambda a=aid: job_oalex_cited_by(a))
    return jobs


def build_dblp_jobs(cfg: dict) -> list:
    jobs = []
    for i, q in enumerate(cfg.get("semantic_scholar_queries", [])):
        jobs.append(lambda q=q, i=i: job_dblp_search(q, i))
    for venue, year in cfg.get("dblp_venues", []):
        jobs.append(lambda v=venue, y=year: job_dblp_venue(v, y))
    return jobs


def build_arxiv_jobs(cfg: dict) -> list:
    return [lambda q=q, i=i: job_arxiv(q, i)
            for i, q in enumerate(cfg.get("arxiv_queries", []))]


def build_gs_jobs(cfg: dict) -> list:
    jobs = []
    for i, q in enumerate(cfg.get("semantic_scholar_queries", [])):
        jobs.append(lambda q=q, i=i: job_gs_search(q, i))
    for title in cfg.get("google_scholar_cite_titles", []):
        jobs.append(lambda t=title: job_gs_cited_by(t))
    return jobs


# ── Search stage ──────────────────────────────────────────────────────────

def run_initial_search(pool: PaperPool, cfg: dict, cookie: str,
                       skip_si: bool, skip_gs: bool) -> None:
    api_jobs = {"SS": build_ss_jobs(cfg), "arXiv": build_arxiv_jobs(cfg),
                "OpenAlex": build_oalex_jobs(cfg), "DBLP": build_dblp_jobs(cfg)}
    if cookie and not skip_si:
        api_jobs["SI"] = build_si_jobs(cfg, cookie)
    elif not skip_si:
        print("No .scholar_inbox_cookie found, skipping Scholar Inbox.", file=sys.stderr)
    if not skip_gs:
        api_jobs["GS"] = build_gs_jobs(cfg)

    total = sum(len(j) for j in api_jobs.values())
    print(f"\nJobs: {' '.join(f'{k}={len(v)}' for k, v in api_jobs.items())} (total={total})",
          file=sys.stderr)
    run_api_threads(pool, api_jobs)


# ── Recursive expansion ──────────────────────────────────────────────────

def run_recursive_expansion(pool: PaperPool, known: set[str], min_score: float,
                            cookie: str, skip_gs: bool,
                            max_depth: int, expand_top: int) -> None:
    for depth in range(1, max_depth + 1):
        seeds = _select_expansion_seeds(pool, known, min_score, expand_top)
        if not seeds:
            print(f"\n[Depth {depth}] No seeds, stopping.", file=sys.stderr)
            break

        before = pool.size
        print(f"\n{'='*50}\n[Depth {depth}] Expanding {len(seeds)} papers\n{'='*50}", file=sys.stderr)

        api_jobs = _build_expansion_jobs(seeds, cookie, skip_gs)
        run_api_threads(pool, api_jobs)

        new_found = pool.size - before
        print(f"[Depth {depth}] +{new_found} new (total: {pool.size})", file=sys.stderr)
        if new_found < 5:
            break


def _select_expansion_seeds(pool: PaperPool, known: set[str],
                            min_score: float, limit: int) -> list[Paper]:
    papers = pool.all()
    score_and_categorize(papers)
    novel = [p for p in papers if not is_known(p.title, known) and p.score >= min_score]
    novel.sort(key=lambda p: p.score, reverse=True)
    return novel[:limit]


def _build_expansion_jobs(seeds: list[Paper], cookie: str,
                          skip_gs: bool) -> dict[str, list]:
    ss_seeds = [p for p in seeds if p.arxiv_id]
    ss_jobs = [lambda a=p.arxiv_id: job_ss_citations(a) for p in ss_seeds]
    si_jobs = _build_si_expansion_jobs(seeds, cookie)
    gs_jobs = _build_gs_expansion_jobs(seeds) if not skip_gs else []
    return {"SS": ss_jobs, "SI": si_jobs, "GS": gs_jobs}


def _build_si_expansion_jobs(seeds: list[Paper], cookie: str) -> list:
    if not cookie:
        return []
    si_ids = _collect_si_ids_for_seeds(seeds, cookie)
    return [lambda pid=pid: job_si_similar(pid, cookie) for pid in si_ids[:40]]


def _build_gs_expansion_jobs(seeds: list[Paper]) -> list:
    jobs = []
    top_seeds = seeds[:10]
    for p in top_seeds:
        jobs.append(lambda t=p.title: job_gs_cited_by(t))
    author_names = _extract_top_authors(seeds)
    for name in author_names:
        jobs.append(lambda n=name: job_gs_author(n))
    return jobs


def _extract_top_authors(seeds: list[Paper]) -> list[str]:
    author_counts: dict[str, int] = {}
    for p in seeds:
        for name in _split_author_string(p.authors):
            author_counts[name] = author_counts.get(name, 0) + 1
    ranked = sorted(author_counts, key=lambda n: author_counts[n], reverse=True)
    return ranked[:5]


def _split_author_string(authors: str) -> list[str]:
    cleaned = authors.replace(" et al.", "")
    return [n.strip() for n in cleaned.split(",") if n.strip()]


def _collect_si_ids_for_seeds(seeds: list[Paper], cookie: str) -> list[int]:
    ids = []
    for p in seeds:
        ids.extend(si_collect_ids(p.title, cookie, limit=5))
        if len(ids) >= 40:
            break
    return ids


# ── LLM evaluation stage ─────────────────────────────────────────────────

def run_llm_evaluation(papers: list[Paper], cfg: dict,
                       model_id: str, top_n: int) -> list[Paper]:
    subset = papers[:top_n]
    context = cfg.get("paper_context", cfg.get("paper_abstract", ""))
    evaluator = LLMEvaluator(model_id)
    evaluator.evaluate_batch(subset, context)
    evaluator.unload()
    return subset


# ── Report generation ─────────────────────────────────────────────────────

def generate_report(papers: list[Paper], category_labels: dict[str, tuple]) -> str:
    has_llm = any(p.llm_verdict and p.llm_verdict != "error" for p in papers)
    lines = ["# Final Reading List", ""]
    if has_llm:
        lines += _build_verdict_sections(papers)
        lines += _build_section_index(papers)
    else:
        lines.append(f"**{len(papers)} papers found** (no LLM evaluation)\n")
    lines += _build_topic_sections(papers, category_labels)
    return "\n".join(lines)


def _build_verdict_sections(papers: list[Paper]) -> list[str]:
    groups = {
        "must_cite": [p for p in papers if p.llm_verdict == "must_cite"],
        "should_cite": [p for p in papers if p.llm_verdict == "should_cite"],
        "maybe_cite": [p for p in papers if p.llm_verdict == "maybe_cite"],
    }
    skip_count = len(papers) - sum(len(g) for g in groups.values())
    lines = [
        "## Summary", "",
        "| Verdict | Count |", "|---------|-------|",
        f"| **Must Cite** | {len(groups['must_cite'])} |",
        f"| **Should Cite** | {len(groups['should_cite'])} |",
        f"| **Maybe Cite** | {len(groups['maybe_cite'])} |",
        f"| Skip | {skip_count} |", "", "---", "",
    ]
    for label, key in [("MUST CITE", "must_cite"), ("SHOULD CITE", "should_cite"),
                       ("MAYBE CITE", "maybe_cite")]:
        if groups[key]:
            lines += [f"## {label}", ""]
            for i, p in enumerate(groups[key], 1):
                lines += _format_paper(i, p)
            lines += ["---", ""]
    return lines


def _build_section_index(papers: list[Paper]) -> list[str]:
    sec_map: dict[str, list[Paper]] = {}
    for p in papers:
        if p.llm_verdict not in ("must_cite", "should_cite"):
            continue
        for s in p.llm_sections:
            sec_map.setdefault(s, []).append(p)
    if not sec_map:
        return []
    lines = ["## Papers by Section", ""]
    for sec in sorted(sec_map):
        lines.append(f"### {sec}")
        for p in sec_map[sec]:
            tag = "**MUST**" if p.llm_verdict == "must_cite" else "SHOULD"
            lines.append(f"- [{tag}] {p.title} ({p.year})")
        lines.append("")
    return lines


def _build_topic_sections(papers: list[Paper], category_labels: dict[str, tuple]) -> list[str]:
    grouped: dict[str, list[Paper]] = {}
    for p in papers:
        grouped.setdefault(p.category or "other", []).append(p)
    lines = ["---", "", "## Papers by Topic", ""]
    for cat in list(category_labels) + ["other"]:
        if cat not in grouped:
            continue
        label, where = category_labels.get(cat, (cat, ""))
        lines += [f"### {label}", f"*Cite in: {where}*", ""]
        for i, p in enumerate(grouped[cat][:15], 1):
            lines += _format_paper(i, p, brief=True)
    return lines


def _format_paper(idx: int, paper: Paper, brief: bool = False) -> list[str]:
    lines = [f"**{idx}. {paper.title}**", ""]
    lines.append(f"- Authors: {paper.authors} | Year: {paper.year} | Venue: {paper.venue or 'preprint'}")
    if paper.arxiv_id:
        lines.append(f"- arXiv: [{paper.arxiv_id}](https://arxiv.org/abs/{paper.arxiv_id})")
    lines += _format_llm_info(paper) if paper.llm_verdict else _format_score_info(paper)
    if not brief and paper.abstract:
        lines.append(f"- Abstract: {paper.abstract[:300]}...")
    lines.append("")
    return lines


def _format_llm_info(paper: Paper) -> list[str]:
    if paper.llm_verdict == "error":
        return [f"- Score: {paper.score} | Found by: {paper.source_count} sources"]
    lines = [
        f"- **LLM: {paper.llm_verdict}** ({paper.llm_score}/10) — {paper.llm_relationship}",
        f"- Cite in: {', '.join(paper.llm_sections)}",
        f"- Why: {paper.llm_reasoning}",
    ]
    if paper.llm_differentiation:
        lines.append(f"- Differs: {paper.llm_differentiation}")
    return lines


def _format_score_info(paper: Paper) -> list[str]:
    return [f"- Score: {paper.score} | Found by: {paper.source_count} sources"]


# ── Paper file generation ─────────────────────────────────────────────────

def write_paper_files(papers: list[Paper], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for paper in papers:
        path = _paper_file_path(paper, output_dir)
        if path.exists():
            continue
        path.write_text(_paper_file_content(paper))
        count += 1
    return count


def _paper_file_path(paper: Paper, directory: Path) -> Path:
    tag = (paper.llm_verdict or paper.category or "other").upper().replace("_", "-")
    slug = re.sub(r"[^a-z0-9\s]", "", paper.title.lower())[:60].replace(" ", "_")
    return directory / f"{tag}_{slug}.md"


def _paper_file_content(paper: Paper) -> str:
    arxiv = f"**arXiv:** [{paper.arxiv_id}](https://arxiv.org/abs/{paper.arxiv_id})\n" if paper.arxiv_id else ""
    llm = _paper_file_llm_section(paper) if paper.llm_verdict else ""
    return (f"# {paper.title}\n\n**Authors:** {paper.authors}\n**Year:** {paper.year}\n"
            f"**Venue:** {paper.venue}\n{arxiv}{llm}\n## Abstract\n\n{paper.abstract or 'N/A'}\n")


def _paper_file_llm_section(paper: Paper) -> str:
    return (f"\n## LLM Evaluation\n\n- Verdict: {paper.llm_verdict} ({paper.llm_score}/10)\n"
            f"- {paper.llm_relationship}\n- Cite in: {', '.join(paper.llm_sections)}\n"
            f"- {paper.llm_reasoning}\n- {paper.llm_differentiation}\n")


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="auto-citetion",
                                 description="Find and evaluate papers for your research.")
    ap.add_argument("config", help="Search config JSON file")
    ap.add_argument("-o", "--output", default=".", help="Output directory")
    ap.add_argument("--refs", help="references.md for dedup")
    ap.add_argument("--skip-search", action="store_true")
    ap.add_argument("--skip-si", action="store_true")
    ap.add_argument("--skip-gs", action="store_true", help="Skip Google Scholar")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--min-score", type=float, default=3.0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--expand-top", type=int, default=25)
    ap.add_argument("--max-retries", type=int, default=3,
                    help="Max 429 retries per request before skipping (default: 3)")
    ap.add_argument("--backoff", type=int, nargs="+", default=[15, 30, 60],
                    help="Backoff seconds per retry (default: 15 30 60)")
    return ap.parse_args()


def _apply_retry_settings(args) -> None:
    from auto_citetion import search
    search.MAX_RETRIES = args.max_retries
    search.BACKOFF_SECONDS = args.backoff


def _load_config(args) -> tuple[Path, dict, set[str], dict[str, tuple]]:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())
    known = load_known_titles(Path(args.refs) if args.refs else out / "references.md")
    category_labels = cfg.get("category_labels", {})
    category_labels = {k: tuple(v) for k, v in category_labels.items()}
    return out, cfg, known, category_labels


def main():
    args = parse_args()
    out, cfg, known, category_labels = _load_config(args)
    raw_path = out / "all_candidates.json"

    _apply_retry_settings(args)
    _apply_scoring_config(cfg)
    papers = run_search_stage(args, cfg, known, out, raw_path)
    papers = run_llm_stage(args, papers, cfg, out)
    run_report_stage(papers, out, category_labels)


def _apply_scoring_config(cfg: dict) -> None:
    from auto_citetion import search
    if "scoring" in cfg:
        scoring = cfg["scoring"]
        if "high_keywords" in scoring:
            search.HIGH_KEYWORDS = scoring["high_keywords"]
        if "medium_keywords" in scoring:
            search.MED_KEYWORDS = scoring["medium_keywords"]
        if "low_keywords" in scoring:
            search.LOW_KEYWORDS = scoring["low_keywords"]
        if "categories" in scoring:
            search.CATEGORIES = scoring["categories"]


def run_search_stage(args, cfg, known, out, raw_path) -> list[Paper]:
    if args.skip_search:
        data = json.loads(raw_path.read_text())
        return filter_novel([Paper.from_dict(d) for d in data], known, args.min_score)

    print("=" * 50, "\nSTAGE 1: Search\n" + "=" * 50, file=sys.stderr)
    pool = PaperPool()
    cookie = load_cookie(out)
    run_initial_search(pool, cfg, cookie, args.skip_si, args.skip_gs)
    print(f"\nInitial pool: {pool.size}", file=sys.stderr)

    depth = 1 if args.fast else args.depth
    if depth > 0:
        run_recursive_expansion(pool, known, args.min_score, cookie, args.skip_gs,
                                depth, args.expand_top)

    papers = pool.all()
    score_and_categorize(papers)
    papers = filter_novel(papers, known, args.min_score)
    raw_path.write_text(json.dumps([p.to_dict() for p in papers], indent=2))
    print(f"\nFinal: {len(papers)} papers", file=sys.stderr)
    return papers


def run_llm_stage(args, papers, cfg, out) -> list[Paper]:
    if args.skip_llm:
        return papers
    print("\n" + "=" * 50, "\nSTAGE 2: LLM Evaluation\n" + "=" * 50, file=sys.stderr)
    top_n = 30 if args.fast else args.top
    papers = run_llm_evaluation(papers, cfg, args.model, top_n)
    (out / "llm_results.json").write_text(json.dumps([p.to_dict() for p in papers], indent=2))
    return papers


def run_report_stage(papers, out, category_labels) -> None:
    print("\n" + "=" * 50, "\nSTAGE 3: Report\n" + "=" * 50, file=sys.stderr)
    report_path = out / "final_reading_list.md"
    report_path.write_text(generate_report(papers, category_labels))
    n = write_paper_files(papers, out / "papers")
    print(f"\nDone! Report: {report_path} | {n} paper files", file=sys.stderr)


if __name__ == "__main__":
    main()
