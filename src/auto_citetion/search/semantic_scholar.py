"""Semantic Scholar API search: keyword search, citation chains, author tracking."""

from __future__ import annotations

import json
import sys
import time
from urllib.parse import quote

from ..paper import Paper, PaperPool
from .http import http_get

API_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,externalIds,title,year,venue,citationCount,abstract,authors,fieldsOfStudy"
DELAY = 0.4


def _parse(data: dict) -> Paper | None:
    if not data or not data.get("title"):
        return None
    ext = data.get("externalIds") or {}
    authors = data.get("authors") or []
    names = ", ".join(a.get("name", "") for a in authors[:4])
    if len(authors) > 4:
        names += " et al."
    return Paper(
        title=data["title"],
        authors=names,
        year=str(data.get("year") or ""),
        venue=data.get("venue") or "",
        arxiv_id=ext.get("ArXiv", ""),
        citation_count=data.get("citationCount") or 0,
        abstract=data.get("abstract") or "",
    )


def _get(path: str) -> dict | None:
    raw = http_get(f"{API_BASE}{path}")
    return json.loads(raw.decode()) if raw else None


class SemanticScholarSearch:

    def __init__(self, delay: float = DELAY):
        self.delay = delay

    def keyword_search(self, pool: PaperPool, queries: list[str]) -> None:
        print("\n[Semantic Scholar] Keyword search", file=sys.stderr)
        for i, q in enumerate(queries):
            print(f"  [{i+1}/{len(queries)}] {q[:60]}", file=sys.stderr)
            encoded = quote(q)
            data = _get(f"/paper/search?query={encoded}&limit=20&fields={FIELDS}")
            if data and "data" in data:
                papers = [p for d in data["data"] if (p := _parse(d))]
                added = pool.add_many(papers, f"ss_keyword:{i}")
                print(f"    {len(papers)} results, {added} new", file=sys.stderr)
            time.sleep(self.delay)

    def citation_chains(self, pool: PaperPool, arxiv_ids: list[str]) -> None:
        print("\n[Semantic Scholar] Citation chains", file=sys.stderr)
        for i, aid in enumerate(arxiv_ids):
            print(f"  [{i+1}/{len(arxiv_ids)}] {aid}", file=sys.stderr)
            for direction in ["citations", "references"]:
                data = _get(f"/paper/ArXiv:{aid}/{direction}?limit=200&fields={FIELDS}")
                if data and "data" in data:
                    key_field = "citingPaper" if direction == "citations" else "citedPaper"
                    papers = [
                        p for item in data["data"]
                        if (d := item.get(key_field)) and (p := _parse(d))
                    ]
                    added = pool.add_many(papers, f"ss_{direction}:{aid}")
                    print(f"    {direction}: {len(papers)}, {added} new", file=sys.stderr)
                time.sleep(self.delay)

    def author_tracking(self, pool: PaperPool, authors: list[str]) -> None:
        print("\n[Semantic Scholar] Author tracking", file=sys.stderr)
        for i, author in enumerate(authors):
            print(f"  [{i+1}/{len(authors)}] {author}", file=sys.stderr)
            encoded = quote(author)
            data = _get(f"/author/search?query={encoded}&limit=1")
            if not data or not data.get("data"):
                time.sleep(self.delay)
                continue
            author_id = data["data"][0].get("authorId")
            if not author_id:
                time.sleep(self.delay)
                continue
            papers_data = _get(f"/author/{author_id}/papers?limit=50&fields={FIELDS}")
            if papers_data and "data" in papers_data:
                papers = [p for d in papers_data["data"] if (p := _parse(d))]
                added = pool.add_many(papers, f"author:{author}")
                print(f"    {len(papers)} papers, {added} new", file=sys.stderr)
            time.sleep(self.delay)
