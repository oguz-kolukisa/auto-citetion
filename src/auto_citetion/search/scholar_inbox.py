"""Scholar Inbox API search: semantic search, similar papers, paper detail."""

from __future__ import annotations

import json
import sys
import time

from ..paper import Paper, PaperPool
from .http import http_get, http_post

API_BASE = "https://api.scholar-inbox.com/api"
DELAY = 1.5


def _parse(data: dict) -> Paper | None:
    if not data or not data.get("title"):
        return None
    year = data.get("publication_date", data.get("year", ""))
    if isinstance(year, str) and len(year) >= 4:
        year = year[:4]
    return Paper(
        title=data["title"],
        authors=data.get("authors", ""),
        year=str(year),
        venue=data.get("venue", ""),
        arxiv_id=data.get("arxiv_id", ""),
        abstract=data.get("abstract", ""),
    )


class ScholarInboxSearch:

    def __init__(self, session_cookie: str, delay: float = DELAY):
        self.cookie = session_cookie
        self.delay = delay

    @property
    def _headers(self) -> dict:
        return {
            "Cookie": f"session={self.cookie}",
            "Origin": "https://www.scholar-inbox.com",
            "Referer": "https://www.scholar-inbox.com/",
        }

    def semantic_search(self, pool: PaperPool, queries: list[str],
                        pages_per_query: int = 3) -> None:
        print("\n[Scholar Inbox] Semantic search", file=sys.stderr)
        for i, q in enumerate(queries):
            print(f"  [{i+1}/{len(queries)}] {q[:60]}...", file=sys.stderr)
            for page in range(pages_per_query):
                resp = http_post(
                    f"{API_BASE}/semantic-search",
                    {"text_input": q, "embedding": None, "p": page},
                    self._headers,
                )
                if not resp:
                    break
                papers_data = resp.get("papers", [])
                if not papers_data:
                    break
                papers = [p for d in papers_data if (p := _parse(d))]
                added = pool.add_many(papers, f"si_semantic:{i}")
                print(f"    page {page}: {len(papers)}, {added} new", file=sys.stderr)
                time.sleep(self.delay)

    def similar_papers(self, pool: PaperPool, paper_ids: list[int]) -> None:
        print("\n[Scholar Inbox] Similar papers", file=sys.stderr)
        for i, pid in enumerate(paper_ids):
            print(f"  [{i+1}/{len(paper_ids)}] id={pid}", file=sys.stderr)
            raw = http_get(f"{API_BASE}/get_similar_papers?paper_id={pid}", self._headers)
            if raw:
                resp = json.loads(raw.decode())
                items = resp.get("similar_papers", resp.get("papers", []))
                papers = [p for d in items if (p := _parse(d))]
                added = pool.add_many(papers, f"si_similar:{pid}")
                print(f"    {len(papers)} similar, {added} new", file=sys.stderr)
            time.sleep(self.delay)

    def paper_detail(self, pool: PaperPool, paper_ids: list[int]) -> None:
        print("\n[Scholar Inbox] Paper detail (refs+cited_by)", file=sys.stderr)
        for i, pid in enumerate(paper_ids):
            print(f"  [{i+1}/{len(paper_ids)}] id={pid}", file=sys.stderr)
            raw = http_get(f"{API_BASE}/paper/{pid}", self._headers)
            if not raw:
                time.sleep(self.delay)
                continue
            resp = json.loads(raw.decode())
            for key in ["references", "cited_by", "similar_papers"]:
                items = resp.get(key, [])
                papers = [p for d in items if (p := _parse(d))]
                added = pool.add_many(papers, f"si_{key}:{pid}")
                if papers:
                    print(f"    {key}: {len(papers)}, {added} new", file=sys.stderr)
            time.sleep(self.delay)

    def collect_paper_ids(self, query: str, limit: int = 30) -> list[int]:
        """Search and return Scholar Inbox paper IDs for expansion."""
        resp = http_post(
            f"{API_BASE}/semantic-search",
            {"text_input": query, "embedding": None, "p": 0},
            self._headers,
        )
        if not resp or "papers" not in resp:
            return []
        ids = []
        for pd in resp["papers"][:limit]:
            pid = pd.get("id", pd.get("paper_id"))
            if pid:
                ids.append(pid)
        return ids
