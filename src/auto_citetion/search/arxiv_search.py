"""arXiv API search for recent preprints."""

from __future__ import annotations

import sys
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

from ..paper import Paper, PaperPool
from .http import http_get

API_URL = "http://export.arxiv.org/api/query"
DELAY = 3.0


def _parse_entry(entry, ns: str) -> Paper | None:
    title = entry.findtext(f"{ns}title", "").replace("\n", " ").strip()
    if not title:
        return None
    authors = [a.findtext(f"{ns}name", "") for a in entry.findall(f"{ns}author")]
    summary = entry.findtext(f"{ns}summary", "").replace("\n", " ").strip()
    published = entry.findtext(f"{ns}published", "")[:4]
    arxiv_id = ""
    for link in entry.findall(f"{ns}link"):
        href = link.get("href", "")
        if "arxiv.org/abs/" in href:
            arxiv_id = href.split("/abs/")[-1].split("v")[0]
    return Paper(
        title=title,
        authors=", ".join(authors[:4]) + (" et al." if len(authors) > 4 else ""),
        year=published,
        venue="arXiv",
        arxiv_id=arxiv_id,
        abstract=summary,
    )


class ArxivSearch:

    def __init__(self, delay: float = DELAY):
        self.delay = delay

    def search(self, pool: PaperPool, queries: list[str],
               max_results: int = 30) -> None:
        print("\n[arXiv] Search", file=sys.stderr)
        ns = "{http://www.w3.org/2005/Atom}"
        for i, q in enumerate(queries):
            print(f"  [{i+1}/{len(queries)}] {q[:60]}", file=sys.stderr)
            encoded = quote(q)
            url = f"{API_URL}?search_query={encoded}&start=0&max_results={max_results}&sortBy=relevance"
            raw = http_get(url, timeout=30)
            if not raw:
                time.sleep(self.delay)
                continue
            try:
                root = ET.fromstring(raw.decode())
                entries = root.findall(f"{ns}entry")
                papers = [p for e in entries if (p := _parse_entry(e, ns))]
                added = pool.add_many(papers, f"arxiv:{i}")
                print(f"    {len(papers)} results, {added} new", file=sys.stderr)
            except Exception as e:
                print(f"    Parse error: {e}", file=sys.stderr)
            time.sleep(self.delay)
