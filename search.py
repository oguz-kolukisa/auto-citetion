"""Paper search across Semantic Scholar, Scholar Inbox, arXiv, OpenAlex, and DBLP.

Each API runs in its own thread with its own rate limiter.
Disk cache avoids re-fetching on reruns.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

# ── Constants ─────────────────────────────────────────────────────────────

SS_BASE = "https://api.semanticscholar.org/graph/v1"
SS_FIELDS = "paperId,externalIds,title,year,venue,citationCount,abstract,authors"
SI_BASE = "https://api.scholar-inbox.com/api"
OA_BASE = "https://api.openalex.org"
DBLP_BASE = "https://dblp.org/search/publ/api"
ARXIV_BASE = "http://export.arxiv.org/api/query"
ARXIV_NS = "{http://www.w3.org/2005/Atom}"

CACHE_DIR = Path(os.environ.get("AUTOCITE_CACHE", Path.home() / ".cache" / "auto-citetion"))
CACHE_TTL = 86400

TOP_VENUES = ["neurips", "icml", "iclr", "cvpr", "eccv", "iccv"]

HIGH_KEYWORDS = [
    "spurious correlation", "shortcut learning", "counterfactual explanation",
    "counterfactual image", "bias discovery", "feature discovery", "model diagnosis",
    "vision-language model", "attention map", "grad-cam", "score-cam",
    "counterfactual generation", "semantic feature", "concept discovery",
]
MED_KEYWORDS = [
    "explainability", "interpretability", "debiasing", "group robustness",
    "diffusion", "image editing", "saliency", "concept-based", "attribution",
    "vlm", "multimodal",
]
LOW_KEYWORDS = [
    "robustness", "distribution shift", "imagenet", "augmentation", "causal",
    "classifier",
]
CATEGORIES = {
    "similar_method": ["counterfactual", "feature discovery", "bias discovery",
                       "model diagnosis", "attention map"],
    "counterfactual_xai": ["counterfactual explanation", "counterfactual image",
                           "counterfactual generation"],
    "shortcut_spurious": ["spurious correlation", "shortcut learning",
                          "group robustness", "debiasing"],
    "vlm_multimodal": ["vision-language", "vlm", "multimodal", "clip",
                       "large language model"],
    "diffusion_editing": ["diffusion", "image editing", "text-guided",
                          "generative model"],
    "explainability": ["explainability", "interpretability", "attribution",
                       "grad-cam", "score-cam", "saliency"],
    "augmentation": ["augmentation", "data augmentation", "synthetic data"],
}


# ── Paper ─────────────────────────────────────────────────────────────────

@dataclass
class Paper:
    title: str = ""
    authors: str = ""
    year: str = ""
    venue: str = ""
    arxiv_id: str = ""
    citation_count: int = 0
    abstract: str = ""
    score: float = 0.0
    category: str = ""
    sources: list[str] = field(default_factory=list)
    source_count: int = 0
    llm_verdict: str = ""
    llm_score: int = 0
    llm_relationship: str = ""
    llm_sections: list[str] = field(default_factory=list)
    llm_reasoning: str = ""
    llm_differentiation: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> Paper:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Paper pool ────────────────────────────────────────────────────────────

class PaperPool:
    """Thread-safe collection that deduplicates by arXiv ID and fuzzy title."""

    def __init__(self):
        self._papers: dict[str, Paper] = {}
        self._arxiv_index: dict[str, str] = {}
        self._lock = threading.Lock()

    def add(self, paper: Paper, source: str) -> bool:
        key = paper.title.lower().strip()
        if not key:
            return False
        with self._lock:
            existing_key = self._find_duplicate(paper, key)
            if existing_key:
                self._merge_into_existing(existing_key, paper, source)
                return False
            self._insert_new(key, paper, source)
            return True

    def add_many(self, papers: list[Paper], source: str) -> int:
        return sum(1 for p in papers if self.add(p, source))

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._papers)

    def all(self) -> list[Paper]:
        with self._lock:
            return list(self._papers.values())

    def _find_duplicate(self, paper: Paper, key: str) -> str | None:
        if paper.arxiv_id and paper.arxiv_id in self._arxiv_index:
            return self._arxiv_index[paper.arxiv_id]
        if key in self._papers:
            return key
        return self._find_fuzzy_match(key)

    def _find_fuzzy_match(self, key: str) -> str | None:
        words = set(key.split())
        for existing_key in self._papers:
            overlap = len(words & set(existing_key.split()))
            union = len(words | set(existing_key.split()))
            if overlap / max(union, 1) > 0.7:
                return existing_key
        return None

    def _merge_into_existing(self, key: str, paper: Paper, source: str) -> None:
        existing = self._papers[key]
        existing.sources.append(source)
        existing.source_count += 1
        if paper.arxiv_id and not existing.arxiv_id:
            existing.arxiv_id = paper.arxiv_id
            self._arxiv_index[paper.arxiv_id] = key
        if paper.abstract and not existing.abstract:
            existing.abstract = paper.abstract

    def _insert_new(self, key: str, paper: Paper, source: str) -> None:
        paper.sources = [source]
        paper.source_count = 1
        self._papers[key] = paper
        if paper.arxiv_id:
            self._arxiv_index[paper.arxiv_id] = key


# ── Rate limiter ──────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, interval: float):
        self._interval = interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            delay = self._interval - (time.monotonic() - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


_limiters = {
    "ss": RateLimiter(1.0),
    "si": RateLimiter(1.5),
    "arxiv": RateLimiter(3.0),
    "oalex": RateLimiter(0.2),
    "dblp": RateLimiter(1.0),
}


# ── Disk cache ────────────────────────────────────────────────────────────

def _cache_path(url: str, body: str = "") -> Path:
    digest = hashlib.md5((url + body).encode()).hexdigest()
    return CACHE_DIR / digest


def _read_cache(url: str, body: str = "") -> bytes | None:
    path = _cache_path(url, body)
    if path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL:
        return path.read_bytes()
    return None


def _write_cache(url: str, data: bytes, body: str = "") -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(url, body).write_bytes(data)


# ── HTTP ──────────────────────────────────────────────────────────────────

def _build_request(url: str, headers: dict | None, body: bytes | None) -> urllib.request.Request:
    req = urllib.request.Request(url, data=body, method="POST" if body else "GET")
    req.add_header("User-Agent", "auto-citetion/1.0")
    if body:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    return req


def _fetch(url: str, headers: dict | None = None, body: bytes | None = None,
           limiter: str | None = None) -> bytes | None:
    body_str = body.decode() if body else ""
    cached = _read_cache(url, body_str)
    if cached:
        return cached
    if limiter:
        _limiters[limiter].wait()
    try:
        req = _build_request(url, headers, body)
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
        _write_cache(url, data, body_str)
        return data
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f"    429 backoff 15s…", file=sys.stderr)
            time.sleep(15)
            return _fetch(url, headers, body, limiter)
        return None
    except Exception:
        return None


def _get_json(url: str, limiter: str, headers: dict | None = None) -> dict | None:
    raw = _fetch(url, headers=headers, limiter=limiter)
    if not raw:
        return None
    return json.loads(raw.decode())


def _post_json(url: str, data: dict, limiter: str, headers: dict | None = None) -> dict | None:
    raw = _fetch(url, headers=headers, body=json.dumps(data).encode(), limiter=limiter)
    if not raw:
        return None
    return json.loads(raw.decode())


# ── Author formatting ─────────────────────────────────────────────────────

def _format_authors(names: list[str], limit: int = 4) -> str:
    result = ", ".join(names[:limit])
    if len(names) > limit:
        result += " et al."
    return result


# ── Parsers ───────────────────────────────────────────────────────────────

def _parse_ss(d: dict) -> Paper | None:
    if not d or not d.get("title"):
        return None
    ext = d.get("externalIds") or {}
    names = [a.get("name", "") for a in (d.get("authors") or [])]
    return Paper(
        title=d["title"], authors=_format_authors(names),
        year=str(d.get("year") or ""), venue=d.get("venue") or "",
        arxiv_id=ext.get("ArXiv", ""),
        citation_count=d.get("citationCount") or 0,
        abstract=d.get("abstract") or "",
    )


def _parse_si(d: dict) -> Paper | None:
    if not d or not d.get("title"):
        return None
    year = d.get("publication_date", d.get("year", ""))
    if isinstance(year, str) and len(year) >= 4:
        year = year[:4]
    return Paper(
        title=d["title"], authors=d.get("authors", ""),
        year=str(year), venue=d.get("venue", ""),
        arxiv_id=d.get("arxiv_id", ""), abstract=d.get("abstract", ""),
    )


def _parse_arxiv_entry(entry) -> Paper | None:
    title = entry.findtext(f"{ARXIV_NS}title", "").replace("\n", " ").strip()
    if not title:
        return None
    names = [a.findtext(f"{ARXIV_NS}name", "") for a in entry.findall(f"{ARXIV_NS}author")]
    arxiv_id = _extract_arxiv_id(entry)
    return Paper(
        title=title, authors=_format_authors(names),
        year=entry.findtext(f"{ARXIV_NS}published", "")[:4],
        venue="arXiv", arxiv_id=arxiv_id,
        abstract=entry.findtext(f"{ARXIV_NS}summary", "").replace("\n", " ").strip(),
    )


def _extract_arxiv_id(entry) -> str:
    for link in entry.findall(f"{ARXIV_NS}link"):
        href = link.get("href", "")
        if "arxiv.org/abs/" in href:
            return href.split("/abs/")[-1].split("v")[0]
    return ""


def _parse_oalex(d: dict) -> Paper | None:
    if not d or not d.get("title"):
        return None
    names = [a.get("author", {}).get("display_name", "")
             for a in (d.get("authorships") or [])]
    return Paper(
        title=d["title"], authors=_format_authors(names),
        year=str(d.get("publication_year") or ""),
        venue=_extract_oalex_venue(d),
        arxiv_id=_extract_oalex_arxiv(d),
        citation_count=d.get("cited_by_count") or 0,
        abstract=_reconstruct_oalex_abstract(d),
    )


def _extract_oalex_venue(d: dict) -> str:
    src = (d.get("primary_location") or {}).get("source") or {}
    return src.get("display_name", "")


def _extract_oalex_arxiv(d: dict) -> str:
    for loc in d.get("locations") or []:
        url = loc.get("landing_page_url") or ""
        if "arxiv.org/abs/" in url:
            return url.split("/abs/")[-1].split("v")[0]
    return ""


def _reconstruct_oalex_abstract(d: dict) -> str:
    inv_index = d.get("abstract_inverted_index")
    if not inv_index:
        return ""
    words: dict[int, str] = {}
    for word, positions in inv_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


def _parse_dblp(hit: dict) -> Paper | None:
    info = hit.get("info", {})
    title = info.get("title", "")
    if not title:
        return None
    raw_authors = info.get("authors", {}).get("author", [])
    if isinstance(raw_authors, dict):
        raw_authors = [raw_authors]
    names = [(a.get("text", a) if isinstance(a, dict) else str(a))
             for a in raw_authors]
    return Paper(
        title=title.rstrip("."), authors=_format_authors(names),
        year=str(info.get("year", "")), venue=info.get("venue", ""),
    )


# ── Search jobs ───────────────────────────────────────────────────────────
# Each returns (list[Paper], source_tag).

def _si_headers(cookie: str) -> dict:
    return {"Cookie": f"session={cookie}",
            "Origin": "https://www.scholar-inbox.com",
            "Referer": "https://www.scholar-inbox.com/"}

# Semantic Scholar

def job_ss_keyword(query: str, idx: int) -> tuple[list[Paper], str]:
    data = _get_json(f"{SS_BASE}/paper/search?query={quote(query)}&limit=20&fields={SS_FIELDS}", "ss")
    papers = [p for d in (data or {}).get("data", []) if (p := _parse_ss(d))]
    return papers, f"ss_kw:{idx}"


def job_ss_citations(arxiv_id: str) -> tuple[list[Paper], str]:
    papers = []
    for direction in ["citations", "references"]:
        data = _get_json(f"{SS_BASE}/paper/ArXiv:{arxiv_id}/{direction}?limit=200&fields={SS_FIELDS}", "ss")
        field_name = "citingPaper" if direction == "citations" else "citedPaper"
        for item in (data or {}).get("data", []):
            if (d := item.get(field_name)) and (p := _parse_ss(d)):
                papers.append(p)
    return papers, f"ss_cite:{arxiv_id}"


def job_ss_author(name: str) -> tuple[list[Paper], str]:
    tag = f"author:{name}"
    data = _get_json(f"{SS_BASE}/author/search?query={quote(name)}&limit=1", "ss")
    author_id = _extract_ss_author_id(data)
    if not author_id:
        return [], tag
    data = _get_json(f"{SS_BASE}/author/{author_id}/papers?limit=50&fields={SS_FIELDS}", "ss")
    papers = [p for d in (data or {}).get("data", []) if (p := _parse_ss(d))]
    return papers, tag


def _extract_ss_author_id(data: dict | None) -> str | None:
    if not data or not data.get("data"):
        return None
    return data["data"][0].get("authorId")

# Scholar Inbox

def job_si_semantic(query: str, idx: int, cookie: str, pages: int = 3) -> tuple[list[Paper], str]:
    papers = []
    for page in range(pages):
        data = _post_json(f"{SI_BASE}/semantic-search",
                          {"text_input": query, "embedding": None, "p": page},
                          "si", _si_headers(cookie))
        if not data or not data.get("papers"):
            break
        papers.extend(p for d in data["papers"] if (p := _parse_si(d)))
    return papers, f"si_sem:{idx}"


def job_si_similar(paper_id: int, cookie: str) -> tuple[list[Paper], str]:
    data = _get_json(f"{SI_BASE}/get_similar_papers?paper_id={paper_id}",
                     "si", _si_headers(cookie))
    items = (data or {}).get("similar_papers", (data or {}).get("papers", []))
    papers = [p for d in items if (p := _parse_si(d))]
    return papers, f"si_sim:{paper_id}"


def job_si_detail(paper_id: int, cookie: str) -> tuple[list[Paper], str]:
    data = _get_json(f"{SI_BASE}/paper/{paper_id}", "si", _si_headers(cookie))
    papers = []
    for key in ["references", "cited_by", "similar_papers"]:
        papers.extend(p for d in (data or {}).get(key, []) if (p := _parse_si(d)))
    return papers, f"si_det:{paper_id}"


def si_collect_ids(query: str, cookie: str, limit: int = 30) -> list[int]:
    data = _post_json(f"{SI_BASE}/semantic-search",
                      {"text_input": query, "embedding": None, "p": 0},
                      "si", _si_headers(cookie))
    if not data or "papers" not in data:
        return []
    return [pid for d in data["papers"][:limit]
            if (pid := d.get("id", d.get("paper_id")))]

# arXiv

def job_arxiv(query: str, idx: int) -> tuple[list[Paper], str]:
    url = f"{ARXIV_BASE}?search_query={quote(query)}&max_results=30&sortBy=relevance"
    raw = _fetch(url, limiter="arxiv")
    if not raw:
        return [], f"arxiv:{idx}"
    try:
        root = ET.fromstring(raw.decode())
        papers = [p for e in root.findall(f"{ARXIV_NS}entry")
                  if (p := _parse_arxiv_entry(e))]
        return papers, f"arxiv:{idx}"
    except Exception:
        return [], f"arxiv:{idx}"

# OpenAlex

def job_oalex_search(query: str, idx: int) -> tuple[list[Paper], str]:
    url = f"{OA_BASE}/works?search={quote(query)}&per_page=25&sort=relevance_score:desc"
    data = _get_json(url, "oalex")
    papers = [p for d in (data or {}).get("results", []) if (p := _parse_oalex(d))]
    return papers, f"oalex:{idx}"


def job_oalex_cited_by(arxiv_id: str) -> tuple[list[Paper], str]:
    url = (f"{OA_BASE}/works?filter=cites:https://arxiv.org/abs/{arxiv_id}"
           f"&per_page=50&sort=cited_by_count:desc")
    data = _get_json(url, "oalex")
    papers = [p for d in (data or {}).get("results", []) if (p := _parse_oalex(d))]
    return papers, f"oalex_cite:{arxiv_id}"

# DBLP

def job_dblp_search(query: str, idx: int) -> tuple[list[Paper], str]:
    data = _get_json(f"{DBLP_BASE}?q={quote(query)}&format=json&h=30", "dblp")
    hits = (data or {}).get("result", {}).get("hits", {}).get("hit", [])
    papers = [p for h in hits if (p := _parse_dblp(h))]
    return papers, f"dblp:{idx}"


def job_dblp_venue(venue: str, year: int) -> tuple[list[Paper], str]:
    data = _get_json(f"{DBLP_BASE}?q=venue:{quote(venue)}+year:{year}&format=json&h=100", "dblp")
    hits = (data or {}).get("result", {}).get("hits", {}).get("hit", [])
    papers = [p for h in hits if (p := _parse_dblp(h))]
    return papers, f"dblp_venue:{venue}:{year}"


# ── Scheduler ─────────────────────────────────────────────────────────────

def run_parallel(**api_jobs: list) -> None:
    """Run job lists across APIs in parallel. Each kwarg is an API name
    mapped to a list of callables returning (list[Paper], tag)."""
    return _parallel_runner

# Actually, let's keep it simple:

def run_api_threads(pool: PaperPool, api_jobs: dict[str, list]) -> None:
    """Run one thread per API. Jobs within an API run sequentially
    (respecting rate limits), but different APIs overlap."""
    all_jobs = [(label, jobs) for label, jobs in api_jobs.items() if jobs]
    total = sum(len(jobs) for _, jobs in all_jobs)
    print(f"  {total} jobs across {len(all_jobs)} APIs", file=sys.stderr)

    counter = _Counter()

    threads = [
        threading.Thread(
            target=_run_job_list,
            args=(pool, jobs, label, counter, total),
            daemon=True,
        )
        for label, jobs in all_jobs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


class _Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self) -> int:
        with self.lock:
            self.value += 1
            return self.value


def _run_job_list(pool: PaperPool, jobs: list, label: str,
                  counter: _Counter, total: int) -> None:
    for job in jobs:
        try:
            papers, source = job()
            added = pool.add_many(papers, source)
            n = counter.increment()
            if papers:
                print(f"  [{n}/{total}] {source}: {len(papers)} found, "
                      f"{added} new (pool: {pool.size})", file=sys.stderr)
        except Exception as e:
            n = counter.increment()
            print(f"  [{n}/{total}] {label} error: {e}", file=sys.stderr)


# ── Scoring ───────────────────────────────────────────────────────────────

def score_paper(paper: Paper) -> float:
    text = f"{paper.title} {paper.abstract}".lower()
    s = _keyword_score(text)
    s += _citation_bonus(paper.citation_count)
    s += _recency_bonus(paper.year)
    s += _venue_bonus(paper.venue)
    s += _cross_reference_bonus(paper)
    return round(max(s, 0), 1)


def _keyword_score(text: str) -> float:
    s = sum(3.0 for kw in HIGH_KEYWORDS if kw in text)
    s += sum(2.0 for kw in MED_KEYWORDS if kw in text)
    s += sum(1.0 for kw in LOW_KEYWORDS if kw in text)
    return s


def _citation_bonus(count: int) -> float:
    return min(count / 40, 8.0)


def _recency_bonus(year: str) -> float:
    if not year.isdigit():
        return 0
    y = int(year)
    bonus = 0
    if y >= 2023:
        bonus += 3
    if y >= 2024:
        bonus += 2
    if y >= 2025:
        bonus += 2
    return bonus


def _venue_bonus(venue: str) -> float:
    if venue and any(v in venue.lower() for v in TOP_VENUES):
        return 4.0
    return 0.0


def _cross_reference_bonus(paper: Paper) -> float:
    source_hits = paper.source_count * 2
    strategy_types = len(set(src.split(":")[0] for src in paper.sources))
    return source_hits + strategy_types * 3


def categorize_paper(paper: Paper) -> str:
    text = f"{paper.title} {paper.abstract}".lower()
    best, best_n = "other", 0
    for cat, keywords in CATEGORIES.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > best_n:
            best, best_n = cat, hits
    return best


def score_and_categorize(papers: list[Paper]) -> None:
    for p in papers:
        p.score = score_paper(p)
        p.category = categorize_paper(p)
