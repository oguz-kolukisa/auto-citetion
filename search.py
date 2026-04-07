"""Paper search across Semantic Scholar, Scholar Inbox, arXiv, OpenAlex, and DBLP.

Uses a concurrent job scheduler with per-API rate limiters so that
when one API is rate-limited, the others keep running.
Includes disk caching to avoid re-fetching on reruns.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

# ── Disk cache ────────────────────────────────────────────────────────────

CACHE_DIR = Path(os.environ.get("AUTOCITE_CACHE", Path.home() / ".cache" / "auto-citetion"))


def _cache_key(url: str, body: str = "") -> str:
    return hashlib.md5((url + body).encode()).hexdigest()


def _cache_get(url: str, body: str = "") -> bytes | None:
    path = CACHE_DIR / _cache_key(url, body)
    if path.exists() and (time.time() - path.stat().st_mtime) < 86400:  # 24h TTL
        return path.read_bytes()
    return None


def _cache_set(url: str, data: bytes, body: str = "") -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / _cache_key(url, body)).write_bytes(data)


# ── Data model ────────────────────────────────────────────────────────────

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


class PaperPool:
    """Thread-safe deduplicating paper collection.
    Deduplicates by title (fuzzy) and arXiv ID."""

    def __init__(self):
        self._papers: dict[str, Paper] = {}
        self._arxiv_index: dict[str, str] = {}  # arxiv_id -> title key
        self._lock = threading.Lock()

    def _find_existing(self, p: Paper) -> str | None:
        """Check if paper already exists by arXiv ID or fuzzy title."""
        if p.arxiv_id and p.arxiv_id in self._arxiv_index:
            return self._arxiv_index[p.arxiv_id]
        key = p.title.lower().strip()
        if key in self._papers:
            return key
        # Fuzzy title match
        title_words = set(key.split())
        for existing_key in self._papers:
            existing_words = set(existing_key.split())
            overlap = len(title_words & existing_words) / max(len(title_words | existing_words), 1)
            if overlap > 0.7:
                return existing_key
        return None

    def add(self, p: Paper, source: str) -> bool:
        key = p.title.lower().strip()
        if not key:
            return False
        with self._lock:
            existing = self._find_existing(p)
            if existing:
                self._papers[existing].sources.append(source)
                self._papers[existing].source_count += 1
                # Merge arXiv ID if the existing one doesn't have it
                if p.arxiv_id and not self._papers[existing].arxiv_id:
                    self._papers[existing].arxiv_id = p.arxiv_id
                    self._arxiv_index[p.arxiv_id] = existing
                # Merge abstract if existing one is empty
                if p.abstract and not self._papers[existing].abstract:
                    self._papers[existing].abstract = p.abstract
                return False
            p.sources = [source]
            p.source_count = 1
            self._papers[key] = p
            if p.arxiv_id:
                self._arxiv_index[p.arxiv_id] = key
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


# ── Per-API rate limiters ─────────────────────────────────────────────────

class RateLimiter:
    """Token-bucket rate limiter for a single API."""

    def __init__(self, min_interval: float):
        self._interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last = time.monotonic()


# One limiter per API — they don't block each other
_ss_limiter = RateLimiter(1.0)
_si_limiter = RateLimiter(1.5)
_arxiv_limiter = RateLimiter(3.0)
_oalex_limiter = RateLimiter(0.2)   # OpenAlex is generous: 10 req/s
_dblp_limiter = RateLimiter(1.0)


# ── HTTP helpers ──────────────────────────────────────────────────────────

def _get(url: str, headers: dict | None = None, limiter: RateLimiter | None = None) -> bytes | None:
    cached = _cache_get(url)
    if cached:
        return cached
    if limiter:
        limiter.wait()
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "auto-citetion/1.0")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
            _cache_set(url, data)
            return data
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f"    429 on {url[:50]}… backoff 15s", file=sys.stderr)
            time.sleep(15)
            return _get(url, headers, limiter)
        return None
    except Exception:
        return None


def _post(url: str, data: dict, headers: dict | None = None,
          limiter: RateLimiter | None = None) -> dict | None:
    body_str = json.dumps(data)
    cached = _cache_get(url, body_str)
    if cached:
        return json.loads(cached.decode())
    if limiter:
        limiter.wait()
    try:
        body = body_str.encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "auto-citetion/1.0")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=30) as r:
            resp_data = r.read()
            _cache_set(url, resp_data, body_str)
            return json.loads(resp_data.decode())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            time.sleep(15)
            return _post(url, data, headers, limiter)
        return None
    except Exception:
        return None


# ── Parsers ───────────────────────────────────────────────────────────────

def _parse_ss(d: dict) -> Paper | None:
    if not d or not d.get("title"):
        return None
    ext = d.get("externalIds") or {}
    au = d.get("authors") or []
    names = ", ".join(a.get("name", "") for a in au[:4])
    if len(au) > 4:
        names += " et al."
    return Paper(title=d["title"], authors=names, year=str(d.get("year") or ""),
                 venue=d.get("venue") or "", arxiv_id=ext.get("ArXiv", ""),
                 citation_count=d.get("citationCount") or 0,
                 abstract=d.get("abstract") or "")


def _parse_si(d: dict) -> Paper | None:
    if not d or not d.get("title"):
        return None
    y = d.get("publication_date", d.get("year", ""))
    if isinstance(y, str) and len(y) >= 4:
        y = y[:4]
    return Paper(title=d["title"], authors=d.get("authors", ""),
                 year=str(y), venue=d.get("venue", ""),
                 arxiv_id=d.get("arxiv_id", ""), abstract=d.get("abstract", ""))


# ── Job definitions (each returns papers to add) ──────────────────────────

# Semantic Scholar jobs

SS = "https://api.semanticscholar.org/graph/v1"
SS_F = "paperId,externalIds,title,year,venue,citationCount,abstract,authors"


def _job_ss_keyword(query: str, idx: int) -> tuple[list[Paper], str]:
    raw = _get(f"{SS}/paper/search?query={quote(query)}&limit=20&fields={SS_F}", limiter=_ss_limiter)
    if not raw:
        return [], f"ss_kw:{idx}"
    data = json.loads(raw.decode())
    papers = [p for d in data.get("data", []) if (p := _parse_ss(d))]
    return papers, f"ss_kw:{idx}"


def _job_ss_citations(arxiv_id: str) -> tuple[list[Paper], str]:
    all_papers = []
    for direction in ["citations", "references"]:
        raw = _get(f"{SS}/paper/ArXiv:{arxiv_id}/{direction}?limit=200&fields={SS_F}", limiter=_ss_limiter)
        if raw:
            data = json.loads(raw.decode())
            fld = "citingPaper" if direction == "citations" else "citedPaper"
            papers = [p for it in data.get("data", []) if (x := it.get(fld)) and (p := _parse_ss(x))]
            all_papers.extend(papers)
    return all_papers, f"ss_cite:{arxiv_id}"


def _job_ss_author(name: str) -> tuple[list[Paper], str]:
    raw = _get(f"{SS}/author/search?query={quote(name)}&limit=1", limiter=_ss_limiter)
    if not raw:
        return [], f"author:{name}"
    data = json.loads(raw.decode())
    if not data.get("data"):
        return [], f"author:{name}"
    aid = data["data"][0].get("authorId")
    if not aid:
        return [], f"author:{name}"
    raw2 = _get(f"{SS}/author/{aid}/papers?limit=50&fields={SS_F}", limiter=_ss_limiter)
    if not raw2:
        return [], f"author:{name}"
    data2 = json.loads(raw2.decode())
    papers = [p for d in data2.get("data", []) if (p := _parse_ss(d))]
    return papers, f"author:{name}"


# Scholar Inbox jobs

SI = "https://api.scholar-inbox.com/api"


def _si_h(cookie: str) -> dict:
    return {"Cookie": f"session={cookie}", "Origin": "https://www.scholar-inbox.com",
            "Referer": "https://www.scholar-inbox.com/"}


def _job_si_semantic(query: str, idx: int, cookie: str, pages: int = 3) -> tuple[list[Paper], str]:
    h = _si_h(cookie)
    all_papers = []
    for page in range(pages):
        r = _post(f"{SI}/semantic-search", {"text_input": query, "embedding": None, "p": page},
                  h, limiter=_si_limiter)
        if not r or not r.get("papers"):
            break
        all_papers.extend(p for d in r["papers"] if (p := _parse_si(d)))
    return all_papers, f"si_sem:{idx}"


def _job_si_similar(paper_id: int, cookie: str) -> tuple[list[Paper], str]:
    raw = _get(f"{SI}/get_similar_papers?paper_id={paper_id}", _si_h(cookie), limiter=_si_limiter)
    if not raw:
        return [], f"si_sim:{paper_id}"
    r = json.loads(raw.decode())
    papers = [p for d in r.get("similar_papers", r.get("papers", [])) if (p := _parse_si(d))]
    return papers, f"si_sim:{paper_id}"


def _job_si_detail(paper_id: int, cookie: str) -> tuple[list[Paper], str]:
    raw = _get(f"{SI}/paper/{paper_id}", _si_h(cookie), limiter=_si_limiter)
    if not raw:
        return [], f"si_det:{paper_id}"
    r = json.loads(raw.decode())
    all_papers = []
    for key in ["references", "cited_by", "similar_papers"]:
        all_papers.extend(p for d in r.get(key, []) if (p := _parse_si(d)))
    return all_papers, f"si_det:{paper_id}"


def si_collect_ids(query: str, cookie: str, limit: int = 30) -> list[int]:
    r = _post(f"{SI}/semantic-search", {"text_input": query, "embedding": None, "p": 0},
              _si_h(cookie), limiter=_si_limiter)
    if not r or "papers" not in r:
        return []
    return [pid for d in r["papers"][:limit] if (pid := d.get("id", d.get("paper_id")))]


# arXiv jobs

def _job_arxiv(query: str, idx: int) -> tuple[list[Paper], str]:
    ns = "{http://www.w3.org/2005/Atom}"
    raw = _get(f"http://export.arxiv.org/api/query?search_query={quote(query)}&max_results=30&sortBy=relevance",
               limiter=_arxiv_limiter)
    if not raw:
        return [], f"arxiv:{idx}"
    papers = []
    try:
        root = ET.fromstring(raw.decode())
        for e in root.findall(f"{ns}entry"):
            title = e.findtext(f"{ns}title", "").replace("\n", " ").strip()
            if not title:
                continue
            au = [a.findtext(f"{ns}name", "") for a in e.findall(f"{ns}author")]
            arxid = ""
            for lnk in e.findall(f"{ns}link"):
                href = lnk.get("href", "")
                if "arxiv.org/abs/" in href:
                    arxid = href.split("/abs/")[-1].split("v")[0]
            papers.append(Paper(
                title=title,
                authors=", ".join(au[:4]) + (" et al." if len(au) > 4 else ""),
                year=e.findtext(f"{ns}published", "")[:4], venue="arXiv",
                arxiv_id=arxid,
                abstract=e.findtext(f"{ns}summary", "").replace("\n", " ").strip(),
            ))
    except Exception:
        pass
    return papers, f"arxiv:{idx}"


# ── OpenAlex jobs ─────────────────────────────────────────────────────────

OA = "https://api.openalex.org"


def _parse_oalex(d: dict) -> Paper | None:
    if not d or not d.get("title"):
        return None
    au_list = d.get("authorships") or []
    names = ", ".join(
        a.get("author", {}).get("display_name", "") for a in au_list[:4]
    )
    if len(au_list) > 4:
        names += " et al."
    year = str(d.get("publication_year") or "")
    venue = ""
    loc = d.get("primary_location") or {}
    src = loc.get("source") or {}
    if src.get("display_name"):
        venue = src["display_name"]
    arxiv_id = ""
    ids = d.get("ids") or {}
    if ids.get("openalex"):
        pass  # we want arxiv
    doi = d.get("doi") or ""
    # Try to extract arxiv from locations
    for location in d.get("locations") or []:
        landing = location.get("landing_page_url") or ""
        if "arxiv.org/abs/" in landing:
            arxiv_id = landing.split("/abs/")[-1].split("v")[0]
            break
    abstract = ""
    inv_index = d.get("abstract_inverted_index")
    if inv_index:
        words: dict[int, str] = {}
        for word, positions in inv_index.items():
            for pos in positions:
                words[pos] = word
        abstract = " ".join(words[i] for i in sorted(words))
    return Paper(
        title=d["title"], authors=names, year=year, venue=venue,
        arxiv_id=arxiv_id, citation_count=d.get("cited_by_count") or 0,
        abstract=abstract,
    )


def _job_oalex_search(query: str, idx: int) -> tuple[list[Paper], str]:
    url = f"{OA}/works?search={quote(query)}&per_page=25&sort=relevance_score:desc"
    raw = _get(url, limiter=_oalex_limiter)
    if not raw:
        return [], f"oalex:{idx}"
    data = json.loads(raw.decode())
    papers = [p for d in data.get("results", []) if (p := _parse_oalex(d))]
    return papers, f"oalex:{idx}"


def _job_oalex_cited_by(arxiv_id: str) -> tuple[list[Paper], str]:
    url = f"{OA}/works?filter=cites:https://arxiv.org/abs/{arxiv_id}&per_page=50&sort=cited_by_count:desc"
    raw = _get(url, limiter=_oalex_limiter)
    if not raw:
        return [], f"oalex_cite:{arxiv_id}"
    data = json.loads(raw.decode())
    papers = [p for d in data.get("results", []) if (p := _parse_oalex(d))]
    return papers, f"oalex_cite:{arxiv_id}"


# ── DBLP jobs ─────────────────────────────────────────────────────────────

DBLP = "https://dblp.org/search/publ/api"


def _parse_dblp(hit: dict) -> Paper | None:
    info = hit.get("info", {})
    title = info.get("title", "")
    if not title:
        return None
    authors_raw = info.get("authors", {}).get("author", [])
    if isinstance(authors_raw, dict):
        authors_raw = [authors_raw]
    names = ", ".join(
        (a.get("text", a) if isinstance(a, dict) else str(a)) for a in authors_raw[:4]
    )
    if len(authors_raw) > 4:
        names += " et al."
    return Paper(
        title=title.rstrip("."), authors=names,
        year=str(info.get("year", "")),
        venue=info.get("venue", ""),
        abstract="",  # DBLP doesn't have abstracts
    )


def _job_dblp_search(query: str, idx: int) -> tuple[list[Paper], str]:
    url = f"{DBLP}?q={quote(query)}&format=json&h=30"
    raw = _get(url, limiter=_dblp_limiter)
    if not raw:
        return [], f"dblp:{idx}"
    data = json.loads(raw.decode())
    hits = data.get("result", {}).get("hits", {}).get("hit", [])
    papers = [p for h in hits if (p := _parse_dblp(h))]
    return papers, f"dblp:{idx}"


def _job_dblp_venue(venue: str, year: int) -> tuple[list[Paper], str]:
    """Search for papers from a specific venue+year (e.g. NeurIPS 2024)."""
    url = f"{DBLP}?q=venue:{quote(venue)}+year:{year}&format=json&h=100"
    raw = _get(url, limiter=_dblp_limiter)
    if not raw:
        return [], f"dblp_venue:{venue}:{year}"
    data = json.loads(raw.decode())
    hits = data.get("result", {}).get("hits", {}).get("hit", [])
    papers = [p for h in hits if (p := _parse_dblp(h))]
    return papers, f"dblp_venue:{venue}:{year}"


# ── Job scheduler ─────────────────────────────────────────────────────────

def run_jobs(pool: PaperPool, jobs: list, max_workers: int = 2) -> None:
    """Run search jobs. Each job returns (list[Paper], source_tag).
    Uses a small worker pool — rate limiters serialize per-API."""
    total = len(jobs)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(job): job for job in jobs}
        for future in as_completed(futures):
            done += 1
            try:
                papers, source = future.result()
                added = pool.add_many(papers, source)
                if papers:
                    print(f"  [{done}/{total}] {source}: {len(papers)} found, {added} new "
                          f"(pool: {pool.size})", file=sys.stderr)
            except Exception as e:
                print(f"  [{done}/{total}] error: {e}", file=sys.stderr)


def run_parallel_apis(pool: PaperPool,
                      ss_jobs: list = None, si_jobs: list = None,
                      arxiv_jobs: list = None, oalex_jobs: list = None,
                      dblp_jobs: list = None) -> None:
    """Run jobs across all APIs truly in parallel.
    Each API gets its own thread so rate limiters serialize within
    an API, but different APIs overlap."""
    ss_jobs = ss_jobs or []
    si_jobs = si_jobs or []
    arxiv_jobs = arxiv_jobs or []
    oalex_jobs = oalex_jobs or []
    dblp_jobs = dblp_jobs or []
    total = len(ss_jobs) + len(si_jobs) + len(arxiv_jobs) + len(oalex_jobs) + len(dblp_jobs)
    apis = sum(1 for j in [ss_jobs, si_jobs, arxiv_jobs, oalex_jobs, dblp_jobs] if j)
    print(f"  Running {total} jobs across {apis} APIs in parallel", file=sys.stderr)
    done = [0]
    lock = threading.Lock()

    def _run_batch(jobs: list, label: str) -> None:
        for job in jobs:
            try:
                papers, source = job()
                added = pool.add_many(papers, source)
                with lock:
                    done[0] += 1
                    if papers:
                        print(f"  [{done[0]}/{total}] {source}: {len(papers)} found, "
                              f"{added} new (pool: {pool.size})", file=sys.stderr)
            except Exception as e:
                with lock:
                    done[0] += 1
                    print(f"  [{done[0]}/{total}] {label} error: {e}", file=sys.stderr)

    threads = []
    for jobs, label in [(ss_jobs, "SS"), (si_jobs, "SI"), (arxiv_jobs, "arXiv"),
                        (oalex_jobs, "OpenAlex"), (dblp_jobs, "DBLP")]:
        if jobs:
            threads.append(threading.Thread(target=_run_batch, args=(jobs, label), daemon=True))

    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ── High-level API (used by auto_citetion.py) ────────────────────────────

def ss_keyword(pool: PaperPool, queries: list[str]) -> None:
    print("\n[Semantic Scholar] keyword search", file=sys.stderr)
    jobs = [lambda q=q, i=i: _job_ss_keyword(q, i) for i, q in enumerate(queries)]
    run_jobs(pool, jobs)


def ss_citations(pool: PaperPool, arxiv_ids: list[str]) -> None:
    print("\n[Semantic Scholar] citation chains", file=sys.stderr)
    jobs = [lambda a=a: _job_ss_citations(a) for a in arxiv_ids]
    run_jobs(pool, jobs)


def ss_authors(pool: PaperPool, authors: list[str]) -> None:
    print("\n[Semantic Scholar] author tracking", file=sys.stderr)
    jobs = [lambda n=n: _job_ss_author(n) for n in authors]
    run_jobs(pool, jobs)


def si_semantic(pool: PaperPool, queries: list[str], cookie: str) -> None:
    print("\n[Scholar Inbox] semantic search", file=sys.stderr)
    jobs = [lambda q=q, i=i: _job_si_semantic(q, i, cookie) for i, q in enumerate(queries)]
    run_jobs(pool, jobs)


def si_similar(pool: PaperPool, paper_ids: list[int], cookie: str) -> None:
    print("\n[Scholar Inbox] similar papers", file=sys.stderr)
    jobs = [lambda pid=pid: _job_si_similar(pid, cookie) for pid in paper_ids]
    run_jobs(pool, jobs)


def si_detail(pool: PaperPool, paper_ids: list[int], cookie: str) -> None:
    print("\n[Scholar Inbox] paper refs+cited_by", file=sys.stderr)
    jobs = [lambda pid=pid: _job_si_detail(pid, cookie) for pid in paper_ids]
    run_jobs(pool, jobs)


def arxiv_search(pool: PaperPool, queries: list[str]) -> None:
    print("\n[arXiv] search", file=sys.stderr)
    jobs = [lambda q=q, i=i: _job_arxiv(q, i) for i, q in enumerate(queries)]
    run_jobs(pool, jobs)


def oalex_search(pool: PaperPool, queries: list[str]) -> None:
    print("\n[OpenAlex] search", file=sys.stderr)
    jobs = [lambda q=q, i=i: _job_oalex_search(q, i) for i, q in enumerate(queries)]
    run_jobs(pool, jobs)


def oalex_cited_by(pool: PaperPool, arxiv_ids: list[str]) -> None:
    print("\n[OpenAlex] cited-by chains", file=sys.stderr)
    jobs = [lambda a=a: _job_oalex_cited_by(a) for a in arxiv_ids]
    run_jobs(pool, jobs)


def dblp_search(pool: PaperPool, queries: list[str]) -> None:
    print("\n[DBLP] search", file=sys.stderr)
    jobs = [lambda q=q, i=i: _job_dblp_search(q, i) for i, q in enumerate(queries)]
    run_jobs(pool, jobs)


def dblp_venues(pool: PaperPool, venues: list[tuple[str, int]]) -> None:
    """Search specific venue+year combos, e.g. [("NeurIPS", 2024), ("CVPR", 2025)]."""
    print("\n[DBLP] venue scan", file=sys.stderr)
    jobs = [lambda v=v, y=y: _job_dblp_venue(v, y) for v, y in venues]
    run_jobs(pool, jobs)


# ── Scoring ───────────────────────────────────────────────────────────────

HIGH = ["spurious correlation", "shortcut learning", "counterfactual explanation",
        "counterfactual image", "bias discovery", "feature discovery", "model diagnosis",
        "vision-language model", "attention map", "grad-cam", "score-cam",
        "counterfactual generation", "semantic feature", "concept discovery"]
MED = ["explainability", "interpretability", "debiasing", "group robustness",
       "diffusion", "image editing", "saliency", "concept-based", "attribution",
       "vlm", "multimodal"]
LOW = ["robustness", "distribution shift", "imagenet", "augmentation", "causal", "classifier"]

CATEGORIES = {
    "similar_method": ["counterfactual", "feature discovery", "bias discovery", "model diagnosis", "attention map"],
    "counterfactual_xai": ["counterfactual explanation", "counterfactual image", "counterfactual generation"],
    "shortcut_spurious": ["spurious correlation", "shortcut learning", "group robustness", "debiasing"],
    "vlm_multimodal": ["vision-language", "vlm", "multimodal", "clip", "large language model"],
    "diffusion_editing": ["diffusion", "image editing", "text-guided", "generative model"],
    "explainability": ["explainability", "interpretability", "attribution", "grad-cam", "score-cam", "saliency"],
    "augmentation": ["augmentation", "data augmentation", "synthetic data"],
}


def score_and_categorize(papers: list[Paper]) -> None:
    for p in papers:
        text = f"{p.title} {p.abstract}".lower()
        s = sum(3.0 for kw in HIGH if kw in text)
        s += sum(2.0 for kw in MED if kw in text)
        s += sum(1.0 for kw in LOW if kw in text)
        s += min(p.citation_count / 40, 8.0)
        if p.year.isdigit():
            y = int(p.year)
            if y >= 2023: s += 3
            if y >= 2024: s += 2
            if y >= 2025: s += 2
        if p.venue and any(v in p.venue.lower() for v in ["neurips", "icml", "iclr", "cvpr", "eccv", "iccv"]):
            s += 4
        s += p.source_count * 2
        s += len(set(src.split(":")[0] for src in p.sources)) * 3
        p.score = round(max(s, 0), 1)

        best, best_n = "other", 0
        for cat, kws in CATEGORIES.items():
            n = sum(1 for kw in kws if kw in text)
            if n > best_n:
                best, best_n = cat, n
        p.category = best
