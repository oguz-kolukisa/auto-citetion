"""Paper search across Semantic Scholar, Scholar Inbox, and arXiv.

Uses a concurrent job scheduler with per-API rate limiters so that
when one API is rate-limited, the others keep running.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from urllib.parse import quote


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
    """Thread-safe deduplicating paper collection."""

    def __init__(self):
        self._papers: dict[str, Paper] = {}
        self._lock = threading.Lock()

    def add(self, p: Paper, source: str) -> bool:
        key = p.title.lower().strip()
        if not key:
            return False
        with self._lock:
            if key in self._papers:
                self._papers[key].sources.append(source)
                self._papers[key].source_count += 1
                return False
            p.sources = [source]
            p.source_count = 1
            self._papers[key] = p
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
_ss_limiter = RateLimiter(0.4)
_si_limiter = RateLimiter(1.5)
_arxiv_limiter = RateLimiter(3.0)


# ── HTTP helpers ──────────────────────────────────────────────────────────

def _get(url: str, headers: dict | None = None, limiter: RateLimiter | None = None) -> bytes | None:
    if limiter:
        limiter.wait()
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "auto-citetion/1.0")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read()
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
    if limiter:
        limiter.wait()
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "auto-citetion/1.0")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
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


# ── Job scheduler ─────────────────────────────────────────────────────────

def run_jobs(pool: PaperPool, jobs: list, max_workers: int = 6) -> None:
    """Run search jobs concurrently. Each job is a callable returning
    (list[Paper], source_tag). Different APIs run in parallel,
    each respecting its own rate limiter."""
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
