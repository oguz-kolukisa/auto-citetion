"""Paper data model and pool for deduplication."""

from __future__ import annotations

from dataclasses import dataclass, field


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
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "arxiv_id": self.arxiv_id,
            "citation_count": self.citation_count,
            "abstract": self.abstract,
            "score": self.score,
            "category": self.category,
            "sources": self.sources,
            "source_count": self.source_count,
            "llm_verdict": self.llm_verdict,
            "llm_score": self.llm_score,
            "llm_relationship": self.llm_relationship,
            "llm_sections": self.llm_sections,
            "llm_reasoning": self.llm_reasoning,
            "llm_differentiation": self.llm_differentiation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Paper:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PaperPool:
    """Deduplicating paper collection."""

    def __init__(self):
        self._papers: dict[str, Paper] = {}

    def add(self, paper: Paper, source: str) -> bool:
        key = paper.title.lower().strip()
        if not key:
            return False
        if key in self._papers:
            self._papers[key].sources.append(source)
            self._papers[key].source_count += 1
            return False
        paper.sources = [source]
        paper.source_count = 1
        self._papers[key] = paper
        return True

    def add_many(self, papers: list[Paper], source: str) -> int:
        return sum(1 for p in papers if self.add(p, source))

    @property
    def size(self) -> int:
        return len(self._papers)

    def all(self) -> list[Paper]:
        return list(self._papers.values())
