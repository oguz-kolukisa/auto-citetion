# Auto-Citetion

Automated paper discovery and LLM-powered relevance evaluation for academic research. Find papers you should cite, scored and grouped by a local LLM.

## What it does

Given your paper's abstract and key terms, Auto-Citetion:

1. **Searches** across 3 academic APIs (Semantic Scholar, Scholar Inbox, arXiv)
2. **Crawls** citation chains, similar papers, and author profiles
3. **Scores** every paper by keyword relevance and cross-reference count
4. **Evaluates** top candidates with a local LLM (gemma-4-E4B-it by default)
5. **Generates** a final reading list grouped by verdict (must cite / should cite / maybe / skip) and paper section

```
Search (8 strategies) → Score & Filter → LLM Evaluate → Final Report
```

## Search strategies

| # | Strategy | Source | What it finds |
|---|----------|--------|---------------|
| 1 | Semantic search | Scholar Inbox | Papers similar to your abstract |
| 2 | Similar papers | Scholar Inbox | Papers similar to top candidates |
| 3 | Paper detail | Scholar Inbox | References + citations of top hits |
| 4 | Keyword search | Semantic Scholar | Papers matching keyword combinations |
| 5 | Citation chains | Semantic Scholar | Who cites / is cited by seed papers |
| 6 | Author tracking | Semantic Scholar | All papers by key researchers |
| 7 | arXiv search | arXiv API | Recent preprints by topic |
| 8 | Cross-reference | Local scoring | Papers found by multiple strategies ranked higher |

## Setup

### 1. Install

```bash
git clone https://github.com/yourusername/auto-citetion.git
cd auto-citetion
uv sync
```

### 2. Scholar Inbox cookie (optional but recommended)

Scholar Inbox has the best semantic search engine. To use it:

1. Go to [scholar-inbox.com](https://www.scholar-inbox.com) and log in
2. Open DevTools (F12) → Application → Cookies → `scholar-inbox.com`
3. Copy the `session` cookie value
4. Save it:

```bash
echo "your_session_cookie" > .scholar_inbox_cookie
```

### 3. HuggingFace access (for LLM evaluation)

The LLM evaluator uses `google/gemma-4-E4B-it` by default. Make sure you have:

```bash
huggingface-cli login
```

You need a GPU with ~16GB VRAM for the default model.

## Usage

### 1. Create a config file

Copy and edit the example:

```bash
cp example_config.json my_paper.json
```

Fill in your paper's abstract, key queries, seed paper arXiv IDs, and key authors. See `example_config.json` for the format.

### 2. Run the full pipeline

```bash
uv run auto-citetion my_paper.json -o results/
```

This runs all 3 stages: search → LLM evaluate → report.

### 3. Options

```bash
# Skip Scholar Inbox (no cookie)
uv run auto-citetion my_paper.json --skip-si

# Skip LLM evaluation (just search + keyword scoring)
uv run auto-citetion my_paper.json --skip-llm

# Re-run LLM eval on existing search results
uv run auto-citetion my_paper.json --skip-search

# Quick test (fewer queries, top 30 only)
uv run auto-citetion my_paper.json --fast

# Use a different LLM
uv run auto-citetion my_paper.json --model google/gemma-4-E2B-it

# Evaluate more/fewer papers with LLM
uv run auto-citetion my_paper.json --top 100

# Lower the keyword score threshold
uv run auto-citetion my_paper.json --min-score 3.0
```

## Output

```
results/
├── final_reading_list.md     # The main output — grouped, scored, explained
├── papers/                   # Individual markdown files per paper
│   ├── MUST-CITE_paper_title.md
│   ├── SHOULD-CITE_paper_title.md
│   └── ...
├── all_candidates_raw.json   # All discovered papers with scores
└── llm_evaluated.json        # LLM evaluation results
```

### `final_reading_list.md` structure

- **Verdict Summary** — counts per verdict
- **Must Cite** — essential papers with LLM reasoning
- **Should Cite** — strong related work
- **Maybe Cite** — tangential, cite if space permits
- **Papers by Section** — which papers to cite where
- **Papers by Topic** — grouped by research area

## Config file reference

```json
{
  "paper_context": "Full context for LLM evaluation (title, abstract, contributions, methods, sections)",
  "paper_abstract": "Shorter abstract for Scholar Inbox semantic search",
  "scholar_inbox_queries": ["semantic query 1", "semantic query 2"],
  "semantic_scholar_queries": ["keyword query 1", "keyword query 2"],
  "seed_arxiv_ids": ["2004.07780", "1911.08731"],
  "key_authors": ["Robert Geirhos", "Been Kim"],
  "arxiv_queries": ["all:\"keyword\" AND cat:cs.CV"]
}
```

**Tips for good queries:**
- `scholar_inbox_queries`: Write 1-2 sentence descriptions of your paper from different angles. More like abstract snippets than keywords.
- `semantic_scholar_queries`: Short keyword combinations (3-5 terms).
- `seed_arxiv_ids`: arXiv IDs of your closest related work. The tool will crawl their citations and references.
- `key_authors`: Researchers in your field. The tool will find all their papers.

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- GPU with ~16GB VRAM (for LLM evaluation with gemma-4-E4B)
- Scholar Inbox account (optional, for best semantic search)

## License

MIT
