# Auto-Citetion

Automated paper discovery and LLM-powered citation evaluation. Searches 6 academic APIs in parallel, recursively expands through citation networks, and uses a local LLM to score each paper's relevance to your research.

## How It Works

```
config.json --> Search (6 APIs x 8 strategies) --> Recursive Expansion --> Score & Filter --> LLM Evaluate --> final_reading_list.md
```

**Search strategies (6 APIs in parallel):**
- **Google Scholar** - keyword search + cited-by traversal (via `scholarly`)
- **Scholar Inbox** - semantic search + similar papers + refs/cited-by
- **Semantic Scholar** - keyword search + citation chains + author tracking
- **arXiv** - API search with category filters
- **OpenAlex** - semantic search + cited-by chains
- **DBLP** - keyword search + venue scanning (e.g., all NeurIPS 2024 papers)

**Recursive expansion:** takes the top-scoring papers, crawls their citations and similar papers, scores those, and repeats up to 3 rounds deep. This snowball effect catches papers that keyword search alone would miss.

**LLM evaluation** (Gemma 4 4B by default):
- Verdict: `must_cite` / `should_cite` / `maybe_cite` / `skip`
- Which section to cite in
- Why to cite + how your work differs

## Installation

```bash
git clone https://github.com/oguz-kolukisa/auto-citetion.git
cd auto-citetion
uv sync
```

### Optional: Scholar Inbox

For best semantic search results:

1. Log in to [scholar-inbox.com](https://www.scholar-inbox.com)
2. Open DevTools (F12) -> Application -> Cookies -> copy the `session` value
3. Save it: `echo "your_cookie" > .scholar_inbox_cookie`

### Optional: LLM evaluation

Requires a GPU with ~16 GB VRAM and a HuggingFace login:

```bash
huggingface-cli login
```

## Quick Start

### 1. Create a config file

```bash
cp example_config.json my_paper.json
```

Edit `my_paper.json` with your paper's abstract, queries, seed papers, and key authors. See [Config Format](#config-format) for all fields.

### 2. Run

```bash
# Full pipeline
uv run auto-citetion my_paper.json -o results/

# Search only (no LLM, just keyword scoring)
uv run auto-citetion my_paper.json --skip-llm

# Re-evaluate existing results with LLM
uv run auto-citetion my_paper.json --skip-search

# Quick test (fewer queries, top 30 only, 1 recursive round)
uv run auto-citetion my_paper.json --fast

# Different model
uv run auto-citetion my_paper.json --model google/gemma-4-E2B-it

# Download PDFs for must_cite and should_cite papers
uv run download-pdfs results/llm_results.json results/pdfs/
```

## CLI Options

```
config                Config JSON file (required)
-o, --output DIR      Output directory (default: .)
--refs PATH           references.md for dedup
--skip-search         Skip search, use existing results
--skip-si             Skip Scholar Inbox
--skip-gs             Skip Google Scholar
--skip-llm            Skip LLM evaluation
--model ID            HuggingFace model (default: google/gemma-4-E4B-it)
--top N               Papers to evaluate with LLM (default: 100)
--min-score N         Minimum keyword score (default: 3.0)
--fast                Fewer queries + top 30 only + 1 recursive round
--depth N             Recursive expansion depth (default: 3, 0=off)
--expand-top N        Papers to expand per round (default: 25)
--max-retries N       Max 429 retries per request (default: 3)
--backoff N [N ...]   Backoff seconds per retry (default: 15 30 60)
```

## Output

```
results/
  final_reading_list.md   # Grouped by verdict + topic with reasoning
  papers/                 # One markdown file per paper
  all_candidates.json     # Raw search results
  llm_results.json        # LLM evaluations
```

## Config Format

```json
{
  "paper_context": "Full paper description for LLM (title, abstract, contributions, sections)",
  "paper_abstract": "Short abstract for Scholar Inbox semantic search.",

  "scholar_inbox_queries": [
    "Describe your paper's core idea in 1-2 sentences.",
    "Describe it from a different angle."
  ],
  "semantic_scholar_queries": [
    "keyword query one",
    "keyword query two"
  ],
  "seed_arxiv_ids": ["2004.07780"],
  "key_authors": ["Author Name"],
  "arxiv_queries": ["all:\"keyword\" AND cat:cs.CV"],
  "dblp_venues": [["NeurIPS", 2024], ["CVPR", 2025]],
  "google_scholar_cite_titles": ["Exact Title of a Key Paper"]
}
```

### Custom scoring keywords (optional)

By default, scoring uses general academic keywords. Override them for your domain:

```json
{
  "scoring": {
    "high_keywords": ["your", "high-weight", "terms"],
    "medium_keywords": ["medium-weight", "terms"],
    "low_keywords": ["low-weight", "terms"],
    "categories": {
      "your_topic": ["keyword1", "keyword2"]
    }
  },
  "category_labels": {
    "your_topic": ["DISPLAY LABEL", "Cite in: Section Name"]
  }
}
```

## Project Structure

```
src/auto_citetion/
  __init__.py
  __main__.py         # python -m auto_citetion
  cli.py              # CLI entry point, report generation
  search.py           # All search strategies, scoring, dedup
  evaluate.py         # LLM evaluation with HuggingFace transformers
  download_pdfs.py    # PDF downloader for top papers
example_config.json   # Template config
```

## License

MIT
