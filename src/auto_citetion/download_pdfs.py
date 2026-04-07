#!/usr/bin/env python3
"""Download PDFs for must_cite and should_cite papers."""

import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def sanitize_filename(title: str) -> str:
    slug = re.sub(r"[^a-z0-9\s-]", "", title.lower().strip())
    return re.sub(r"[\s]+", "_", slug)[:80]


def download_pdf(url: str, path: Path) -> bool:
    if path.exists():
        return True
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "auto-citetion/1.0")
        with urllib.request.urlopen(req, timeout=30) as r:
            path.write_bytes(r.read())
        return True
    except Exception as e:
        print(f"    FAILED: {e}", file=sys.stderr)
        return False


def download_from_arxiv(arxiv_id: str, path: Path) -> bool:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return download_pdf(url, path)


def download_from_semantic_scholar(title: str, path: Path) -> bool:
    from urllib.parse import quote
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote(title)}&limit=1&fields=openAccessPdf"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "auto-citetion/1.0")
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        if data.get("data"):
            pdf_info = data["data"][0].get("openAccessPdf")
            if pdf_info and pdf_info.get("url"):
                return download_pdf(pdf_info["url"], path)
    except Exception:
        pass
    return False


def main():
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("neurips_results/llm_results.json")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_path.parent / "pdfs"

    with open(results_path) as f:
        papers = json.load(f)

    to_download = [p for p in papers if p.get("llm_verdict") in ("must_cite", "should_cite")]
    must = [p for p in to_download if p["llm_verdict"] == "must_cite"]
    should = [p for p in to_download if p["llm_verdict"] == "should_cite"]

    must_dir = output_dir / "must_cite"
    should_dir = output_dir / "should_cite"
    must_dir.mkdir(parents=True, exist_ok=True)
    should_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(must)} must_cite + {len(should)} should_cite = {len(to_download)} papers")
    print(f"Output: {output_dir}/\n")

    downloaded, failed, skipped = 0, 0, 0

    for i, paper in enumerate(to_download):
        verdict = paper["llm_verdict"]
        title = paper.get("title", "unknown")
        arxiv_id = paper.get("arxiv_id", "")
        dest_dir = must_dir if verdict == "must_cite" else should_dir
        filename = sanitize_filename(title) + ".pdf"
        path = dest_dir / filename

        if path.exists():
            skipped += 1
            continue

        print(f"[{i+1}/{len(to_download)}] {title[:60]}…", file=sys.stderr)

        success = False
        if arxiv_id:
            success = download_from_arxiv(arxiv_id, path)
            time.sleep(1)
        if not success:
            success = download_from_semantic_scholar(title, path)
            time.sleep(1)

        if success:
            downloaded += 1
        else:
            failed += 1

    print(f"\nDone! Downloaded: {downloaded} | Cached: {skipped} | Failed: {failed}")
    print(f"Must cite PDFs:   {must_dir}/")
    print(f"Should cite PDFs: {should_dir}/")


if __name__ == "__main__":
    main()
