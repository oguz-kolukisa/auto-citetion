"""Shared HTTP helpers for API calls."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request


def http_get(url: str, headers: dict | None = None, timeout: int = 15) -> bytes | None:
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "auto-citetion/1.0")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("    Rate limited, waiting 10s...", file=sys.stderr)
            time.sleep(10)
            return http_get(url, headers, timeout)
        if e.code != 404:
            print(f"    HTTP {e.code}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)
        return None


def http_post(url: str, data: dict, headers: dict | None = None) -> dict | None:
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "auto-citetion/1.0")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("    Rate limited, waiting 10s...", file=sys.stderr)
            time.sleep(10)
            return http_post(url, data, headers)
        return None
    except Exception:
        return None
