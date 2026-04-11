#!/usr/bin/env python3
"""
Download the second half of the corpus (contiguous slice of xc_metadata_unified.csv).

Use a different network than whoever runs download_xc_first_half.py when possible.

From repo root (lets-solve-it/):
  python scripts/download_xc_second_half.py
  python scripts/download_xc_second_half.py --out-dir D:/xc_audio

Merge: copy your audio/xc/*.mp3 into the first dev's OUT_DIR (same layout).
"""

from __future__ import annotations

import sys

from download_xc_audio import main as run_download


def main() -> int:
    return run_download(["--shard-index", "1", "--num-shards", "2", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
