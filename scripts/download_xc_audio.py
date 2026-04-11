#!/usr/bin/env python3
"""
Bulk-download Xeno-canto recordings listed in xc_metadata_unified.csv.

Files are stored under OUT_DIR using the same relative paths as the CSV
(e.g. OUT_DIR/audio/xc/1060250.mp3). Point CLAP training at OUT_DIR as the
audio root so pair JSON paths resolve.

Default is no pause after success (`--sleep 0`). Default `--workers 30` with
adaptive throttling: if Xeno-canto returns many HTTP 429s in a short window,
concurrency is reduced automatically (down to `--min-workers`). Override with
`--no-adaptive`. See https://xeno-canto.org/help/terms-of-use

Usage (from repo root):
  python scripts/download_xc_audio.py
  python scripts/download_xc_audio.py --limit 50 --dry-run
  python scripts/download_xc_audio.py --out-dir D:/xc_mirror

Two machines / two developers (disjoint halves of the same CSV order):
  python scripts/download_xc_first_half.py
  python scripts/download_xc_second_half.py
  (Same flags as above; merge both trees into one OUT_DIR before training.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

XC_ID_RE = re.compile(r"audio/xc/(\d+)\.(\w+)", re.I)
HEADERS = {
    "User-Agent": "lets-solve-it-xc-bulk/1.0 (research; respectful sequential downloads)",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_csv(path: str | None) -> Path:
    if path:
        p = Path(path)
        if p.is_file():
            return p
        raise FileNotFoundError(path)
    root = repo_root()
    for candidate in (root / "scripts" / "xc_metadata_unified.csv", root / "xc_metadata_unified.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("xc_metadata_unified.csv (pass --csv)")


def parse_filepath(filepath: str) -> tuple[int, str] | None:
    m = XC_ID_RE.search(str(filepath).strip())
    if not m:
        return None
    return int(m.group(1)), m.group(2).lower()


def download_url(xc_id: int) -> str:
    return f"https://xeno-canto.org/{xc_id}/download"


def fetch_to_path(
    url: str,
    dest: Path,
    session: requests.Session,
    timeout: int,
    retries: int,
    sleep_after: float,
) -> tuple[bool, str, int]:
    """
    Returns (success, message, num_429_responses_seen).
    Counts each HTTP 429 encountered (including before a later success).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    seen_429 = 0

    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=timeout, stream=True)
            if r.status_code == 429:
                seen_429 += 1
                r.close()
                wait = min(60.0, 2.0 ** (attempt + 2))
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                r.close()
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            nbytes = 0
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    if chunk:
                        f.write(chunk)
                        nbytes += len(chunk)
            if nbytes < 256:
                part.unlink(missing_ok=True)
                return False, f"too_small ({nbytes} bytes)", seen_429
            part.replace(dest)
            if sleep_after > 0:
                time.sleep(sleep_after)
            return True, "ok", seen_429
        except requests.RequestException as e:
            part.unlink(missing_ok=True)
            if attempt + 1 >= retries:
                return False, str(e), seen_429
            time.sleep(1.0 * (attempt + 1))
    return False, "retries_exhausted", seen_429


# One Session per thread (requests.Session is not thread-safe).
_tls = threading.local()


def _pool_init_worker() -> None:
    _tls.session = requests.Session()


class AdaptiveConcurrencyLimiter:
    """
    Caps how many downloads run at once. On many HTTP 429s inside a sliding
    window, lowers the cap (halves, down to min_cap).
    """

    def __init__(
        self,
        initial_cap: int,
        min_cap: int,
        window_s: float,
        burst_threshold: int,
        enabled: bool,
    ) -> None:
        self._min_cap = max(1, min_cap)
        self._cap = max(self._min_cap, initial_cap)
        self._in_use = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._window_s = max(1.0, window_s)
        self._burst_threshold = max(1, burst_threshold)
        self._enabled = enabled
        self._429_times: deque[float] = deque()

    @property
    def cap(self) -> int:
        return self._cap

    def acquire_slot(self) -> None:
        with self._cond:
            while self._in_use >= self._cap:
                self._cond.wait()
            self._in_use += 1

    def release_slot(self) -> None:
        with self._cond:
            self._in_use -= 1
            self._cond.notify_all()

    def register_429_hits(self, count: int) -> None:
        if count <= 0 or not self._enabled:
            return
        now = time.monotonic()
        with self._cond:
            for _ in range(count):
                self._429_times.append(now)
            cutoff = now - self._window_s
            while self._429_times and self._429_times[0] < cutoff:
                self._429_times.popleft()
            if len(self._429_times) >= self._burst_threshold:
                old = self._cap
                new = max(self._min_cap, (old + 1) // 2)
                if new < old:
                    self._cap = new
                    self._429_times.clear()
                    tqdm.write(
                        f"[adaptive] HTTP 429 burst (≥{self._burst_threshold} in "
                        f"{self._window_s:.0f}s) → concurrency {old} → {new}",
                        file=sys.stderr,
                    )
                self._cond.notify_all()


def try_one_download_network(
    xc_id: int,
    dest: Path,
    session: requests.Session,
    timeout: int,
    retries: int,
    sleep_after: float,
) -> tuple[str, str | None, int]:
    """Network-only: returns (status, err, n_429_hits)."""
    url = download_url(xc_id)
    success, msg, n429 = fetch_to_path(
        url, dest, session, timeout, retries, sleep_after
    )
    if success:
        return "ok", None, n429
    return "fail", msg, n429


def slice_jobs_for_shard(
    jobs: list[tuple[int, str, Path]], shard_index: int, num_shards: int
) -> list[tuple[int, str, Path]]:
    """
    Contiguous split so shard 0 is the first slice, shard 1 the second, etc.
    Remainder rows go to the earliest shards (e.g. 5 items / 2 -> [3, 2]).
    """
    n = len(jobs)
    base = n // num_shards
    rem = n % num_shards
    sizes = [base + (1 if i < rem else 0) for i in range(num_shards)]
    offset = sum(sizes[:shard_index])
    return jobs[offset : offset + sizes[shard_index]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bulk download Xeno-canto audio from unified CSV.")
    ap.add_argument("--csv", default=None, help="Path to xc_metadata_unified.csv")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Root directory for audio tree (default: scripts/data/xc_audio)",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to pause after each successful download (default 0; use e.g. 0.35 if rate-limited)",
    )
    ap.add_argument("--timeout", type=int, default=120, help="Per-request timeout seconds")
    ap.add_argument("--retries", type=int, default=4, help="Attempts per file on errors")
    ap.add_argument("--limit", type=int, default=0, help="Max files to download (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only")
    ap.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even when destination already exists and is non-empty",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="Append JSONL log of each attempt (default: OUT_DIR/download_manifest.jsonl)",
    )
    ap.add_argument(
        "--shard-index",
        type=int,
        default=None,
        metavar="N",
        help="0-based shard index (use with --num-shards). Omit to process the full list.",
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        metavar="K",
        help="Split the deduplicated CSV job list into K contiguous slices (default 1 = all).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=30,
        metavar="N",
        help="Initial concurrent download threads (default 30).",
    )
    ap.add_argument(
        "--min-workers",
        type=int,
        default=4,
        metavar="N",
        help="Floor when adaptive throttling reduces concurrency (default 4).",
    )
    ap.add_argument(
        "--429-window",
        type=float,
        default=12.0,
        dest="rate_window_s",
        metavar="SEC",
        help="Sliding window (seconds) for counting 429s (default 12).",
    )
    ap.add_argument(
        "--429-burst",
        type=int,
        default=5,
        dest="rate_burst",
        metavar="N",
        help="429 responses in window before halving concurrency (default 5).",
    )
    ap.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable automatic worker reduction on 429 bursts.",
    )
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.num_shards < 1:
        print("--num-shards must be >= 1", file=sys.stderr)
        return 1
    if args.shard_index is not None:
        if args.num_shards < 2:
            print("--shard-index requires --num-shards >= 2", file=sys.stderr)
            return 1
        if not (0 <= args.shard_index < args.num_shards):
            print(f"--shard-index must be in [0, {args.num_shards - 1}]", file=sys.stderr)
            return 1
    if args.workers < 1:
        print("--workers must be >= 1", file=sys.stderr)
        return 1
    if args.min_workers < 1:
        print("--min-workers must be >= 1", file=sys.stderr)
        return 1
    if args.min_workers > args.workers:
        print("--min-workers cannot be greater than --workers", file=sys.stderr)
        return 1

    csv_path = resolve_csv(args.csv)
    out_root = Path(args.out_dir) if args.out_dir else repo_root() / "scripts" / "data" / "xc_audio"
    manifest_path = Path(args.manifest) if args.manifest else out_root / "download_manifest.jsonl"

    df = pd.read_csv(csv_path)
    if "filepath" not in df.columns:
        print("CSV missing 'filepath' column.", file=sys.stderr)
        return 1

    jobs: list[tuple[int, str, Path]] = []
    bad_rows = 0
    for fp in df["filepath"].astype(str).unique():
        parsed = parse_filepath(fp)
        if not parsed:
            bad_rows += 1
            continue
        xc_id, ext = parsed
        rel = Path(fp.replace("\\", "/"))
        dest = out_root / rel
        jobs.append((xc_id, ext, dest))

    # Order follows first appearance in CSV (stable for --limit smoke tests)
    n_full = len(jobs)
    if args.shard_index is not None:
        jobs = slice_jobs_for_shard(jobs, args.shard_index, args.num_shards)
    n_in_shard = len(jobs)

    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]

    n_total = len(jobs)
    print(f"CSV:        {csv_path}")
    print(f"Out root:   {out_root.resolve()}")
    if args.shard_index is not None:
        print(
            f"Shard:      {args.shard_index + 1}/{args.num_shards} "
            f"({n_in_shard} files in slice of {n_full} total)"
        )
    print(f"Files:      {n_total} (skipped {bad_rows} bad filepath rows in CSV)")
    print(
        f"Workers:    {args.workers} threads, cap starts at {args.workers}"
        + (
            f" (adaptive down to {args.min_workers} on 429 bursts)"
            if not args.no_adaptive
            else " (fixed cap; --no-adaptive)"
        )
    )
    if args.sleep > 0:
        print(f"Sleep:      {args.sleep}s after each success")

    if args.dry_run:
        for xc_id, _ext, dest in jobs[:10]:
            print(f"  would fetch {xc_id} -> {dest}")
        if n_total > 10:
            print(f"  ... and {n_total - 10} more")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)

    ok = skip = fail = 0
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_lock = threading.Lock()

    def log_line(rec: dict) -> None:
        rec["ts"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with manifest_lock:
            with open(manifest_path, "a", encoding="utf-8") as mf:
                mf.write(line)

    limiter = AdaptiveConcurrencyLimiter(
        initial_cap=args.workers,
        min_cap=args.min_workers,
        window_s=args.rate_window_s,
        burst_threshold=args.rate_burst,
        enabled=not args.no_adaptive,
    )

    def run_one(job: tuple[int, str, Path]) -> tuple[str, int, str | None]:
        xc_id, _ext, dest = job
        if (
            not args.no_skip_existing
            and dest.is_file()
            and dest.stat().st_size > 512
        ):
            log_line({"xc_id": xc_id, "path": str(dest), "status": "skipped_exists"})
            return "skipped_exists", xc_id, ""

        limiter.acquire_slot()
        try:
            status, err, n429 = try_one_download_network(
                xc_id,
                dest,
                _tls.session,
                args.timeout,
                args.retries,
                args.sleep,
            )
        finally:
            limiter.release_slot()

        if n429:
            limiter.register_429_hits(n429)

        if status == "ok":
            log_line({"xc_id": xc_id, "path": str(dest), "status": "ok"})
        else:
            log_line(
                {
                    "xc_id": xc_id,
                    "path": str(dest),
                    "status": "error",
                    "detail": err or "",
                }
            )
        return status, xc_id, err or ""

    with ThreadPoolExecutor(
        max_workers=args.workers,
        initializer=_pool_init_worker,
    ) as ex:
        futures = [ex.submit(run_one, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="download", unit="file"):
            status, xc_id, err = fut.result()
            if status == "skipped_exists":
                skip += 1
            elif status == "ok":
                ok += 1
            else:
                fail += 1
                tqdm.write(f"[fail] id={xc_id} {err}", file=sys.stderr)

    print(f"\nDone. ok={ok} skipped={skip} failed={fail}")
    if not args.no_adaptive:
        print(f"Final download concurrency cap: {limiter.cap}")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Use this directory as the audio root for training pairs: {out_root.resolve()}")
    return 0 if fail == 0 else 2


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
