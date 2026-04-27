#!/usr/bin/env python3
"""
download_targeted_xc.py
========================
Targeted Xeno-canto audio downloader that fills dataset coverage gaps.

Unlike download_xc_audio.py (which re-downloads everything already in the CSV),
this script discovers *new* recordings needed to improve quality:

  Phase 1  Audit    — reads xc_metadata_unified.csv + clap_all_labels.json to
                       find every (species, voc_type) combo below MIN_RECORDINGS.
  Phase 2  Fetch    — queries the XC API globally (no cnt:canada) for each weak
                       species. One API query per species covers all voc types.
                       Only quality A and B recordings are kept.
  Phase 3  Select   — drops IDs already in the CSV, caps at MAX_PER_COMBO new
                       downloads per combo, writes plan to a state JSON.
  Phase 4  Download — fetches audio with adaptive concurrency and .part resume.
  Phase 5  Append   — adds new rows to xc_metadata_unified.csv.

Priority ordering (most urgent fetched first):
  owls (Strigiformes) > raptors (Accipitriformes / Falconiformes) > waterbirds
  (Anseriformes) > shorebirds (Charadriiformes) > all others.

Vocalization type normalization matches build_clap_training_pairs.py exactly:
  voc_type = raw_type.split(",")[0].strip().lower()

Usage (from repo root):
  python scripts/download_targeted_xc.py
  python scripts/download_targeted_xc.py --dry-run
  python scripts/download_targeted_xc.py --audit-only
  python scripts/download_targeted_xc.py --fetch-only
  python scripts/download_targeted_xc.py --download-only
  python scripts/download_targeted_xc.py --min-recordings 20 --max-per-combo 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a, **_kw):  # type: ignore[misc]
        pass

from tqdm import tqdm

# ── constants ──────────────────────────────────────────────────────────────────
DEFAULT_MIN_RECORDINGS = 15       # floor clips per (species, voc_type) combo
DEFAULT_MAX_PER_COMBO  = 25       # max new downloads per combo
DEFAULT_WORKERS        = 15       # initial concurrent download threads
DEFAULT_MIN_WORKERS    = 4

QUALITY_FLOOR = 4                 # 4=B 5=A; C/D/E/unscored rejected
Q_MAP = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "no score": 0}

XC_API_URL = "https://xeno-canto.org/api/3/recordings"
XC_DL_URL  = "https://xeno-canto.org/{xc_id}/download"
XC_ID_RE   = re.compile(r"audio/xc/(\d+)\.", re.I)

HEADERS = {"User-Agent": "lets-solve-it-targeted/1.0 (research; respectful downloads)"}

# Taxonomic order → priority (lower = fetch first)
ORDER_PRIORITY: dict[str, int] = {
    "Strigiformes":      0,   # owls
    "Accipitriformes":   1,   # hawks, eagles, ospreys
    "Falconiformes":     2,   # falcons
    "Cathartiformes":    2,   # vultures
    "Anseriformes":      3,   # ducks, geese, swans
    "Charadriiformes":   4,   # shorebirds, gulls, terns, skuas
    "Gaviiformes":       5,   # loons
    "Podicipediformes":  5,   # grebes
    "Procellariiformes": 5,   # petrels, shearwaters
    "Pelecaniformes":    5,   # herons, ibises, pelicans
    "Suliformes":        5,   # cormorants, boobies
    "Gruiformes":        5,   # cranes, rails
}

# Match build_clap_training_pairs.py SKIP_NAMES and SKIP_TYPES
SKIP_NAMES = frozenset({
    "soundscape", "identity unknown", "noise", "speech", "canine", "squirrel",
    "insects", "rooster", "other", "background", "unknown", "nan", "",
})
SKIP_TYPES = frozenset({"uncertain", "various", "various calls", "nan", ""})


# ── helpers ───────────────────────────────────────────────────────────────────

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_api_key() -> str:
    root = repo_root()
    for env in (root / ".env", Path.cwd() / ".env"):
        if env.is_file():
            load_dotenv(env)
            break
    else:
        load_dotenv()
    key = os.getenv("XC_API_KEY", "").strip()
    if not key:
        print("ERROR: XC_API_KEY not set.  Add it to .env in the repo root.", file=sys.stderr)
        sys.exit(1)
    return key


def normalize_voc(raw: str) -> str:
    """Identical to build_clap_training_pairs.py line 143."""
    return raw.split(",")[0].strip().lower()


def existing_xc_ids(csv_path: Path) -> set[int]:
    if not csv_path.is_file():
        return set()
    df = pd.read_csv(csv_path, usecols=["filepath"])
    ids: set[int] = set()
    for fp in df["filepath"].astype(str):
        m = XC_ID_RE.search(fp)
        if m:
            ids.add(int(m.group(1)))
    return ids


# ── Phase 1: audit ─────────────────────────────────────────────────────────────

def audit_coverage(
    csv_path: Path,
    labels: dict,
    tax_db: dict,
    min_recordings: int,
) -> list[dict]:
    """
    Returns list of weak-combo dicts sorted by (priority_tier, existing_count).
    Each dict has keys: combo_key, species, voc_type, existing_count,
                        scientific, order, priority.
    """
    df = pd.read_csv(csv_path)

    # Count current clips per combo using the same normalization as the trainer
    combo_count: dict[str, int] = defaultdict(int)
    for _, row in df.iterrows():
        name    = str(row.get("common_name", "")).strip()
        voc_raw = str(row.get("vocalization_type", "")).strip()
        if name.lower() in SKIP_NAMES:
            continue
        # Birds-only (skip mammals/amphibians the taxonomy can identify)
        tax_class = tax_db.get(name, {}).get("class", "")
        if tax_class and tax_class != "Aves":
            continue
        voc = normalize_voc(voc_raw)
        if voc in SKIP_TYPES:
            continue
        key = f"{name}||{voc}"
        if key in labels:
            combo_count[key] += 1

    weak: list[dict] = []
    for combo_key in labels:
        count = combo_count.get(combo_key, 0)
        if count >= min_recordings:
            continue
        species, voc_type = combo_key.split("||", 1)
        if species.lower() in SKIP_NAMES:
            continue
        tax = tax_db.get(species, {})
        if tax.get("class", "Aves") != "Aves":
            continue
        order      = tax.get("order", "")
        scientific = tax.get("scientific", "")
        priority   = ORDER_PRIORITY.get(order, 99)
        weak.append({
            "combo_key":      combo_key,
            "species":        species,
            "voc_type":       voc_type,
            "existing_count": count,
            "scientific":     scientific,
            "order":          order,
            "priority":       priority,
        })

    weak.sort(key=lambda x: (x["priority"], x["existing_count"]))
    return weak


# ── Phase 2: fetch XC metadata ────────────────────────────────────────────────

def _xc_page(
    query: str,
    api_key: str,
    page: int,
    session: requests.Session,
) -> dict | None:
    params = {"query": query, "key": api_key, "page": page}
    for attempt in range(3):
        try:
            r = session.get(XC_API_URL, params=params, headers=HEADERS, timeout=25)
            if r.status_code == 429:
                time.sleep(30)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                tqdm.write(f"  [api-warn] query={query!r} page={page}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    return None


def fetch_species_recordings(
    species: str,
    scientific: str,
    api_key: str,
    inter_page_sleep: float = 0.5,
) -> list[dict]:
    """
    Fetch *all* XC recordings for a species (global, no country filter).
    Prefers scientific name for precision; falls back to English name.
    """
    if scientific and " " in scientific:
        parts  = scientific.split()
        query  = f"gen:{parts[0]} sp:{parts[1]}"
    elif scientific:
        query  = scientific
    else:
        # Wrap multi-word English names so XC treats them as a phrase
        query  = f'"{species}"'

    session = requests.Session()
    first   = _xc_page(query, api_key, page=1, session=session)
    if not first or "recordings" not in first:
        return []

    total_pages = int(first.get("numPages", 1))
    all_recs    = list(first["recordings"])

    for p in range(2, total_pages + 1):
        time.sleep(inter_page_sleep)
        data = _xc_page(query, api_key, page=p, session=session)
        if data and "recordings" in data:
            all_recs.extend(data["recordings"])
        else:
            break

    return all_recs


# ── Phase 3: select new recordings ───────────────────────────────────────────

def select_for_species(
    species: str,
    all_recs: list[dict],
    weak_voc_types: dict[str, int],  # {voc_type: existing_count} for this species
    existing_ids: set[int],
    max_per_combo: int,
) -> list[dict]:
    """
    Filter `all_recs` to quality A/B, matching one of the weak voc_types,
    not already in the CSV, up to (max_per_combo - existing_count) per voc_type.

    Returns a list of unified-schema row dicts, each with an extra '_xc_id' key.
    """
    # Per-voc-type budget remaining: how many more we can still accept
    budget: dict[str, int] = {
        vt: max_per_combo - cnt
        for vt, cnt in weak_voc_types.items()
        if max_per_combo - cnt > 0
    }
    if not budget:
        return []

    selected: list[dict] = []
    seen_ids: set[int]   = set()

    for rec in all_recs:
        xc_id_str = str(rec.get("id", "")).strip()
        if not xc_id_str.isdigit():
            continue
        xc_id = int(xc_id_str)
        if xc_id in existing_ids or xc_id in seen_ids:
            continue

        # Quality gate: A or B only
        q_val = Q_MAP.get(str(rec.get("q", "no score")).strip(), 0)
        if q_val < QUALITY_FLOOR:
            continue

        # Vocalization type: normalize and check against weak targets
        voc_raw  = str(rec.get("type", "")).strip()
        voc_norm = normalize_voc(voc_raw)
        if voc_norm in SKIP_TYPES:
            continue
        if voc_norm not in budget or budget[voc_norm] <= 0:
            continue

        # All checks passed — add this recording
        xc_id_int = xc_id
        filepath  = f"audio/xc/{xc_id_int}.mp3"

        row = {
            "filepath":          filepath,
            "species_code":      species.lower().replace(" ", "_"),
            "common_name":       species,
            "vocalization_type": voc_raw,
            "quality_rating":    q_val,
            "duration":          str(rec.get("length", "0:00")),
            "source":            "xeno-canto",
            "_xc_id":            xc_id_int,
        }
        selected.append(row)
        seen_ids.add(xc_id_int)
        budget[voc_norm] -= 1

        # Stop early if all budgets are exhausted
        if all(b <= 0 for b in budget.values()):
            break

    return selected


# ── Phase 4: download audio ────────────────────────────────────────────────────

class _AdaptiveLimiter:
    def __init__(self, cap: int, min_cap: int, window_s: float, burst: int) -> None:
        self._cap   = max(min_cap, cap)
        self._min   = max(1, min_cap)
        self._in    = 0
        self._lock  = threading.Lock()
        self._cond  = threading.Condition(self._lock)
        self._times: deque[float] = deque()
        self._ws    = window_s
        self._burst = burst

    @property
    def cap(self) -> int:
        return self._cap

    def acquire(self) -> None:
        with self._cond:
            while self._in >= self._cap:
                self._cond.wait()
            self._in += 1

    def release(self) -> None:
        with self._cond:
            self._in -= 1
            self._cond.notify_all()

    def report_429(self, n: int) -> None:
        if n <= 0:
            return
        now = time.monotonic()
        with self._cond:
            for _ in range(n):
                self._times.append(now)
            cutoff = now - self._ws
            while self._times and self._times[0] < cutoff:
                self._times.popleft()
            if len(self._times) >= self._burst:
                old = self._cap
                new = max(self._min, (old + 1) // 2)
                if new < old:
                    self._cap = new
                    self._times.clear()
                    tqdm.write(
                        f"[adaptive] 429 burst → concurrency {old} → {new}",
                        file=sys.stderr,
                    )
                self._cond.notify_all()


_tls = threading.local()


def _init_worker() -> None:
    _tls.session = requests.Session()


def _download_one(
    xc_id: int,
    dest: Path,
    timeout: int,
    retries: int,
) -> tuple[bool, str, int]:
    """Returns (success, message, n_429)."""
    url  = XC_DL_URL.format(xc_id=xc_id)
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    n429 = 0

    for attempt in range(retries):
        try:
            r = _tls.session.get(url, headers=HEADERS, timeout=timeout, stream=True)
            if r.status_code == 429:
                n429 += 1
                r.close()
                time.sleep(min(60.0, 2.0 ** (attempt + 2)))
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
                return False, f"too_small ({nbytes} bytes)", n429
            part.replace(dest)
            return True, "ok", n429
        except requests.RequestException as e:
            part.unlink(missing_ok=True)
            if attempt + 1 >= retries:
                return False, str(e), n429
            time.sleep(1.0 * (attempt + 1))

    return False, "retries_exhausted", n429


def download_scheduled(
    scheduled: list[dict],
    audio_root: Path,
    workers: int,
    min_workers: int,
    timeout: int,
    retries: int,
    manifest_path: Path,
) -> tuple[int, int, int]:
    """
    Download all rows in `scheduled`.  Each row must have `filepath` and `_xc_id`.
    Returns (ok, skipped, failed).
    """
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_lock = threading.Lock()

    def log(rec: dict) -> None:
        rec["ts"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with manifest_lock:
            with open(manifest_path, "a", encoding="utf-8") as mf:
                mf.write(line)

    limiter = _AdaptiveLimiter(
        cap=workers, min_cap=min_workers,
        window_s=12.0, burst=5,
    )

    jobs = [
        (row["_xc_id"], audio_root / row["filepath"])
        for row in scheduled
    ]

    ok = skip = fail = 0

    def run_one(job: tuple[int, Path]) -> tuple[str, int, str]:
        xc_id, dest = job
        if dest.is_file() and dest.stat().st_size > 512:
            log({"xc_id": xc_id, "path": str(dest), "status": "skipped_exists"})
            return "skip", xc_id, ""
        limiter.acquire()
        try:
            success, msg, n429 = _download_one(xc_id, dest, timeout, retries)
        finally:
            limiter.release()
        if n429:
            limiter.report_429(n429)
        status = "ok" if success else "error"
        log({"xc_id": xc_id, "path": str(dest), "status": status, "detail": msg})
        return status, xc_id, msg

    with ThreadPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futures = [ex.submit(run_one, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="download", unit="file"):
            status, xc_id, msg = fut.result()
            if status == "skip":
                skip += 1
            elif status == "ok":
                ok += 1
            else:
                fail += 1
                tqdm.write(f"[fail] id={xc_id}  {msg}", file=sys.stderr)

    return ok, skip, fail


# ── Phase 5: append to CSV ────────────────────────────────────────────────────

def append_to_csv(csv_path: Path, new_rows: list[dict]) -> None:
    """
    Atomically append new rows (without the _xc_id scratch field) to the CSV.
    Deduplicates against the current CSV content by filepath before writing.
    """
    CSV_COLS = ["filepath", "species_code", "common_name",
                "vocalization_type", "quality_rating", "duration", "source"]

    existing = pd.read_csv(csv_path) if csv_path.is_file() else pd.DataFrame(columns=CSV_COLS)
    existing_fps = set(existing["filepath"].astype(str))

    clean_rows = [
        {k: r[k] for k in CSV_COLS}
        for r in new_rows
        if r["filepath"] not in existing_fps
    ]
    if not clean_rows:
        print("  No new rows to append (all already present in CSV).")
        return

    new_df   = pd.DataFrame(clean_rows, columns=CSV_COLS)
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Write to temp file then replace (avoids partial-write corruption)
    tmp = csv_path.with_suffix(".csv.tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(csv_path)
    print(f"  Appended {len(clean_rows)} rows to {csv_path}  "
          f"(total now: {len(combined)})")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Targeted XC downloader — fills dataset coverage gaps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--csv",
        default=None,
        help="Path to xc_metadata_unified.csv (default: scripts/xc_metadata_unified.csv)",
    )
    ap.add_argument(
        "--labels",
        default=None,
        help="Path to clap_all_labels.json (default: data/clap_all_labels.json)",
    )
    ap.add_argument(
        "--taxonomy",
        default=None,
        help="Path to species_taxonomy.json (default: data/species_taxonomy.json)",
    )
    ap.add_argument(
        "--audio-root",
        default=None,
        help="Root dir for audio files (default: scripts/data/xc_audio)",
    )
    ap.add_argument(
        "--state-file",
        default=None,
        help="JSON state file for fetch/download plan (default: data/targeted_download_state.json)",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="JSONL download manifest path (default: scripts/data/xc_audio/targeted_manifest.jsonl)",
    )
    ap.add_argument(
        "--min-recordings",
        type=int,
        default=DEFAULT_MIN_RECORDINGS,
        metavar="N",
        help=f"Minimum clips per (species, voc_type) combo (default {DEFAULT_MIN_RECORDINGS})",
    )
    ap.add_argument(
        "--max-per-combo",
        type=int,
        default=DEFAULT_MAX_PER_COMBO,
        metavar="N",
        help=f"Max new downloads per combo (default {DEFAULT_MAX_PER_COMBO})",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"Initial concurrent download threads (default {DEFAULT_WORKERS})",
    )
    ap.add_argument(
        "--min-workers",
        type=int,
        default=DEFAULT_MIN_WORKERS,
        metavar="N",
        help=f"Minimum threads after adaptive throttle (default {DEFAULT_MIN_WORKERS})",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-download request timeout in seconds (default 120)",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Download retry attempts per file (default 4)",
    )
    ap.add_argument(
        "--api-page-sleep",
        type=float,
        default=0.5,
        metavar="SEC",
        help="Sleep between XC API page requests (default 0.5s)",
    )
    # Run-mode flags
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--audit-only",
        action="store_true",
        help="Print coverage gap table and exit (no API calls, no downloads)",
    )
    mode.add_argument(
        "--fetch-only",
        action="store_true",
        help="Query XC API and build state file, but do not download audio",
    )
    mode.add_argument(
        "--download-only",
        action="store_true",
        help="Skip audit/fetch, load existing state file and download audio",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan (including API query results) without downloading or writing files",
    )
    return ap.parse_args(argv)


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    root       = repo_root()
    csv_path   = Path(args.csv)        if args.csv        else root / "scripts" / "xc_metadata_unified.csv"
    labels_path= Path(args.labels)     if args.labels     else root / "data"    / "clap_all_labels.json"
    tax_path   = Path(args.taxonomy)   if args.taxonomy   else root / "data"    / "species_taxonomy.json"
    audio_root = Path(args.audio_root) if args.audio_root else root / "scripts" / "data" / "xc_audio"
    state_path = Path(args.state_file) if args.state_file else root / "data"    / "targeted_download_state.json"
    manifest   = Path(args.manifest)   if args.manifest   else audio_root / "targeted_manifest.jsonl"

    for p, label in ((csv_path, "--csv"), (labels_path, "--labels"), (tax_path, "--taxonomy")):
        if not p.is_file():
            print(f"ERROR: {label} file not found: {p}", file=sys.stderr)
            return 1

    print(f"Loading labels … {labels_path}")
    labels: dict = json.loads(labels_path.read_text(encoding="utf-8"))

    print(f"Loading taxonomy … {tax_path}")
    tax_db: dict = json.loads(tax_path.read_text(encoding="utf-8"))

    # ── Phase 1: audit ────────────────────────────────────────────────────────
    if not args.download_only:
        print(f"\nAuditing coverage ({csv_path.name}) …")
        weak = audit_coverage(csv_path, labels, tax_db, args.min_recordings)

        total_combos = sum(1 for k in labels
                           if tax_db.get(k.split("||")[0], {}).get("class", "Aves") == "Aves"
                           and k.split("||")[0].lower() not in SKIP_NAMES)
        print(f"  Total bird combos in labels  : {total_combos}")
        print(f"  Combos below {args.min_recordings} recordings  : {len(weak)}")

        if not weak:
            print("  All combos meet the minimum — nothing to download.")
            return 0

        # Summary by priority tier
        by_order: dict[str, list[dict]] = defaultdict(list)
        for c in weak:
            by_order[c["order"] or "(unknown)"].append(c)

        print("\n  Top-20 most urgent combos:")
        print(f"  {'combo':<45} {'have':>4} {'order'}")
        print(f"  {'-'*45} {'-'*4} {'-'*20}")
        for c in weak[:20]:
            print(f"  {c['combo_key']:<45} {c['existing_count']:>4}   {c['order']}")

        print(f"\n  By order (weak combos):")
        for order_name, items in sorted(by_order.items(),
                                        key=lambda kv: ORDER_PRIORITY.get(kv[0], 99)):
            print(f"    {order_name:<25} {len(items):>4} combos")

        if args.audit_only:
            return 0

    # ── Phase 2+3: fetch metadata from XC API ─────────────────────────────────
    scheduled: list[dict] = []

    if args.download_only:
        if not state_path.is_file():
            print(f"ERROR: state file not found: {state_path}", file=sys.stderr)
            print("  Run without --download-only first to build the plan.", file=sys.stderr)
            return 1
        state = json.loads(state_path.read_text(encoding="utf-8"))
        scheduled = state.get("scheduled", [])
        print(f"Loaded {len(scheduled)} scheduled downloads from {state_path}")
    else:
        api_key = load_api_key()

        # Load existing state to resume a partial fetch
        already_fetched: set[str] = set()
        existing_scheduled: list[dict] = []
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            already_fetched    = set(state.get("fetched_species", []))
            existing_scheduled = state.get("scheduled", [])
            if already_fetched:
                print(f"  Resuming: {len(already_fetched)} species already fetched, "
                      f"{len(existing_scheduled)} downloads already planned.")

        # Existing CSV IDs (so we never re-download)
        print(f"Reading existing IDs from {csv_path.name} …")
        exist_ids = existing_xc_ids(csv_path)
        # Also add IDs from any rows already scheduled (to avoid scheduling them again)
        for r in existing_scheduled:
            exist_ids.add(r["_xc_id"])

        # Group weak combos by species so we make one API call per species
        species_combos: dict[str, list[dict]] = defaultdict(list)
        for c in weak:
            species_combos[c["species"]].append(c)

        # Remove species we've already fully fetched in a previous run
        todo_species = [
            (sp, combos)
            for sp, combos in species_combos.items()
            if sp not in already_fetched
        ]

        print(f"\nPhase 2+3: fetching XC metadata for {len(todo_species)} species …")
        new_scheduled: list[dict] = []

        for i, (species, combos) in enumerate(tqdm(todo_species, desc="species", unit="sp")):
            scientific = combos[0].get("scientific", "")
            # Build per-voc-type budget for this species
            weak_voc_types = {c["voc_type"]: c["existing_count"] for c in combos}

            all_recs = fetch_species_recordings(
                species, scientific, api_key, args.api_page_sleep
            )
            tqdm.write(
                f"  {species} ({scientific or 'no sci'})  "
                f"→ {len(all_recs)} global recs from XC"
            )

            if all_recs:
                selected = select_for_species(
                    species        = species,
                    all_recs       = all_recs,
                    weak_voc_types = weak_voc_types,
                    existing_ids   = exist_ids,
                    max_per_combo  = args.max_per_combo,
                )
                # Register newly selected IDs so later species don't duplicate them
                for r in selected:
                    exist_ids.add(r["_xc_id"])
                new_scheduled.extend(selected)

            already_fetched.add(species)

            # Save state after every species in case we're interrupted
            if not args.dry_run:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                all_scheduled = existing_scheduled + new_scheduled
                state_path.write_text(
                    json.dumps({
                        "fetched_species": sorted(already_fetched),
                        "scheduled":       all_scheduled,
                        "generated_at":    datetime.now(timezone.utc).isoformat(),
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        scheduled = existing_scheduled + new_scheduled

        # Summary
        by_combo: dict[str, int] = defaultdict(int)
        for r in scheduled:
            sp  = r["common_name"]
            voc = normalize_voc(r["vocalization_type"])
            by_combo[f"{sp}||{voc}"] += 1

        print(f"\nPlan summary:")
        print(f"  Species queried   : {len(already_fetched)}")
        print(f"  New downloads     : {len(scheduled)}")
        print(f"  Combos addressed  : {len(by_combo)}")
        if by_combo:
            top10 = sorted(by_combo.items(), key=lambda kv: -kv[1])[:10]
            print(f"  Top-10 combos by new files:")
            for combo_k, cnt in top10:
                print(f"    {combo_k:<45} +{cnt}")

        if args.fetch_only or args.dry_run:
            if not scheduled:
                print("Nothing new found — dataset already well-covered!")
            else:
                print(f"\nState written to {state_path}")
                if args.dry_run:
                    print("(dry-run: no audio downloaded, CSV not modified)")
            return 0

    if not scheduled:
        print("Nothing scheduled for download.")
        return 0

    # ── Phase 4: download ─────────────────────────────────────────────────────
    print(f"\nPhase 4: downloading {len(scheduled)} audio files …")
    print(f"  Audio root : {audio_root.resolve()}")
    print(f"  Manifest   : {manifest.resolve()}")
    print(f"  Workers    : {args.workers} (adaptive, floor {args.min_workers})")

    ok, skip, fail = download_scheduled(
        scheduled   = scheduled,
        audio_root  = audio_root,
        workers     = args.workers,
        min_workers = args.min_workers,
        timeout     = args.timeout,
        retries     = args.retries,
        manifest_path = manifest,
    )
    print(f"\nDownload complete.  ok={ok}  skipped={skip}  failed={fail}")

    # ── Phase 5: append to CSV ────────────────────────────────────────────────
    # Only append rows that were actually downloaded (or skipped_exists)
    downloaded_ids = set()
    if manifest.is_file():
        with open(manifest, encoding="utf-8") as mf:
            for line in mf:
                try:
                    entry = json.loads(line)
                    if entry.get("status") in ("ok", "skipped_exists"):
                        m = re.search(r"/(\d+)\.", entry.get("path", ""))
                        if m:
                            downloaded_ids.add(int(m.group(1)))
                except json.JSONDecodeError:
                    pass

    rows_to_append = [r for r in scheduled if r["_xc_id"] in downloaded_ids]
    if fail == 0:
        # All succeeded — append everything scheduled
        rows_to_append = scheduled

    if rows_to_append:
        print(f"\nPhase 5: appending {len(rows_to_append)} rows to {csv_path.name} …")
        append_to_csv(csv_path, rows_to_append)
    else:
        print("\nPhase 5: nothing to append (no successful downloads).")

    if fail > 0:
        print(
            f"\nWARNING: {fail} files failed to download. "
            "Re-run (state file preserved) to retry.",
            file=sys.stderr,
        )
        return 2

    print("\nAll done.  Next steps:")
    print("  1. python scripts/build_clap_training_pairs.py   # rebuild train/val pairs")
    print("  2. python scripts/precompute_clap_features.py    # optional: regenerate .clap.pt sidecars")
    print("  3. python scripts/launch_training.py ...         # start the next fine-tune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
