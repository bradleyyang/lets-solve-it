#!/usr/bin/env python3
"""
Pre-convert Xeno-Canto MP3s to 48 kHz mono float32 WAV, centre-cropped /
zero-padded to exactly CLIP_DURATION_S seconds (same logic as train_clap.py).

The resulting .wav files sit NEXT TO their .mp3 counterparts:
    scripts/data/xc_audio/audio/xc/1234567.mp3   (original, untouched)
    scripts/data/xc_audio/audio/xc/1234567.wav   (pre-clipped, fast to load)

train_clap.py (updated) will auto-detect the .wav and skip MP3 decoding.

Disk: each WAV is 480 000 samples × 4 bytes = ~1.84 MB.  For ~17 k files
      expect ~32 GB of new .wav files alongside the existing ~62 GB of .mp3s.

Usage (from repo root):
    python scripts/convert_to_wav.py                       # default audio-root
    python scripts/convert_to_wav.py --audio-root D:/path  # custom root
    python scripts/convert_to_wav.py --workers 8           # parallel threads
    python scripts/convert_to_wav.py --dry-run             # report only

Re-running always skips files already converted (unless --force).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    import librosa
except ImportError:
    print("Missing librosa.  Run: pip install -r requirements-ml.txt", file=sys.stderr)
    raise

# Must match train_clap.py constants exactly.
TARGET_SR       = 48_000
CLIP_DURATION_S = 10.0
MIN_DURATION_S  = 0.5


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def convert_one(mp3_path: Path, force: bool) -> tuple[str, str]:
    """
    Convert a single MP3 to a pre-clipped WAV.
    Returns (mp3_path_str, status) where status is 'ok', 'skip', or 'error:<msg>'.
    """
    wav_path = mp3_path.with_suffix(".wav")

    if wav_path.is_file() and not force:
        return str(mp3_path), "skip"

    try:
        y, _ = librosa.load(str(mp3_path), sr=TARGET_SR, mono=True)
    except Exception as exc:
        return str(mp3_path), f"error:librosa:{exc}"

    target_len = int(CLIP_DURATION_S * TARGET_SR)
    min_len    = int(MIN_DURATION_S   * TARGET_SR)

    if len(y) < min_len:
        return str(mp3_path), "error:too_short"

    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    y = y.astype(np.float32)

    try:
        sf.write(str(wav_path), y, TARGET_SR, subtype="FLOAT")
    except Exception as exc:
        return str(mp3_path), f"error:write:{exc}"

    return str(mp3_path), "ok"


def main() -> int:
    root = repo_root()

    ap = argparse.ArgumentParser(
        description="Pre-convert XC MP3s to 48 kHz float32 WAV for faster training."
    )
    ap.add_argument(
        "--audio-root",
        default=str(root / "scripts" / "data" / "xc_audio"),
        help="Root that contains audio/xc/*.mp3 (default: scripts/data/xc_audio)",
    )
    ap.add_argument(
        "--workers", type=int, default=4,
        help="Parallel threads for conversion (default 4; I/O-bound, try 6-8)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-convert even if .wav already exists",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print stats but do not write any files",
    )
    args = ap.parse_args()

    xc_dir = Path(args.audio_root) / "audio" / "xc"
    if not xc_dir.is_dir():
        print(f"ERROR: {xc_dir} does not exist.", file=sys.stderr)
        return 1

    mp3_files = sorted(xc_dir.glob("*.mp3"))
    total = len(mp3_files)
    if total == 0:
        print("No .mp3 files found.")
        return 0

    already = sum(1 for f in mp3_files if f.with_suffix(".wav").is_file())
    to_do   = total if args.force else (total - already)

    print(f"Audio root : {xc_dir}")
    print(f"MP3 files  : {total:,}")
    print(f"WAV exists : {already:,}")
    print(f"To convert : {to_do:,}")
    print(f"Workers    : {args.workers}")
    print(f"Target     : {TARGET_SR} Hz  |  {CLIP_DURATION_S} s  |  float32 WAV")
    print(f"Est. disk  : ~{to_do * 1.84 / 1024:.1f} GB of new WAV files\n")

    if args.dry_run:
        print("Dry-run mode — no files written.")
        return 0

    n_ok = n_skip = n_err = 0
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(convert_one, f, args.force): f
            for f in mp3_files
        }
        pbar = tqdm(as_completed(futures), total=total, unit="file", dynamic_ncols=True)
        for fut in pbar:
            _, status = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                errors.append(status)
            pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err)

    print(f"\nDone.  converted={n_ok:,}  skipped={n_skip:,}  errors={n_err:,}")
    if errors:
        print(f"\nFirst 10 errors:")
        for e in errors[:10]:
            print(f"  {e}")

    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
