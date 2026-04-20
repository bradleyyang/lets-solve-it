#!/usr/bin/env python3
"""
UTF-8-safe training launcher for Windows PowerShell environments.

Bypasses PowerShell's Out-File / >> encoding issues by having Python
itself write all training output to the log file in UTF-8.

Usage (from repo root):
    python scripts/launch_training.py [extra train_clap.py args...]

Example fresh run:
    python scripts/launch_training.py --checkpoint-dir checkpoints/run6 --workers 4

Example warm-start from previous run:
    python scripts/launch_training.py \
        --checkpoint-dir checkpoints/run7 \
        --finetune-from  checkpoints/run6/best.pt \
        --lr 1e-5 --workers 4

Example full resume (continue same run):
    python scripts/launch_training.py \
        --checkpoint-dir checkpoints/run6 \
        --resume checkpoints/run6/latest.pt

The log file is set via TRAINING_LOG env var (recommended) or defaults to
training.log in the repo root.
"""

from __future__ import annotations

import datetime
import io
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    repo_root = Path(__file__).resolve().parent.parent
    log_path = Path(os.environ.get("TRAINING_LOG", repo_root / "training.log"))

    train_args = sys.argv[1:]  # everything after this script's name

    cmd = [sys.executable, "-u", str(repo_root / "scripts" / "train_clap.py")] + train_args

    banner = f"\n=== RESUMED {datetime.datetime.now().isoformat()} ===\n\n"

    with open(log_path, "a", encoding="utf-8") as log_fh:
        log_fh.write(banner)
        log_fh.flush()

        print(f"[launcher] Logging to: {log_path}", flush=True)
        print(f"[launcher] Command: {' '.join(cmd)}", flush=True)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=repo_root,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
                 "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"},
        )

        assert proc.stdout is not None
        for line in io.TextIOWrapper(proc.stdout.buffer, encoding="utf-8", errors="replace"):
            log_fh.write(line)
            log_fh.flush()
            sys.stdout.write(line)
            sys.stdout.flush()

        proc.wait()

    print(f"\n[launcher] Training exited with code {proc.returncode}", flush=True)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
