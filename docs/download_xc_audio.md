# `download_xc_audio.py` — Bulk Xeno-canto Audio Downloader

Downloads all MP3 recordings listed in `xc_metadata_unified.csv` from
[xeno-canto.org](https://xeno-canto.org) and stores them in a directory tree
that matches the `audio` paths in `clap_train_pairs.json`.

**No API key required.** Downloads use plain HTTP GETs to
`https://xeno-canto.org/<id>/download`.

See also: [`download_xc_first_half.py`](download_xc_audio.md#two-machine-split)
and [`download_xc_second_half.py`](download_xc_audio.md#two-machine-split).

---

## Quick start

```bash
# from repo root (lets-solve-it/)
python scripts/download_xc_audio.py
```

Downloads all ~17 k recordings to `scripts/data/xc_audio/audio/xc/`.  
Files already present (> 512 bytes) are **skipped** automatically — safe to
interrupt and re-run.

---

## Output layout

```
scripts/data/xc_audio/          ← audio root (gitignored)
  audio/
    xc/
      1060250.mp3
      1020851.mp3
      …
  download_manifest.jsonl       ← one JSON line per attempted file
```

The `audio_root` value to pass to `train_clap.py` is
`scripts/data/xc_audio`.

---

## CLI reference

```
python scripts/download_xc_audio.py [OPTIONS]

--csv               Path to xc_metadata_unified.csv    default: auto-resolved
--out-dir           Root directory for audio tree       default: scripts/data/xc_audio
--workers           Concurrent download threads         default: 30
--min-workers       Adaptive floor on concurrency       default: 4
--sleep             Seconds to pause after each success default: 0.0
--timeout           Per-request timeout (seconds)       default: 120
--retries           Attempts per file on error          default: 4
--limit             Max files to download (0 = all)     default: 0
--dry-run           Print plan, do not download         flag
--no-skip-existing  Re-download even if file exists     flag
--no-adaptive       Disable auto worker reduction       flag
--429-window SEC    Sliding window for 429 counting     default: 12.0
--429-burst N       429s in window to trigger reduction default: 5
--manifest          JSONL log path                      default: OUT_DIR/download_manifest.jsonl
--shard-index N     0-based shard (use with --num-shards)
--num-shards K      Split CSV into K contiguous slices
```

---

## Adaptive concurrency

By default the script starts **30 concurrent threads**.  If Xeno-canto returns
**≥ 5 HTTP 429 responses in a 12-second window**, the thread cap is **halved**
(e.g. 30 → 15 → 8 → 4) and the 429 counter is cleared.  The cap never goes
below `--min-workers` (default 4).

To disable:

```bash
python scripts/download_xc_audio.py --no-adaptive --workers 10
```

If the server starts responding with 429s frequently, add a small sleep:

```bash
python scripts/download_xc_audio.py --sleep 0.35
```

---

## Manifest log

Every attempted file appends one JSON line to `download_manifest.jsonl`:

```json
{"xc_id": 1060250, "path": "…/audio/xc/1060250.mp3", "status": "ok", "ts": "2026-04-11T16:00:02Z"}
{"xc_id": 1020851, "path": "…/audio/xc/1020851.mp3", "status": "skipped_exists", "ts": "…"}
{"xc_id": 9999999, "path": "…", "status": "error", "detail": "404 Client Error", "ts": "…"}
```

Use it to audit failed downloads or resume counts.

---

## Two-machine split

Use the wrapper scripts to share the download work across two networks:

```bash
# Developer A (first half — rows 0..8893)
python scripts/download_xc_first_half.py

# Developer B (second half — rows 8893..17787)
python scripts/download_xc_second_half.py
```

Both write the **same directory layout** under their own `--out-dir`.  
Merge by copying one `audio/xc/` tree into the other (no ID conflicts).

Any flag accepted by `download_xc_audio.py` is also accepted by the wrapper
scripts and is forwarded automatically, e.g.:

```bash
python scripts/download_xc_first_half.py --out-dir D:/xc_mirror --sleep 0.2
```

Splitting only helps if the two developers are on **different networks**
(different IPs); two simultaneous jobs from the same IP double the load on
Xeno-canto without proportional speed benefit.

---

## Smoke test

```bash
python scripts/download_xc_audio.py --limit 5 --dry-run
python scripts/download_xc_audio.py --limit 5
```

---

## Etiquette

Xeno-canto recordings are free for non-commercial research use.
See [xeno-canto.org/help/terms-of-use](https://xeno-canto.org/help/terms-of-use).
The script uses an identifying `User-Agent` and the adaptive 429 logic to stay
within reasonable bounds; please don't bypass these safeguards.
