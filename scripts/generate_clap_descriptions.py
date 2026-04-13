"""
CLAP Per-Recording Description Generator (RAG + OpenAI API)
============================================================
Generates ONE unique natural language description per audio recording.

Previous version generated 4 descriptions per (species, type) combo — meaning
every recording of the same species got the exact same text. That caused the
contrastive loss to learn species-level shortcuts rather than acoustic features.

This version generates one description per recording, with six rotating
description angles so that recordings of the same species receive descriptions
with different emphases (acoustic qualities, behavioral context, field ID cues,
etc.). The angle is assigned deterministically from the XC recording ID so
re-running produces the same assignment.

Output format:
    data/clap_descriptions.json
    {
        "audio/xc/1060250.mp3": "A sharp, chattering alarm call from a North American Red Squirrel...",
        "audio/xc/1020851.mp3": "The howling call of a Coyote, a rising-then-falling wail...",
        ...
    }

Inputs:
    data/species_descriptions.json  — expert text per species (from scrape_species_descriptions.py)
    data/xc_metadata_unified.csv    — one row per recording (filepath, common_name, vocalization_type,
                                      quality_rating, duration)

Usage:
    export OPENAI_API_KEY=sk-...
    conda activate birdclap
    python scripts/generate_clap_descriptions.py

    Re-running skips filepaths already in the output JSON (resumable).
    Use --no-skip-existing to regenerate everything.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI, APIStatusError

# ── paths ─────────────────────────────────────────────────────────────────────
AAB_PATH  = Path("data/species_descriptions.json")
META_PATH = Path("data/xc_metadata_unified.csv")
OUT_PATH  = Path("data/clap_descriptions.json")

MODEL         = "gpt-4o-mini"
DEFAULT_DELAY = 0.35   # seconds between API calls; raise to ~21 if on 3 RPM tier

SKIP_TYPES = frozenset({"uncertain", "various", "various calls", "nan", ""})
SKIP_NAMES = frozenset({
    "soundscape", "identity unknown", "noise", "speech", "canine",
    "squirrel", "insects", "rooster", "other", "background", "unknown", "nan", "",
})

# ── description angles ────────────────────────────────────────────────────────
# Six rotated prompts so recordings of the same species get different emphases.
# Angle is assigned by (xc_recording_id % 6) — deterministic across re-runs.
DESCRIPTION_ANGLES = [
    # 0: acoustic
    ("acoustic",
     "Focus on the acoustic qualities of the sound itself: pitch, rhythm, timbre, "
     "frequency range, and temporal patterning (e.g. rapid trills, slow phrases, "
     "rising or falling contour)."),
    # 1: casual observer
    ("observer",
     "Write from the perspective of a casual nature walker hearing this sound for "
     "the first time — what they would notice and how they would naturally describe it."),
    # 2: behavioral
    ("behavioral",
     "Emphasize what the animal is likely doing and why it makes this sound — "
     "behavioral triggers, social function, and context (alarm, territory, courtship, contact)."),
    # 3: field identification
    ("field_id",
     "Describe what a birder or field naturalist would use to identify this sound — "
     "the most diagnostic acoustic features that distinguish it from similar species."),
    # 4: ecological
    ("ecological",
     "Emphasize ecological and environmental context: when and where this sound is "
     "typically heard — habitat, season, time of day, and who else might be singing nearby."),
    # 5: technical
    ("technical",
     "Use precise descriptive language: note count, syllable structure, frequency contour, "
     "interval patterns, and any distinctive harmonic or tonal qualities."),
]

SYSTEM_PROMPT = """\
You are an expert ornithologist and bioacoustician writing training descriptions
for a bird and wildlife audio machine learning model. Your descriptions will be
used to train a model that matches audio recordings to natural language queries.

Rules:
- Ground ALL acoustic details strictly in the provided source text.
  Do not invent acoustic facts not present in the source.
- If the source text does not describe this specific vocalization type at all,
  write exactly: INSUFFICIENT_SOURCE_DATA
- Write 1–2 natural sentences, 15–40 words total.
- A non-expert should be able to use your description as a search query.
- Include: the species common name, vocalization type, and key acoustic or
  contextual qualities matching the requested angle.
- Do not use Latin names. Do not comment on recording quality or equipment.\
"""

USER_TEMPLATE = """\
Species: {common_name}
Vocalization type: {voc_type}
Recording duration: {duration_desc}
{quality_context}
Description angle: {angle_label}
Angle instruction: {angle_instruction}

Source text:
\"\"\"
{aab_text}
\"\"\"

Write ONE description of a '{voc_type}' recording of a {common_name}.
Follow the angle instruction above. Ground all acoustic details in the source text.
Return only the description, no other text.\
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _xc_id_from_filepath(filepath: str) -> int | None:
    m = re.search(r"/xc/(\d+)\.", str(filepath))
    return int(m.group(1)) if m else None


def _duration_desc(raw: str) -> str:
    """Convert '1:23' or '83.5' seconds into a human-readable string."""
    raw = str(raw).strip()
    try:
        if ":" in raw:
            parts = raw.split(":")
            secs = int(parts[-2]) * 60 + float(parts[-1]) if len(parts) >= 2 else float(parts[0])
        else:
            secs = float(raw)
    except (ValueError, IndexError):
        return "unknown duration"
    if secs < 5:
        return f"very short clip (~{secs:.0f}s)"
    if secs < 30:
        return f"short clip (~{secs:.0f}s)"
    if secs < 120:
        return f"medium-length recording (~{secs:.0f}s)"
    return f"extended recording (~{secs/60:.1f} min)"


def _quality_context(rating) -> str:
    try:
        q = int(float(str(rating)))
    except (ValueError, TypeError):
        return ""
    labels = {5: "excellent (minimal noise)", 4: "good", 3: "average", 2: "poor", 1: "very poor (noisy)"}
    label = labels.get(q, "")
    return f"Recording quality: {q}/5 ({label})" if label else ""


def _wait_from_429(message: str) -> float:
    m = re.search(r"try again in (\d+)s", str(message), re.I)
    return float(m.group(1)) + 2.0 if m else 25.0


def _openai_error_code(e: APIStatusError) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("code"):
            return str(err["code"])
        if body.get("code"):
            return str(body["code"])
    return str(getattr(e, "code", ""))


def generate_description(
    client: OpenAI,
    common_name: str,
    voc_type: str,
    aab_text: str,
    duration_raw: str,
    quality_raw,
    xc_id: int | None,
    max_retries: int = 12,
) -> str | None:
    """Call OpenAI to generate one per-recording description. Returns None on failure."""
    angle_idx = (xc_id % len(DESCRIPTION_ANGLES)) if xc_id is not None else 0
    angle_label, angle_instruction = DESCRIPTION_ANGLES[angle_idx]

    prompt = USER_TEMPLATE.format(
        common_name=common_name,
        voc_type=voc_type,
        duration_desc=_duration_desc(duration_raw),
        quality_context=_quality_context(quality_raw),
        angle_label=angle_label,
        angle_instruction=angle_instruction,
        aab_text=aab_text[:2000],
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=128,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
            if "INSUFFICIENT_SOURCE_DATA" in text:
                return None
            return text
        except APIStatusError as e:
            code = _openai_error_code(e)
            if e.status_code == 429 and code == "insufficient_quota":
                print(f"\nOpenAI quota exhausted — add credits, then re-run (progress auto-saved).")
                sys.exit(1)
            if e.status_code == 429:
                wait = _wait_from_429(str(e))
                print(f"  [429] waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})…")
                time.sleep(wait)
                continue
            print(f"  [API error] {e}")
            return None
        except Exception as e:
            print(f"  [error] {e}")
            return None

    print(f"  [give up] {common_name} | {voc_type}: too many 429 retries")
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(META_PATH),
                        help="Metadata CSV with filepath, common_name, vocalization_type columns")
    parser.add_argument("--aab",      default=str(AAB_PATH),
                        help="Species descriptions JSON (from scrape_species_descriptions.py)")
    parser.add_argument("--output",   default=str(OUT_PATH))
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Regenerate descriptions even if filepath already in output")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"Seconds between API calls (default {DEFAULT_DELAY}). "
                             f"Use ~21 if on 3 RPM tier.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N recordings (0 = all). Useful for testing.")
    args = parser.parse_args()
    delay = max(0.0, args.delay)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: set OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key)

    # Load species descriptions
    raw_aab = json.loads(Path(args.aab).read_text(encoding="utf-8"))
    aab = {k: (v["text"] if isinstance(v, dict) else v) for k, v in raw_aab.items()}
    print(f"Loaded AAB text for {len(aab)} species from {args.aab}")

    # Load existing output for resuming
    out_path = Path(args.output)
    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}
    print(f"Loaded {len(existing)} existing entries from {out_path}" if existing else
          f"Starting fresh — no existing output at {out_path}")

    # Load metadata
    meta_path = Path(args.metadata)
    if not meta_path.exists():
        raise SystemExit(f"ERROR: metadata not found at {meta_path}")
    df = pd.read_csv(meta_path, encoding="utf-8")

    # Normalise column names
    name_col  = next((c for c in ("common_name", "species", "name") if c in df.columns), None)
    type_col  = next((c for c in ("vocalization_type", "type") if c in df.columns), None)
    path_col  = next((c for c in ("filepath", "file_path", "path", "filename") if c in df.columns), None)
    qual_col  = next((c for c in ("quality_rating", "quality") if c in df.columns), None)
    dur_col   = next((c for c in ("duration",) if c in df.columns), None)

    if not name_col or not type_col or not path_col:
        raise SystemExit(f"ERROR: metadata missing required columns. Found: {list(df.columns)}")

    # Build list of recordings to process
    records = []
    for _, row in df.iterrows():
        name  = str(row[name_col]).strip()
        fpath = str(row[path_col]).strip()
        voc_raw = str(row[type_col]).strip()
        voc_type = voc_raw.split(",")[0].strip().lower()

        if name.lower() in SKIP_NAMES or not fpath or fpath == "nan":
            continue
        if voc_type in SKIP_TYPES:
            continue
        if not aab.get(name):
            continue  # no source text — can't generate

        records.append({
            "filepath": fpath,
            "common_name": name,
            "voc_type": voc_type,
            "quality": row[qual_col] if qual_col else None,
            "duration": str(row[dur_col]) if dur_col else "",
            "xc_id": _xc_id_from_filepath(fpath),
        })

    if args.limit > 0:
        records = records[:args.limit]

    # Count what needs to be done
    if not args.no_skip_existing:
        pending = [r for r in records if r["filepath"] not in existing]
    else:
        pending = records

    already_done = len(records) - len(pending)
    print(f"\nTotal recordings in metadata : {len(records)}")
    print(f"Already generated            : {already_done}")
    print(f"To generate                  : {len(pending)}")
    print(f"Post-request delay           : {delay}s\n")

    results = dict(existing)
    skipped = 0

    for i, rec in enumerate(pending, 1):
        aab_text = aab.get(rec["common_name"], "")
        if not aab_text:
            skipped += 1
            continue

        desc = generate_description(
            client,
            common_name=rec["common_name"],
            voc_type=rec["voc_type"],
            aab_text=aab_text,
            duration_raw=rec["duration"],
            quality_raw=rec["quality"],
            xc_id=rec["xc_id"],
        )

        if desc:
            results[rec["filepath"]] = desc
            angle_idx = (rec["xc_id"] % len(DESCRIPTION_ANGLES)) if rec["xc_id"] else 0
            angle_label = DESCRIPTION_ANGLES[angle_idx][0]
            print(f"  [{i:5d}/{len(pending)}] [{angle_label:10s}] {rec['common_name']} | {rec['voc_type']}")
        else:
            skipped += 1
            print(f"  [{i:5d}/{len(pending)}] [skip      ] {rec['common_name']} | {rec['voc_type']}: no source data")

        # Save after every entry (crash-safe)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        if delay > 0:
            time.sleep(delay)

    print(f"\nDone.")
    print(f"  Total entries saved : {len(results)} → {out_path}")
    print(f"  Skipped             : {skipped} (no source text or insufficient data)")
    print(f"\nAngle distribution across recordings:")
    from collections import Counter
    angle_counts = Counter()
    for r in records:
        if r["xc_id"] is not None:
            angle_counts[DESCRIPTION_ANGLES[r["xc_id"] % len(DESCRIPTION_ANGLES)][0]] += 1
    for label, count in sorted(angle_counts.items()):
        print(f"  {label:<12}: {count}")
    print(f"\nNext step:")
    print(f"  python scripts/build_clap_labels.py")
    print(f"  python scripts/build_clap_training_pairs.py")


if __name__ == "__main__":
    main()
