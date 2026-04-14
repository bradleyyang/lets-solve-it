"""
CLAP Text Description Generator (RAG + OpenAI API)
===================================================
For each unique (species, vocalization_type) combination in the audio dataset,
generates 4 varied natural language descriptions grounded in species expert text.
These become the text side of CLAP (audio, text) training pairs.

RAG approach: species description is fed as context so the model paraphrases
a verified source rather than generating from memory.

Descriptions are per (species, type) — all clips of the same combo share the
same 4 descriptions. The multi-positive contrastive loss in train_clap.py
handles this correctly by treating all same-combo clips as joint positives,
so there is no false negative problem.

Inputs:
    data/species_descriptions.json  — expert text per species (multi-source)
    data/xc_metadata_unified.csv    — (or any metadata CSV with common_name + vocalization_type)

Output:
    data/clap_descriptions.json  (written after each successful species|type; use --output to change)
    {
        "American Robin||song": [
            "American Robin singing its rich, melodic series of flute-like phrases...",
            ...
        ],
        ...
    }

Usage:
    export OPENAI_API_KEY=sk-...
    conda activate birdclap
    python scripts/generate_clap_descriptions.py
    python scripts/generate_clap_descriptions.py --metadata data/xc_metadata_unified.csv

    Re-running always skips (species, type) keys that already have 4 descriptions
    in the output JSON. Use --no-skip-existing to regenerate everything.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from openai import APIStatusError

AAB_PATH   = Path("data/species_descriptions.json")
OUT_PATH   = Path("data/clap_descriptions.json")
MODEL      = "gpt-4o-mini"   # cheap + fast; swap to "gpt-4o" for higher quality
# Default is a light throttle; 429 retries back off automatically. Use e.g. --delay 21
# if your org is still capped at ~3 RPM on gpt-4o-mini.
DEFAULT_DELAY = 0.35

# Vocalization types that are noisy/uninformative — skip these
SKIP_TYPES = {"uncertain", "various", "various calls", "nan", ""}

SYSTEM_PROMPT = """You are an expert ornithologist writing training descriptions
for a bird audio machine learning model. Your descriptions will be used to train
a model that matches audio recordings to natural language queries.

Rules:
- Base ALL acoustic details strictly on the provided species description.
  Do not add acoustic facts not mentioned in the source text.
- If the source text does not describe this specific vocalization type,
  write: INSUFFICIENT_SOURCE_DATA
- Write in natural, varied language a non-expert could use to search for audio.
- Each description should be 1-2 sentences, 15-40 words.
- Vary vocabulary, sentence structure, and emphasis across descriptions.
- Include: species name, vocalization type, acoustic qualities (pitch/rhythm/timbre),
  and behavioral context where available.
- Do not use Latin names. Do not mention recording quality or equipment."""

USER_TEMPLATE = """Species: {common_name}
Vocalization type: {voc_type}

AllAboutBirds source text:
\"\"\"
{aab_text}
\"\"\"

Generate exactly 4 varied natural language descriptions of a '{voc_type}'
recording of a {common_name}. Return only a JSON array of 4 strings, no other text.
Example format: ["desc1", "desc2", "desc3", "desc4"]"""


def _entry_complete(value: object) -> bool:
    """True if this output key already has a full set of generated strings."""
    if not isinstance(value, list) or len(value) < 4:
        return False
    return all(isinstance(s, str) and s.strip() for s in value)


def get_combos(metadata_paths: list[str]) -> list[tuple[str, str]]:
    """Extract unique (common_name, type) pairs from one or more metadata CSVs."""
    combos = set()
    for path in metadata_paths:
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        if "common_name" not in df.columns:
            print(f"  Skipping {path} — missing common_name column")
            continue
        if "type" not in df.columns and "vocalization_type" in df.columns:
            df = df.rename(columns={"vocalization_type": "type"})
        if "type" not in df.columns:
            print(f"  Skipping {path} — missing type or vocalization_type column")
            continue
        for _, row in df.iterrows():
            name = str(row["common_name"]).strip()
            for t in str(row["type"]).split(","):
                t = t.strip().lower()
                if t not in SKIP_TYPES and name and name != "nan":
                    combos.add((name, t))
    return sorted(combos)


def _wait_seconds_from_429(message: str) -> float:
    m = re.search(r"try again in (\d+)s", str(message), re.I)
    if m:
        return float(m.group(1)) + 2.0
    return 25.0


def _openai_error_code(e: APIStatusError) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("code"):
            return str(err["code"])
        if body.get("code"):
            return str(body["code"])
    if getattr(e, "code", None):
        return str(e.code)
    return ""


def generate_descriptions(client: OpenAI,
                           common_name: str,
                           voc_type: str,
                           aab_text: str,
                           max_retries: int = 12) -> list[str] | None:
    """Call OpenAI to generate 4 descriptions. Retries on RPM rate limits."""
    prompt = USER_TEMPLATE.format(
        common_name=common_name,
        voc_type=voc_type,
        aab_text=aab_text[:2000],
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()

            if "INSUFFICIENT_SOURCE_DATA" in text:
                return None

            start = text.find("[")
            end   = text.rfind("]") + 1
            if start == -1 or end == 0:
                print(f"  [parse error] {common_name} | {voc_type}: {text[:80]}")
                return None
            descs = json.loads(text[start:end])
            if isinstance(descs, list) and all(isinstance(d, str) for d in descs):
                return descs
            return None
        except APIStatusError as e:
            code = _openai_error_code(e)
            if e.status_code == 429 and code == "insufficient_quota":
                print(f"  [API error] {common_name} | {voc_type}: {e}")
                print("\nOpenAI quota/billing — add credits, then re-run (progress is auto-saved).")
                sys.exit(1)
            if e.status_code == 429:
                wait = _wait_seconds_from_429(str(e))
                print(f"  [429 rate limit] waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})…")
                time.sleep(wait)
                continue
            print(f"  [API error] {common_name} | {voc_type}: {e}")
            return None
        except Exception as e:
            print(f"  [API error] {common_name} | {voc_type}: {e}")
            return None
    print(f"  [give up] {common_name} | {voc_type}: too many 429 retries")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", nargs="+",
                        default=["data/xc_metadata_unified.csv"],
                        help="One or more metadata CSVs with common_name and vocalization_type columns")
    parser.add_argument("--output", default=str(OUT_PATH))
    parser.add_argument("--aab", default=str(AAB_PATH),
                        help="Species descriptions JSON")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Regenerate all keys even if already present in output file")
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)  # no-op; resume is default
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"Seconds to sleep after each API call (default: {DEFAULT_DELAY}). "
                             f"Use ~21 if you still hit RPM limits (~3/min tier). Use 0 to disable.")
    args = parser.parse_args()
    delay = max(0.0, args.delay)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: set OPENAI_API_KEY environment variable")

    client   = OpenAI(api_key=api_key)
    raw      = json.loads(Path(args.aab).read_text())
    # Support both old format {"species": "text"} and new {"species": {"text": ..., "source": ...}}
    aab = {k: (v["text"] if isinstance(v, dict) else v) for k, v in raw.items()}
    out_path = Path(args.output)

    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}

    combos = get_combos(args.metadata)
    results = dict(existing)
    already_done = 0
    if not args.no_skip_existing:
        for name, vtype in combos:
            if _entry_complete(results.get(f"{name}||{vtype}")):
                already_done += 1

    print(f"Found {len(combos)} unique (species, type) combos across {len(args.metadata)} metadata file(s)")
    if existing:
        print(f"Loaded {len(existing)} keys from {out_path}")
    if already_done and not args.no_skip_existing:
        print(f"Auto-skip {already_done} combos already complete (4 descriptions each)")
    if args.no_skip_existing:
        print("--no-skip-existing: will regenerate every combo")
    print(f"Post-request delay: {delay}s (429s are retried with backoff)\n")

    skipped = 0

    for i, (name, vtype) in enumerate(combos, 1):
        key = f"{name}||{vtype}"
        if not args.no_skip_existing and _entry_complete(results.get(key)):
            continue

        aab_text = aab.get(name, "")
        if not aab_text:
            print(f"  [{i:3d}/{len(combos)}] {name} | {vtype}: no AAB text — skipping")
            skipped += 1
            continue

        descs = generate_descriptions(client, name, vtype, aab_text)
        if descs:
            results[key] = descs
            print(f"  [{i:3d}/{len(combos)}] {name} | {vtype}: {len(descs)} descriptions")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print(f"  [{i:3d}/{len(combos)}] {name} | {vtype}: insufficient source — skipping")
            skipped += 1

        if delay > 0:
            time.sleep(delay)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nDone.")
    print(f"  Saved:   {len(results)} (species, type) pairs → {out_path}")
    print(f"  Skipped: {skipped} (no AAB text or insufficient source)")
    print(f"\nNext step:")
    print(f"  python scripts/build_clap_labels.py")
    print(f"  python scripts/build_clap_training_pairs.py")


if __name__ == "__main__":
    main()
