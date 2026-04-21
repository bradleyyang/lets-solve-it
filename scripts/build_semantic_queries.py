#!/usr/bin/env python3
"""
Build semantic probe queries from held-out LLM descriptions.

Strips all species-identifying names (common name, scientific name, genus)
from each holdout description so the resulting queries test acoustic
understanding — not species-name matching.

Output: data/semantic_queries.json — ready to pass to:
    python scripts/evaluate_clap.py \
        --semantic-queries data/semantic_queries.json

Usage:
    python scripts/build_semantic_queries.py
    python scripts/build_semantic_queries.py --min-length 60 --max-per-species 3
    python scripts/build_semantic_queries.py --output data/my_queries.json

Why holdout descriptions?
    clap_descriptions_holdout.json contains descriptions that were never seen
    during training. Using them as probe queries gives a genuinely unseen test:
    the model cannot have memorised the phrasing, so any retrieval signal must
    come from acoustic feature alignment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Name stripping
# ─────────────────────────────────────────────────────────────────────────────

def _name_patterns(common_name: str, tax: dict) -> list[tuple[str, str]]:
    """
    Build (regex_pattern, replacement) pairs for all species identifiers.
    Order: longest match first so 'Empidonax virescens' is replaced before
    'Empidonax' alone (avoiding partial replacements).
    """
    candidates: list[str] = []
    for field in ("scientific", "genus", "family"):
        v = tax.get(field, "")
        if v:
            candidates.append(v)
    if common_name:
        candidates.append(common_name)
    # Deduplicate while preserving order
    seen: set[str] = set()
    names: list[str] = []
    for n in candidates:
        if n not in seen:
            seen.add(n); names.append(n)
    # Longest first to avoid partial-match collisions
    names.sort(key=len, reverse=True)

    pairs: list[tuple[str, str]] = []
    for name in names:
        pat = re.escape(name)
        # Possessive: "Acadian Flycatcher's" (singular + plural possessive)
        pairs.append((rf"\b{pat}s?'\s*s\b", "this bird's"))
        pairs.append((rf"\b{pat}'s\b",      "this bird's"))
        # Plural: "American Avocets"
        pairs.append((rf"\b{pat}s\b",       "these birds"))
        # Plain singular
        pairs.append((rf"\b{pat}\b",        "this bird"))
    return pairs


def strip_species_names(text: str, patterns: list[tuple[str, str]]) -> str:
    for pat, repl in patterns:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    # Clean up: "a this bird" / "the this bird" → "this bird"
    text = re.sub(r"\b(a|an|the)\s+this bird", "this bird", text, flags=re.IGNORECASE)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    return text.strip()


def name_still_present(text: str, common_name: str, tax: dict) -> bool:
    """Return True if any species identifier survived stripping."""
    for field in ("scientific", "genus"):
        v = tax.get(field, "")
        if v and re.search(rf"\b{re.escape(v)}\b", text, re.IGNORECASE):
            return True
    if common_name and re.search(rf"\b{re.escape(common_name)}s?\b", text, re.IGNORECASE):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout-descs",   default="data/clap_descriptions_holdout.json",
                    help="Primary source (held-out, unseen during training)")
    ap.add_argument("--main-descs",      default="data/clap_descriptions.json",
                    help="Fallback source for combos missing from holdout")
    ap.add_argument("--taxonomy",        default="data/species_taxonomy.json")
    ap.add_argument("--output",          default="data/semantic_queries.json")
    ap.add_argument("--min-length",      type=int, default=55,
                    help="Min character length of stripped query to keep (default 55)")
    ap.add_argument("--max-per-species", type=int, default=2,
                    help="Max queries per species — avoids over-representing prolific species "
                         "(default 2)")
    ap.add_argument("--include-leaky",   action="store_true",
                    help="Include queries where a species name could not be fully stripped "
                         "(normally excluded)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    holdout_path = repo_root / args.holdout_descs
    main_path    = repo_root / args.main_descs
    tax_path     = repo_root / args.taxonomy
    out_path     = repo_root / args.output

    if not holdout_path.exists():
        sys.exit(f"[error] Not found: {holdout_path}")
    if not tax_path.exists():
        sys.exit(f"[error] Not found: {tax_path}")

    holdout   = json.loads(holdout_path.read_text(encoding="utf-8"))
    main_desc = json.loads(main_path.read_text(encoding="utf-8")) if main_path.exists() else {}
    tax_db    = json.loads(tax_path.read_text(encoding="utf-8"))

    queries: list[dict] = []
    species_count: dict[str, int] = {}

    stats = dict(total=0, skipped_short=0, skipped_leaky=0,
                 skipped_limit=0, used_fallback=0)

    # Merge keys: holdout first, fall back to main_desc for missing combos
    all_keys = sorted(set(holdout) | set(main_desc))

    for key in all_keys:
        if "||" not in key:
            continue
        stats["total"] += 1
        species, voc_type = key.split("||", 1)
        tax      = tax_db.get(species, {})
        patterns = _name_patterns(species, tax)

        # Per-species cap
        if species_count.get(species, 0) >= args.max_per_species:
            stats["skipped_limit"] += 1
            continue

        # Pick description: prefer holdout (unseen), fall back to main
        descs = holdout.get(key) or main_desc.get(key) or []
        if not descs:
            continue
        if key not in holdout:
            stats["used_fallback"] += 1

        raw = descs[0]
        stripped = strip_species_names(raw, patterns)

        if len(stripped) < args.min_length:
            stats["skipped_short"] += 1
            continue

        if name_still_present(stripped, species, tax) and not args.include_leaky:
            stats["skipped_leaky"] += 1
            continue

        queries.append({
            "query":           stripped,
            "positive_combos": [key],
            # Metadata fields — ignored by evaluate_clap.py, useful for review
            "_species":        species,
            "_voc_type":       voc_type,
            "_source":         "holdout" if key in holdout else "main",
        })
        species_count[species] = species_count.get(species, 0) + 1

    queries.sort(key=lambda q: q["positive_combos"][0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(queries, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    n_species = len(species_count)
    print(f"Generated {len(queries)} probe queries across {n_species} species")
    print(f"  From holdout (preferred): "
          f"{sum(1 for q in queries if q['_source'] == 'holdout')}")
    print(f"  From main (fallback):     {stats['used_fallback']}")
    print(f"  Skipped — per-species cap ({args.max_per_species}): {stats['skipped_limit']}")
    print(f"  Skipped — too short (<{args.min_length} chars):     {stats['skipped_short']}")
    print(f"  Skipped — name leaked through stripping:           {stats['skipped_leaky']}")
    print(f"\nSaved -> {out_path}")
    print(f"\nUsage:")
    print(f"  python scripts/evaluate_clap.py \\")
    print(f"      --checkpoint checkpoints/run6/best.pt \\")
    print(f"      --semantic-queries {args.output}")


if __name__ == "__main__":
    main()
