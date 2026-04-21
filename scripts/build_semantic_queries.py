#!/usr/bin/env python3
"""
Build data/semantic_queries.json — natural-language probe texts for semantic retrieval.

Each key is  "common_name||vocalization_type"  (same convention as clap_all_labels.json).
Values are one or more paraphrased queries that describe the sound *without* always
using the exact same wording as taxonomy templates (tests semantic search, not
memorised label matching).

One-time run (~seconds):
    python scripts/build_semantic_queries.py
    python scripts/build_semantic_queries.py --output data/semantic_queries.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def combos_from_val_pairs(val_pairs: list[dict]) -> list[tuple[str, str]]:
    """Unique (common_name, voc_type) from each pair's \"combo\" field (\"Name||type\")."""
    out: set[tuple[str, str]] = set()
    for p in val_pairs:
        raw = p.get("combo") or ""
        if "||" not in raw:
            continue
        name, vtype = raw.split("||", 1)
        name  = name.strip()
        vtype = vtype.strip().lower()
        if name and vtype:
            out.add((name, vtype))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-pairs", default="data/clap_val_pairs.json")
    ap.add_argument("--labels", default="data/clap_all_labels.json")
    ap.add_argument("--taxonomy", default="data/species_taxonomy.json")
    ap.add_argument("--output", default="data/semantic_queries.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = repo_root()
    val_pairs = json.loads((root / args.val_pairs).read_text(encoding="utf-8"))
    all_labels = json.loads((root / args.labels).read_text(encoding="utf-8"))
    tax_db = json.loads((root / args.taxonomy).read_text(encoding="utf-8"))

    val_combos = combos_from_val_pairs(val_pairs)
    if not val_combos:
        raise SystemExit("No \"combo\" fields found in val pairs — check clap_val_pairs.json")
    rng = random.Random(args.seed)

    out: dict[str, list[str]] = {}
    for name, vtype in val_combos:
        key = f"{name}||{vtype}"
        tax = tax_db.get(name, {})
        sci = tax.get("scientific", "")
        variants = all_labels.get(key, [])
        rich = variants[5:] if len(variants) > 5 else []

        probes: list[str] = []

        # Paraphrase templates (surface form differs from standard "name" strategy)
        probes.append(
            f"An acoustic recording capturing {vtype} vocalisations from {name}."
        )
        if sci:
            probes.append(
                f"Sounds typical of {sci} during {vtype}, in the field."
            )
        probes.append(
            f"What does {name} sound like when performing {vtype}? "
            f"Find matching field audio."
        )
        if rich:
            # Second / third rich strings if present — different index than eval "rich"
            idx2 = rng.randint(0, len(rich) - 1)
            probes.append(rich[idx2])
            if len(rich) > 1:
                idx3 = (idx2 + len(rich) // 2) % len(rich)
                if idx3 != idx2:
                    probes.append(rich[idx3])

        # Dedupe while preserving order
        seen_t: set[str] = set()
        uniq = []
        for t in probes:
            t = t.strip()
            if t and t not in seen_t:
                seen_t.add(t)
                uniq.append(t)
        out[key] = uniq

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(out)} combo keys  ->  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
