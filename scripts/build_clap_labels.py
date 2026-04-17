"""
CLAP Label Builder (Taxonomy + Rich Descriptions)
==================================================
Generates AnimalCLAP-style multi-template text labels for each unique
(species, vocalization_type) combination.

Five taxonomy templates per the AnimalCLAP paper (ICASSP 2026):
  1. Common name + voc_type          → "American Robin song"
  2. Scientific name + voc_type      → "Turdus migratorius song"
  3. Taxonomy chain + voc_type       → "Aves > Passeriformes > Turdidae > ... song"
  4. Scientific + common + voc_type  → "Turdus migratorius, American Robin song"
  5. Taxonomy + common + voc_type    → "Aves > ... > Turdus migratorius, American Robin song"

If data/clap_descriptions.json exists (from generate_clap_descriptions.py),
the RAG-generated rich descriptions are appended as additional variants.
These are per-species+type (not per-recording) — all clips of the same combo
share the same descriptions. The multi-positive contrastive loss in train_clap.py
handles this correctly by treating all same-combo clips as joint positives.

Held-out descriptions (--holdout-rich N, default 1):
  The last N rich descriptions per combo are withheld from training labels and
  saved separately to data/clap_descriptions_holdout.json. The evaluation script
  uses these as "unseen acoustic queries" — text the model never trained on —
  to measure whether it genuinely learned acoustic language understanding.

Output:
    data/clap_all_labels.json          — training labels (5 taxonomy + 3 rich)
    data/clap_descriptions_holdout.json — held-out descriptions for eval only

Usage:
    conda activate birdclap
    python scripts/build_clap_labels.py
    python scripts/build_clap_labels.py --metadata data/unified_metadata.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
TAXONOMY_DB      = Path("data/species_taxonomy.json")
RICH_DESC_PATH   = Path("data/clap_descriptions.json")
OUT_PATH         = Path("data/clap_all_labels.json")
HOLDOUT_OUT_PATH = Path("data/clap_descriptions_holdout.json")
HOLDOUT_RICH_COUNT = 1   # hold back last N rich descriptions for eval

SKIP_TYPES    = {"uncertain", "various", "various calls", "nan", ""}
SKIP_NAMES    = {"identity unknown", "soundscape", "noise", "speech",
                 "canine", "squirrel", "insects", "rooster", "other",
                 "background", "unknown", "nan", ""}
BIRD_CLASS    = "Aves"   # only generate labels for birds


# ── label template builder ────────────────────────────────────────────────────

def build_templates(common_name: str, tax: dict, voc_type: str) -> list[str]:
    """
    Generate AnimalCLAP-style text templates (broad-to-narrow is critical).
    Works for any taxon — non-birds get their real Kingdom/Class (e.g. Mammalia).
    """
    sci    = tax.get("scientific", "")
    kingdom= tax.get("kingdom", "Animalia")
    phylum = tax.get("phylum",  "Chordata")
    cls    = tax.get("class",   "")
    order  = tax.get("order",   "")
    family = tax.get("family",  "")
    genus  = tax.get("genus",   "")

    chain_parts = [p for p in [kingdom, phylum, cls, order, family, genus, sci] if p]
    chain = " > ".join(chain_parts)

    vt = voc_type.strip()

    templates = [f"{common_name} {vt}"]                          # 1: common + type
    if sci:
        templates += [
            f"{sci} {vt}",                                       # 2: scientific + type
            f"{chain} {vt}",                                     # 3: full chain + type
            f"{sci}, {common_name} {vt}",                        # 4: sci + common + type
            f"{chain}, {common_name} {vt}",                      # 5: full chain + common + type
        ]
    return templates


# ── metadata reader ───────────────────────────────────────────────────────────

def get_combos(metadata_paths: list[str]) -> list[tuple[str, str]]:
    combos = set()
    for path in metadata_paths:
        p = Path(path)
        if not p.exists():
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_csv(p, encoding="utf-8")
        # Accept either 'type' or 'vocalization_type' column
        name_col = next((c for c in ("common_name", "species", "name") if c in df.columns), None)
        type_col = next((c for c in ("vocalization_type", "type") if c in df.columns), None)
        if name_col is None or type_col is None:
            print(f"  [skip] {path} missing name/type columns (found: {list(df.columns)})")
            continue
        for _, row in df.iterrows():
            name = str(row[name_col]).strip()
            if not name or name.lower() in SKIP_NAMES:
                continue
            for t in str(row[type_col]).split(","):
                t = t.strip().lower()
                if t not in SKIP_TYPES:
                    combos.add((name, t))
    return sorted(combos)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", nargs="+",
        default=["data/xc_metadata_unified.csv"],
        help="Metadata CSVs with common_name and vocalization_type columns",
    )
    parser.add_argument("--output", default=str(OUT_PATH))
    parser.add_argument("--holdout-rich", type=int, default=HOLDOUT_RICH_COUNT,
                        help=f"Hold back last N rich descriptions from training labels "
                             f"and save to clap_descriptions_holdout.json (default {HOLDOUT_RICH_COUNT})")
    args = parser.parse_args()

    # Load taxonomy DB (built by build_taxonomy_db.py — no network calls needed)
    if not TAXONOMY_DB.exists():
        raise SystemExit(f"ERROR: {TAXONOMY_DB} not found. Run build_taxonomy_db.py first.")
    tax_db: dict = json.loads(TAXONOMY_DB.read_text(encoding="utf-8"))
    print(f"Loaded taxonomy for {len(tax_db)} species from {TAXONOMY_DB}")

    # Load rich descriptions if available (from generate_clap_descriptions.py)
    rich_descs: dict = {}
    if RICH_DESC_PATH.exists():
        rich_descs = json.loads(RICH_DESC_PATH.read_text(encoding="utf-8"))
        print(f"Loaded {len(rich_descs)} rich description keys from {RICH_DESC_PATH}")
    else:
        print(f"No rich descriptions at {RICH_DESC_PATH} — taxonomy templates only")

    # Get all (species, vocalization_type) combos from metadata
    combos = get_combos(args.metadata)
    print(f"\nFound {len(combos)} unique (species, type) combos\n")

    results  = {}
    holdout  = {}
    no_tax   = 0
    skipped_non_bird = 0

    for i, (name, vtype) in enumerate(combos, 1):
        key = f"{name}||{vtype}"
        tax = tax_db.get(name, {})

        # Skip non-bird species (mammals, amphibians, insects, unknowns)
        tax_class = tax.get("class", "")
        if tax_class and tax_class != BIRD_CLASS:
            skipped_non_bird += 1
            continue

        if not tax.get("scientific"):
            no_tax += 1

        templates = build_templates(name, tax, vtype)
        rich      = rich_descs.get(key, [])

        # Split rich descriptions: train on first (N - holdout), hold back last N
        n_holdout = args.holdout_rich if len(rich) > args.holdout_rich else 0
        rich_train   = rich[:len(rich) - n_holdout] if n_holdout else rich
        rich_holdout = rich[len(rich) - n_holdout:] if n_holdout else []

        results[key] = templates + rich_train
        if rich_holdout:
            holdout[key] = rich_holdout

        status = f"{len(templates)} templates + {len(rich_train)} rich (holdout: {len(rich_holdout)})"
        print(f"  [{i:4d}/{len(combos)}] {name} | {vtype}: {status}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    holdout_path = HOLDOUT_OUT_PATH
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path.write_text(
        json.dumps(holdout, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    print(f"\n{'-'*60}")
    print(f"Done.")
    print(f"  Labels written  : {len(results)} (species, type) pairs -> {out_path}")
    print(f"  Holdout written : {len(holdout)} combos -> {holdout_path}")
    print(f"  Skipped non-bird: {skipped_non_bird} combos (class != Aves)")
    print(f"  No taxonomy     : {no_tax} species (common name only template)")
    counts = [len(v) for v in results.values()]
    if counts:
        print(f"  Variants/combo  : min={min(counts)}  max={max(counts)}  avg={sum(counts)/len(counts):.1f}")
    print(f"\nNext step:")
    print(f"  python scripts/build_clap_training_pairs.py --labels {out_path}")


if __name__ == "__main__":
    main()
