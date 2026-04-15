"""
CLAP Training Pairs Builder
============================
Converts audio metadata + clap_all_labels.json into the training JSON
expected by LAION-CLAP and most CLAP fine-tuning frameworks.

Each audio file is paired with ALL text variants from its label pool
(5 taxonomy templates + up to 4 rich descriptions = ~9 variants per clip).
Each pair also carries a "combo" field (species||type) so the training
script can build a multi-positive mask — clips from the same combo are
treated as joint positives, avoiding false negatives in the contrastive loss.

Filters applied:
  - Birds only (taxonomic class == Aves); mammals, amphibians, insects removed.
  - Combos with fewer than MIN_CLIPS_PER_COMBO unique clips are dropped —
    they cannot contribute to multi-positive contrastive learning.

Output format (LAION-CLAP compatible):
    data/clap_train_pairs.json   — list of {audio, text, combo} dicts
    data/clap_val_pairs.json     — held-out 10% for validation

    [
        {"audio": "audio/xc/1060250.mp3", "text": "American Robin song",          "combo": "American Robin||song"},
        {"audio": "audio/xc/1060250.mp3", "text": "Turdus migratorius song",       "combo": "American Robin||song"},
        {"audio": "audio/xc/1060250.mp3", "text": "Animalia > ... song",           "combo": "American Robin||song"},
        {"audio": "audio/xc/1060250.mp3", "text": "A cheerful series of whistles", "combo": "American Robin||song"},
        ...
    ]

Balanced sampling: caps at MAX_PER_COMBO clips per (species, type) combo
so common species don't dominate. The cap applies to unique audio clips;
each clip still generates one pair per text variant.

Usage:
    conda activate birdclap
    python scripts/build_clap_training_pairs.py
    python scripts/build_clap_training_pairs.py --max-per-combo 50 --min-clips-per-combo 3 --val-split 0.1
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
UNIFIED_META  = Path("data/xc_metadata_unified.csv")
LABELS_PATH   = Path("data/clap_all_labels.json")
TAXONOMY_PATH = Path("data/species_taxonomy.json")
TRAIN_OUT     = Path("data/clap_train_pairs.json")
VAL_OUT       = Path("data/clap_val_pairs.json")

SKIP_NAMES = frozenset({
    "soundscape", "identity unknown", "noise", "speech", "canine", "squirrel",
    "insects", "rooster", "other", "background", "unknown", "nan", "",
})
SKIP_TYPES = frozenset({"uncertain", "various", "various calls", "nan", ""})

MIN_CLIPS_PER_COMBO = 3   # combos below this threshold add no multi-positive signal


def build_pairs(metadata_paths: list[str],
                labels: dict,
                tax_db: dict,
                max_per_combo: int,
                min_clips_per_combo: int,
                seed: int) -> tuple[list[dict], list[tuple[str, str]]]:
    """
    For each audio file, look up its (species, voc_type) key in labels and
    emit one {audio, text, combo} pair per text variant.

    Filters:
      - Birds only: species whose taxonomic class != 'Aves' are skipped.
      - min_clips_per_combo: combos with fewer unique clips are dropped after
        the first pass (they add no multi-positive signal).
      - max_per_combo cap: applied per combo to limit class imbalance.

    Returns (all_pairs, accepted_clips).
    """
    rng = random.Random(seed)

    # First pass: collect all valid clips per combo (respecting cap)
    combo_clips: dict[str, list[str]] = {}

    for path in metadata_paths:
        p = Path(path)
        if not p.exists():
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_csv(p)

        name_col = next((c for c in ("common_name", "species", "name") if c in df.columns), None)
        type_col = next((c for c in ("vocalization_type", "type") if c in df.columns), None)
        path_col = next((c for c in ("filepath", "file_path", "path", "filename") if c in df.columns), None)

        if not name_col or not type_col or not path_col:
            print(f"  [skip] {path} — missing required columns "
                  f"(need name, type, filepath). Found: {list(df.columns)}")
            continue

        for _, row in df.iterrows():
            name    = str(row[name_col]).strip()
            voc_raw = str(row[type_col]).strip()
            fpath   = str(row[path_col]).strip()

            if name.lower() in SKIP_NAMES or not fpath or fpath == "nan":
                continue

            # Birds only
            tax_class = tax_db.get(name, {}).get("class", "")
            if tax_class and tax_class != "Aves":
                continue

            voc_type = voc_raw.split(",")[0].strip().lower()
            if voc_type in SKIP_TYPES:
                continue

            key = f"{name}||{voc_type}"
            if key not in labels:
                key_full = f"{name}||{voc_raw.lower()}"
                if key_full not in labels:
                    continue
                key = key_full

            clips = combo_clips.setdefault(key, [])
            if len(clips) < max_per_combo:
                clips.append(fpath)

    # Second pass: drop combos below the minimum clip threshold
    dropped_combos = sum(1 for c in combo_clips.values() if len(c) < min_clips_per_combo)
    print(f"  Combos before min-clip filter : {len(combo_clips)}")
    print(f"  Combos dropped (<{min_clips_per_combo} clips)  : {dropped_combos}")
    combo_clips = {k: v for k, v in combo_clips.items() if len(v) >= min_clips_per_combo}
    print(f"  Combos after filter           : {len(combo_clips)}")

    accepted: list[tuple[str, str]] = [
        (fpath, key)
        for key, clips in combo_clips.items()
        for fpath in clips
    ]
    rng.shuffle(accepted)

    # Expand each clip into one pair per text variant.
    # "combo" field lets train_clap.py build a multi-positive mask so clips
    # from the same species+type are not penalised as negatives of each other.
    all_pairs: list[dict] = []
    for fpath, key in accepted:
        for text in labels[key]:
            all_pairs.append({"audio": fpath, "text": text, "combo": key})

    return all_pairs, accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata",  nargs="+", default=[str(UNIFIED_META)])
    parser.add_argument("--labels",    default=str(LABELS_PATH))
    parser.add_argument("--taxonomy",  default=str(TAXONOMY_PATH))
    parser.add_argument("--train-out", default=str(TRAIN_OUT))
    parser.add_argument("--val-out",   default=str(VAL_OUT))
    parser.add_argument("--max-per-combo", type=int, default=50,
                        help="Max audio clips per (species, type) combo (default 50)")
    parser.add_argument("--min-clips-per-combo", type=int, default=MIN_CLIPS_PER_COMBO,
                        help=f"Drop combos with fewer than this many clips (default {MIN_CLIPS_PER_COMBO})")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of clips held out for validation (default 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = json.loads(Path(args.labels).read_text())
    print(f"Loaded {len(labels)} label keys from {args.labels}")

    tax_db: dict = {}
    tax_path = Path(args.taxonomy)
    if tax_path.exists():
        tax_db = json.loads(tax_path.read_text(encoding="utf-8"))
        print(f"Loaded taxonomy for {len(tax_db)} species (birds-only filter active)")
    else:
        print(f"[warn] Taxonomy file not found at {tax_path} — bird filter disabled")

    all_pairs, accepted_clips = build_pairs(
        args.metadata, labels, tax_db,
        args.max_per_combo, args.min_clips_per_combo, args.seed,
    )

    n_clips   = len(accepted_clips)
    n_val_clips = max(1, int(n_clips * args.val_split))
    n_train_clips = n_clips - n_val_clips

    print(f"Total unique clips: {n_clips}  "
          f"(train: {n_train_clips}, val: {n_val_clips})")

    # Split at the clip level, then expand to pairs
    # accepted_clips is already shuffled; first n_val_clips go to val
    val_clip_set = set(fpath for fpath, _ in accepted_clips[:n_val_clips])

    train = [p for p in all_pairs if p["audio"] not in val_clip_set]
    val   = [p for p in all_pairs if p["audio"]     in val_clip_set]

    train_path = Path(args.train_out)
    val_path   = Path(args.val_out)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(json.dumps(train, indent=2, ensure_ascii=False))
    val_path.write_text(json.dumps(val,   indent=2, ensure_ascii=False))

    print(f"\n{'─'*60}")
    print(f"Done.")
    print(f"  Train : {len(train)} pairs ({n_train_clips} clips × avg "
          f"{len(train)/n_train_clips:.1f} variants) → {train_path}")
    print(f"  Val   : {len(val)} pairs ({n_val_clips} clips × avg "
          f"{len(val)/n_val_clips:.1f} variants) → {val_path}")

    # Text length stats
    from collections import Counter
    text_lengths = [len(p["text"].split()) for p in train]
    print(f"\n  Text length (words): "
          f"min={min(text_lengths)}  max={max(text_lengths)}  "
          f"avg={sum(text_lengths)/len(text_lengths):.1f}")

    # Show 3 sample clips (all variants for each)
    rng = random.Random(args.seed)
    sample_clips = rng.sample(list(set(p["audio"] for p in train)), min(3, n_train_clips))
    print(f"\n  Sample clips (all variants shown):")
    for clip in sample_clips:
        clip_pairs = [p for p in train if p["audio"] == clip]
        print(f"    audio: {clip}")
        for p in clip_pairs:
            print(f"      text: {p['text'][:80]}")
        print()

    print(f"Next step:")
    print(f"  Fine-tune CLAP with {train_path} and {val_path}")


if __name__ == "__main__":
    main()
