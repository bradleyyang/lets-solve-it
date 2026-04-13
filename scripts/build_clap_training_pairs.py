"""
CLAP Training Pairs Builder
============================
Converts audio metadata + clap_all_labels.json + clap_descriptions.json into
the training JSON expected by LAION-CLAP and most CLAP fine-tuning frameworks.

Each audio file is paired with:
  1. ALL taxonomy template variants from clap_all_labels.json (5 per clip)
  2. Its own UNIQUE per-recording description from clap_descriptions.json (1 per clip)

This means a clip with taxonomy + a rich description appears 6 times in
training data, each with a different text. Crucially, the rich description is
UNIQUE to that recording — two clips of the same species get different text —
so the contrastive loss cannot shortcut by mapping identical text to many audios.

Output format (LAION-CLAP compatible):
    data/clap_train_pairs.json   — list of {audio, text} dicts
    data/clap_val_pairs.json     — held-out 10% for validation

    [
        {"audio": "audio/xc/1060250.mp3", "text": "American Robin song"},
        {"audio": "audio/xc/1060250.mp3", "text": "Turdus migratorius song"},
        {"audio": "audio/xc/1060250.mp3", "text": "Animalia > Chordata > Aves > ... song"},
        {"audio": "audio/xc/1060250.mp3", "text": "Turdus migratorius, American Robin song"},
        {"audio": "audio/xc/1060250.mp3", "text": "Animalia > ... > Turdus migratorius, American Robin song"},
        {"audio": "audio/xc/1060250.mp3", "text": "American Robin singing its rich flute-like phrases..."},
        ...
    ]

Balanced sampling: caps at MAX_PER_COMBO clips per (species, type) combo
so common species don't dominate (follows AnimalCLAP's 30-clips-per-species
strategy). The cap applies to unique audio clips; each clip still generates
one pair per text variant.

Usage:
    conda activate birdclap
    python scripts/build_clap_training_pairs.py
    python scripts/build_clap_training_pairs.py --max-per-combo 30 --val-split 0.1
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
UNIFIED_META  = Path("data/xc_metadata_unified.csv")
LABELS_PATH   = Path("data/clap_all_labels.json")
REC_DESC_PATH = Path("data/clap_descriptions.json")   # per-recording descriptions
TRAIN_OUT     = Path("data/clap_train_pairs.json")
VAL_OUT       = Path("data/clap_val_pairs.json")

SKIP_NAMES = frozenset({
    "soundscape", "identity unknown", "noise", "speech", "canine", "squirrel",
    "insects", "rooster", "other", "background", "unknown", "nan", "",
})
SKIP_TYPES = frozenset({"uncertain", "various", "various calls", "nan", ""})


def build_pairs(metadata_paths: list[str],
                labels: dict,
                rec_descs: dict,
                max_per_combo: int,
                seed: int) -> tuple[list[dict], list[tuple[str, str]]]:
    """
    For each audio file:
      - Look up its (species, voc_type) key in labels → emit one pair per taxonomy template
      - Look up its filepath in rec_descs → emit one pair for its unique per-recording description

    The cap (max_per_combo) applies to unique audio clips per combo.
    Returns (all_pairs, accepted_clips).
    """
    rng = random.Random(seed)
    accepted: list[tuple[str, str]] = []
    combo_counts: dict[str, int] = {}

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

            voc_type = voc_raw.split(",")[0].strip().lower()
            if voc_type in SKIP_TYPES:
                continue

            key = f"{name}||{voc_type}"
            if key not in labels:
                key_full = f"{name}||{voc_raw.lower()}"
                if key_full not in labels:
                    continue
                key = key_full

            combo_counts[key] = combo_counts.get(key, 0) + 1
            if combo_counts[key] > max_per_combo:
                continue

            accepted.append((fpath, key))

    rng.shuffle(accepted)

    # Expand: taxonomy templates + unique per-recording description (if available)
    all_pairs: list[dict] = []
    n_with_desc = 0
    for fpath, key in accepted:
        for text in labels[key]:
            all_pairs.append({"audio": fpath, "text": text})
        desc = rec_descs.get(fpath)
        if desc:
            all_pairs.append({"audio": fpath, "text": desc})
            n_with_desc += 1

    print(f"  Clips with per-recording description : {n_with_desc}/{len(accepted)} "
          f"({100*n_with_desc/max(len(accepted),1):.1f}%)")
    return all_pairs, accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata",  nargs="+", default=[str(UNIFIED_META)])
    parser.add_argument("--labels",    default=str(LABELS_PATH))
    parser.add_argument("--rec-descs", default=str(REC_DESC_PATH),
                        help="Per-recording descriptions JSON (from generate_clap_descriptions.py)")
    parser.add_argument("--train-out", default=str(TRAIN_OUT))
    parser.add_argument("--val-out",   default=str(VAL_OUT))
    parser.add_argument("--max-per-combo", type=int, default=30,
                        help="Max audio clips per (species, type) combo (default 30)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of clips held out for validation (default 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = json.loads(Path(args.labels).read_text())
    print(f"Loaded {len(labels)} label keys from {args.labels}")

    rec_descs: dict = {}
    rec_desc_path = Path(args.rec_descs)
    if rec_desc_path.exists():
        rec_descs = json.loads(rec_desc_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(rec_descs)} per-recording descriptions from {rec_desc_path}")
    else:
        print(f"No per-recording descriptions found at {rec_desc_path} — taxonomy templates only")

    all_pairs, accepted_clips = build_pairs(args.metadata, labels, rec_descs, args.max_per_combo, args.seed)

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
