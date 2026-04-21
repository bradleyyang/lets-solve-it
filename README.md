New to the repo? Read **[CODEBASE_GUIDE.md](CODEBASE_GUIDE.md)** for architecture, data contracts, pitfalls, and next steps.

## Setup

1. Create the virtual environment
```bash
python3 -m venv .venv
```

2. Activate it

On Windows:
```bash
.\.venv\Scripts\activate
```

On macOS/Linux:
```bash
source .venv/bin/activate
```

3. Install requirements
```bash
pip install -r requirements.txt
```

4. Verify setup (after adding `.env` with `XC_API_KEY` for API v3)
```bash
python scripts/check_environment.py
python scripts/check_environment.py --with-ml
```

5. Optional: CLAP smoke test (installs PyTorch stack; MP3 decoding usually needs [ffmpeg](https://ffmpeg.org/download.html) on PATH)
```bash
pip install -r requirements-ml.txt
python scripts/mini_clap_xc_sample.py --sample 6
```

---

## CLAP Labelling Pipeline

Builds AnimalCLAP-style (audio, text) training pairs for fine-tuning CLAP on bird and wildlife audio.

### Step 1 — Scrape species descriptions
Fetches acoustic descriptions from AllAboutBirds (birds) and Animal Diversity Web (non-birds).
```bash
python scripts/scrape_species_descriptions.py
```
Output: `data/species_descriptions.json`

### Step 2 — Build taxonomy database
Looks up full Kingdom→Species taxonomy for every species via the GBIF API.
```bash
python scripts/build_taxonomy_db.py
```
Output: `data/species_taxonomy.json`

### Step 3 — Generate rich text descriptions
Uses OpenAI GPT to generate 4 varied acoustic descriptions per (species, vocalization type) combo,
grounded in the scraped species text (RAG pipeline). Requires `OPENAI_API_KEY`.
```bash
export OPENAI_API_KEY=sk-...
python scripts/generate_clap_descriptions.py
```
Output: `data/clap_descriptions.json` — 4 descriptions per (species, type) combo

### Step 4 — Build label variants
Combines 5 AnimalCLAP taxonomy templates with the rich descriptions into a single label pool
per (species, type) combo (~9 variants each).
```bash
python scripts/build_clap_labels.py
```
Output: `data/clap_all_labels.json`

### Step 5 — Build training pairs
Pairs every audio clip with all its text variants (clip-level 90/10 train/val split).
```bash
python scripts/build_clap_training_pairs.py
```
Outputs:
- `data/clap_train_pairs.json` — training set, each audio × all text variants
- `data/clap_val_pairs.json`   — held-out val set (no audio overlap with train)

### Step 6 — Fine-tune CLAP

```bash
# Fresh run
python scripts/launch_training.py --checkpoint-dir checkpoints/run7 --workers 4

# Warm-start from previous run (reset optimizer, new LR)
python scripts/launch_training.py \
    --checkpoint-dir checkpoints/run7 \
    --finetune-from  checkpoints/run6/best.pt \
    --lr 1e-5 --workers 4

# Full resume (continue same run)
python scripts/launch_training.py \
    --checkpoint-dir checkpoints/run6 \
    --resume checkpoints/run6/latest.pt
```

Key flags: `--batch-size` (default 16), `--epochs` (default 20), `--lr` (default 2e-5), `--workers`.  
Run `python scripts/train_clap.py --help` for the full list.

---

## Evaluation

Evaluates text-to-audio retrieval on the held-out val set across 8 query strategies:

| Strategy | Example query |
|---|---|
| `name` | `"Northern Cardinal call"` |
| `scientific` | `"Cardinalis cardinalis call"` |
| `chain` | `"Animalia > Chordata > Aves > ... call"` |
| `sci_common` | `"Cardinalis cardinalis, Northern Cardinal call"` |
| `chain_common` | `"Animalia > ... > Cardinalis cardinalis, Northern Cardinal call"` |
| `rich` | LLM-generated acoustic description (one variant) |
| `rich_holdout` | Held-out description not seen during training |
| `all_variants` | Ensemble of all 8 label variants per combo (free inference improvement) |

```bash
# Fine-tuned checkpoint
python scripts/evaluate_clap.py --checkpoint checkpoints/run6/best.pt

# Fine-tuned vs base model (zero-shot baseline)
python scripts/evaluate_clap.py --checkpoint checkpoints/run6/best.pt --also-base

# Base model only (zero-shot)
python scripts/evaluate_clap.py

# Full semantic eval suite (requires hand-written query file)
python scripts/evaluate_clap.py --checkpoint checkpoints/run6/best.pt \
    --semantic-queries data/semantic_queries_example.json \
    --acoustic-coherence \
    --cross-species-eval
```

### Outputs

**Console** — metric table (mAP, MRR, R@1, R@5, R@10, MedRank per strategy) plus delta table when both models evaluated.

**`results/eval_results.json`** — per-query metrics for every (species, type) combo + semantic eval summaries.

**`results/figures/`** — PDF plots (5 core + up to 3 semantic):

| File | Description |
|---|---|
| `strategy_comparison.pdf` | mAP / MRR / R@1/5/10 bar chart per strategy |
| `class_breakdown_{model}.pdf` | mAP by taxonomic class × strategy |
| `hardest_easiest_{model}.pdf` | Top-20 hardest and easiest species |
| `rank_cdf_{model}.pdf` | CDF of first-hit rank across the full gallery |
| `delta_finetuned_vs_base.pdf` | Δmetric (fine-tuned − base), only with `--also-base` |
| `semantic_probe_{model}.pdf` | R@K per hand-written acoustic query (`--semantic-queries`) |
| `acoustic_coherence_{model}.pdf` | Genus/family neighbour recall in text-embedding space (`--acoustic-coherence`) |
| `cross_species_transfer_{model}.pdf` | Description transfer R@K by genus (`--cross-species-eval`) |

### Semantic search evaluations

Three opt-in evaluations probe whether the model learned *acoustic* semantics rather than species identity:

- **`--semantic-queries data/semantic_queries_example.json`** — hand-written purely-acoustic queries (no species names) evaluated against the full gallery. Edit the example file to add real positive combos for your data.
- **`--acoustic-coherence`** — encodes one rich acoustic description per combo and checks whether top-K nearest neighbours in text-embedding space share the same genus/family. Genus R@K >> random → model learned acoustic structure.
- **`--cross-species-eval`** — uses species A's acoustic description to retrieve species B's audio (same genus, different species). Above-random R@K → transferable acoustic features learned.

### Run history

| Run | Epochs | Notes |
|---|---|---|
| run 1–2 | — | Initial experiments |
| run 3–4 | — | Dataset and label improvements |
| run 5 | 20 | Low LR (audio encoder effectively frozen at 5e-7) |
| run 6 | ongoing | Differential LR, WeightedRandomSampler, SpecAugment, hard negative boosting, mixup, ensemble eval, logit_scale freeze, ParetoCheckpointManager |


