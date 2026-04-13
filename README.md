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
python scripts/train_clap.py
```
Saves checkpoints to `checkpoints/`. See `--help` for batch size, learning rate, etc.

---

## Evaluation

Evaluates retrieval performance on the held-out val set across 6 query strategies:

| Strategy | Example query |
|---|---|
| `name` | `"Northern Cardinal call"` |
| `scientific` | `"Cardinalis cardinalis call"` |
| `chain` | `"Animalia > Chordata > Aves > ... call"` |
| `sci_common` | `"Cardinalis cardinalis, Northern Cardinal call"` |
| `chain_common` | `"Animalia > ... > Cardinalis cardinalis, Northern Cardinal call"` |
| `rich` | LLM-generated acoustic description |

```bash
# Final run checkpoint (epoch 09)
python scripts/evaluate_clap.py --checkpoint checkpoints/latest.pt

# Evaluate a specific epoch snapshot
python scripts/evaluate_clap.py --checkpoint checkpoints/epochs/epoch_09.pt

# Fine-tuned vs base model (zero-shot baseline)
python scripts/evaluate_clap.py --checkpoint checkpoints/latest.pt --also-base

# Base model only (zero-shot)
python scripts/evaluate_clap.py
```

### Outputs

**Console** — metric table (mAP, MRR, R@1, R@5, R@10 per strategy) and delta table if both models evaluated.

**`results/eval_results.json`** — full per-query metrics for every (species, type) combo.

**`results/figures/`** — PDF plots:

| File | Description |
|---|---|
| `strategy_comparison.pdf` | Grouped bar chart of mAP / MRR / R@K per query strategy |
| `recall_at_k.pdf` | R@K curves from K=1→20, one line per strategy |
| `map_distribution_{model}.pdf` | Violin plot of per-query mAP — shows variance across species |
| `class_breakdown_{model}.pdf` | mAP broken down by taxonomic class (Aves, Mammalia, Amphibia, Insecta) |
| `similarity_distribution_{model}.pdf` | Positive vs negative cosine similarity distributions with Δμ |
| `hardest_easiest_{model}.pdf` | Top 20 hardest and easiest species to retrieve |
| `delta_finetuned_vs_base.pdf` | Per-strategy Δmetric (fine-tuned − base), only with `--also-base` |

### Latest completed run (epochs 0-9)

- Training reached the configured 10 epochs (00 through 09).
- Best validation loss remained at epoch 00 (`best.pt`, 1.8329).
- Final checkpoint (`latest.pt` / `epoch_09.pt`) achieved:
  - train loss 0.0073
  - val loss 1.8330
  - R@1 0.0723
- Final retrieval evaluation artifacts:
  - `results/eval_results_epoch09.json`
  - `results/figures_epoch09/`


