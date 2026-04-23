# Codebase guide — `lets-solve-it`

This document is for **new teammates** and for **LLM-assisted coding**: it states what the repo is, what is stable, what to avoid breaking, and sensible next steps.

---

## 1. What this project is (product vs repo)

**Product direction (team vision)**  
Use **audio** (especially birds and wildlife) together with **text** so users can **search or retrieve clips** using natural language — often discussed in terms of **joint audio–text embeddings** (e.g. CLAP-style models), separate from classic **species-only classifiers**.

**What this repository actually implements today**  
A **data pipeline + fine-tuning** workspace **and** a **client-only web prototype** for exploring UX flows:

1. **Fetch metadata** from [Xeno-canto](https://xeno-canto.org/) API v3 for a fixed query (\cnt:canada\) and export a **unified CSV**.
2. **Explore** that CSV (counts, missing fields, bird vs non-bird heuristics) in a notebook.
3. **Bulk-download** ~17 k audio recordings with adaptive concurrency and resume support.
4. **Build training labels** using multi-template taxonomy strings and RAG-enriched descriptions.
5. **Fine-tune CLAP** (\laion/clap-htsat-fused\) on 96 k \(audio, text)\ pairs with contrastive loss, AMP, and gradient accumulation.
6. **`web/`** — Vite + React SPA: mock search/classification, saved list, compare slots, and a **Three.js** temporal embedding visualization (see **`web/docs/README.md`**).

There is **no production API server** in this repo; the web app runs entirely in the browser with mocked data except for optional **Web Audio** decoding of user uploads.

---

## 2. Repository layout

| Path | Purpose |
|------|--------|
| `README.md` | Minimal setup: venv, `pip install`, env check, optional CLAP test |
| `requirements.txt` | Pinned **core** stack (pandas, requests, jupyter, …) |
| `requirements-ml.txt` | PyTorch / `transformers` / librosa for CLAP training and inference |
| `scripts/get_xenocanto.ipynb` | **Source of truth** for building `xc_metadata_unified.csv` from API v3 |
| `scripts/eda_xc_metadata.ipynb` | EDA on the CSV — taxon mix, vocalization labels, missing fields |
| `scripts/check_environment.py` | Validates Python, imports, CSV presence, `.env` / `XC_API_KEY`, live API ping |
| `scripts/mini_clap_xc_sample.py` | End-to-end **sample**: CSV → download MP3s → HF CLAP similarities |
| `scripts/download_xc_audio.py` | **Bulk downloader** — ~17 k MP3s, 30 threads, adaptive 429 handling, resume ([docs](docs/download_xc_audio.md)) |
| `scripts/download_xc_first_half.py` | Wrapper — shard 1/2 for two-machine parallel download |
| `scripts/download_xc_second_half.py` | Wrapper — shard 2/2 for two-machine parallel download |
| `scripts/train_clap.py` | **Fine-tunes CLAP** with contrastive loss, AMP, grad accumulation, checkpointing ([docs](docs/train_clap.md)) |
| `scripts/build_clap_labels.py` | Builds `data/clap_all_labels.json` — 5 taxonomy templates + rich descriptions per `(species, type)` |
| `scripts/build_clap_training_pairs.py` | Expands labels → `data/clap_train_pairs.json` / `data/clap_val_pairs.json` |
| `scripts/generate_clap_descriptions.py` | GPT-generated rich text descriptions → `data/clap_descriptions.json` |
| `scripts/build_taxonomy_db.py` | GBIF taxonomy lookup → `data/species_taxonomy.json` |
| `scripts/xc_metadata_unified.csv` | **Committed artifact** (~18k rows) — full metadata export |
| `scripts/data/` | **Gitignored** — audio downloads (`xc_audio/`), mini sample (`xc_mini/`) |
| `data/clap_train_pairs.json` | **96 k** `{audio, text}` training pairs |
| `data/clap_val_pairs.json` | **10 k** `{audio, text}` validation pairs |
| `data/clap_all_labels.json` | Label pool — 2 739 `(species, type)` keys → list of text variants |
| `data/species_taxonomy.json` | Cached GBIF taxonomy per species |
| `data/clap_descriptions.json` | RAG-generated rich descriptions per species |
| `docs/` | Extended documentation for new scripts |
| `docs/train_clap.md` | Full reference for `train_clap.py` |
| `docs/download_xc_audio.md` | Full reference for the bulk downloader scripts |
| `checkpoints/` | **Gitignored** — `best.pt` and `latest.pt` written by `train_clap.py` |
| `web/` | **React + Vite** prototype UI — mock catalog, uploads, 3D viz; onboarding in **`web/docs/`** |
| `web/docs/README.md` | Index to web feature + development docs |
| `web/docs/FEATURES.md` | Full feature map of the SPA (routes, state, mocks, viz pipeline) |
| `web/docs/DEVELOPMENT.md` | Run/build, aliases, extension notes for web |

**Notebooks may write CSVs relative to the Jupyter **current working directory** (often `scripts/` if the server was started there). The fetch notebook saves `xc_metadata_unified.csv` next to the CWD; `check_environment.py` and `mini_clap_xc_sample.py` look in **`scripts/` first**, then repo root.

---

## 3. Data contract (do not break without updating all consumers)

The unified metadata file is **`xc_metadata_unified.csv`** with header:

```text
filepath,species_code,common_name,vocalization_type,quality_rating,duration,source
```

| Column | Meaning | Constraints |
|--------|---------|--------------|
| `filepath` | Logical path / filename pattern | **Must** match `audio/xc/<numeric_id>.<ext>` so scripts can parse Xeno-canto **recording id** (regex in `mini_clap_xc_sample.py`: `audio/xc/(\d+)\.`) |
| `species_code` | Slug (often from English name) | Used for grouping; may be empty for some rows |
| `common_name` | Human-readable name | `mini_clap` **drops rows** missing `filepath`, `species_code`, or `common_name` |
| `vocalization_type` | Call/song/etc. | Free text; may contain commas (quoted in CSV) |
| `quality_rating` | Numeric (mapped from XC letter grades in notebook) | Integer-ish |
| `duration` | String like `M:SS` | Not normalized to seconds in CSV |
| `source` | Provenance | Currently `xeno-canto` |

**If you add or rename columns**, update:

- `scripts/get_xenocanto.ipynb` (`clean_to_unified_schema` or equivalent),
- `scripts/mini_clap_xc_sample.py` (column reads and `build_label`),
- `scripts/eda_xc_metadata.ipynb` (any hard-coded column lists),
- This guide.

---

## 4. What is done vs not done

### Done (working in repo)

- **API v3 ingestion** with `XC_API_KEY`, pagination, unified schema → CSV (`get_xenocanto.ipynb`).
- **Large CSV** (~18 k rows) for Canada-tagged recordings — includes birds and non-birds.
- **Environment verification** (`check_environment.py`) including optional HF CLAP processor download (`--with-ml`).
- **EDA notebook** for distribution and data-quality questions.
- **CLAP smoke test** using **Transformers** (`laion/clap-htsat-fused`).
- **Bulk audio downloader** (`download_xc_audio.py`) — 30 concurrent threads, adaptive 429 handling, resume, two-machine sharding.
- **~17 k MP3s downloaded** (~62 GB) to `scripts/data/xc_audio/`.
- **Multi-template label builder** (`build_clap_labels.py`) — 5 taxonomy templates + rich descriptions per `(species, type)`.
- **Training pair builder** (`build_clap_training_pairs.py`) — 96 k train / 10 k val `(audio, text)` pairs.
- **CLAP fine-tuning script** (`train_clap.py`) — symmetric InfoNCE loss, AMP, gradient accumulation, cosine LR, checkpointing, val R@1 metric.
- **WAV pre-conversion path** (`convert_to_wav.py` + `train_clap.py` fast-path) — pre-clipped 48 kHz WAV siblings can be used to skip MP3 decode overhead.
- **Evaluation harness** (`evaluate_clap.py`) — retrieval metrics + figures in `results/`.
- **Run audit artifact** (`TRAINING_AUDIT.md`) — postmortem and run-2 recommendations.

### Not done (out of scope or future work)

- Bird-only filtering (explicit taxon rules or allowlist).
- **Backend API** wired to real embeddings / Xeno-canto (the `web/` app is mock-first; swap `web/src/api/mock.ts` when ready).
- Evaluation harness / retrieval benchmark beyond R@1.
- Multi-GPU / distributed training.
- LoRA / PEFT (currently full fine-tune only).

### Current fine-tune status (second fine-tune completed)

- A second fine-tune run completed 10 epochs (00-09) in `checkpoints/second-fine-tune/`.
- Training curve highlights:
  - epoch 02: train 0.3773, val 0.6744, R@1 0.3320
  - epoch 07: train 0.0200, val **0.6520**, R@1 0.4501 (best val checkpoint)
  - epoch 09: train 0.0321, val 0.6736, R@1 0.4471 (latest checkpoint)
- Best checkpoint: `checkpoints/second-fine-tune/best.pt` (epoch 07).
- Latest checkpoint: `checkpoints/second-fine-tune/latest.pt` (epoch 09).
- Fresh retrieval eval (best checkpoint vs base model) is in:
  - `results/eval_results_second_fine_tune_best.json`
  - `results/figures_second_fine_tune_best/`
- Run audit is documented in:
  - `TRAINING_AUDIT_SECOND_FINE_TUNE.md`

---

## 5. How to run (quick reference)

From repo root (`lets-solve-it/`):

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

Create **`.env`** in repo root (gitignored):

```env
XC_API_KEY=your_key_here
```

Verify:

```bash
python scripts/check_environment.py
python scripts/check_environment.py --with-ml   # after: pip install -r requirements-ml.txt
```

Optional CLAP smoke test (needs **ffmpeg** on PATH for MP3 on many systems):

```bash
pip install -r requirements-ml.txt
python scripts/mini_clap_xc_sample.py --sample 6
```

Bulk audio download (first run takes hours; safe to interrupt and resume):

```bash
python scripts/download_xc_audio.py          # full corpus, 30 threads, adaptive throttle
python scripts/download_xc_first_half.py     # half for developer A
python scripts/download_xc_second_half.py    # half for developer B (different network)
```

Fine-tune CLAP (requires GPU + ffmpeg + ~62 GB audio already downloaded):

```bash
python scripts/train_clap.py                            # default settings
python scripts/train_clap.py --batch-size 4 --accum 16 # low VRAM
python scripts/train_clap.py --resume checkpoints/latest.pt
```

See `docs/train_clap.md` and `docs/download_xc_audio.md` for full CLI reference.

Jupyter: open `scripts/get_xenocanto.ipynb` or `scripts/eda_xc_metadata.ipynb` (ensure kernel uses the same venv).

---

## 6. Pitfalls (read before you change code)

### Secrets and sharing

- **Never commit** `.env` or API keys. **Never paste keys** in Discord/slack screenshots.
- Each developer should use their **own** Xeno-canto key where possible.

### API and scraping etiquette

- download_xc_audio.py uses **adaptive concurrency** (default 30 threads, backs off to 4 on 429 bursts). Do not disable the adaptive logic or use extreme concurrency without monitoring for 429 responses.
- The mini sample script (mini_clap_xc_sample.py) sleeps ~0.35 s between downloads; keep that for its small-scale usage.
- Use a **identifying User-Agent** when adding new HTTP clients (see `HEADERS` in `mini_clap_xc_sample.py`).
- `check_environment.py` performs a **minimal** API call when `XC_API_KEY` is set; avoid wrapping it in tight CI loops that hammer Xeno-canto.

### Regenerating CSV

- Re-running `get_xenocanto.ipynb` **overwrites** `xc_metadata_unified.csv` in the notebook’s CWD. Commit diffs deliberately; row counts and ordering can change as Xeno-canto grows.

### Notebook working directory

- EDA notebook tries `xc_metadata_unified.csv` then `scripts/xc_metadata_unified.csv`. If paths fail, set `CSV_PATH` explicitly or start Jupyter from a consistent folder.

### ML stack size and GPU

- `requirements-ml.txt` pulls **PyTorch** and **transformers**; first CLAP run **downloads large weights**. CI without GPU should skip `--with-ml` or cache models.
- CUDA is optional; CPU runs work but are slower.
- On Windows, prefer UTF-8 logging for long runs (`PYTHONUTF8=1` and `Out-File -Encoding utf8`) to avoid mixed-encoding progress logs.

### Why the current training quality is poor (important)

- The richest text labels are generated **per species**, not per recording, so many different audio clips share identical rich descriptions.
- Contrastive learning then learns species-level text shortcuts instead of clip-level acoustic distinctions.
- Net result: train loss drops hard, but validation loss and retrieval metrics plateau (memorization > generalization).
- See `TRAINING_AUDIT.md` for full evidence, metrics, and remediation plan.

### `filepath` format

- Downstream code **depends** on `audio/xc/<id>.<ext>`. If you change the pattern, **update** `xc_id_from_row` and any download URLs (`https://xeno-canto.org/{id}/download`).

### Two CLAP stacks in the wild

- This repo’s **supported** path is **Hugging Face** `ClapModel` / `ClapProcessor` in `mini_clap_xc_sample.py`.
- Older snippets may use **`laion_clap`**; mixing both in one environment can cause confusion — prefer one stack per branch unless you document why.

---

## 7. Conventions for writing code that does not break the repo

1. **Treat the CSV schema as a public API** — version or migrate columns explicitly (see §3).
2. **Keep scripts runnable from repo root** with paths derived from `Path(__file__)` (see `repo_root()` in existing scripts).
3. **Large or downloaded data** goes under `scripts/data/` (gitignored) or a path passed via CLI flags — **do not** commit MP3s or model weights.
4. **Pin or bound** new dependencies: add to `requirements.txt` or `requirements-ml.txt` with a short comment if optional.
5. **Extend `check_environment.py`** when you add mandatory imports or services (so newcomers fail fast with a clear message).
6. **Notebooks for exploration**; **promote** stable logic to `.py` modules if multiple entry points need it (future refactor).

---

## 8. Suggested next steps (prioritized)

1. **Regenerate rich descriptions per recording** (not per species) using clip-specific metadata (date/location/habitat/notes/behavior).
2. **Deduplicate repeated text targets** in pair building so multiple clips do not share identical rich labels.
3. **Rebalance species distribution** (cap dominant species, upsample rare species) before rebuilding train/val pairs.
4. **Start run-2 from base pretrained CLAP**, not from current fine-tuned checkpoints (to avoid carrying over shortcut biases).
5. **Tune optimization for stability** (lower LR, longer warmup, optional early text-encoder freeze).
6. **Keep eval-first cadence**: after each epoch, compare mAP/MRR/R@K using `evaluate_clap.py`, not train loss alone.

---

## 9. Who to ask

- **Data semantics** (what a column means, whether to drop soundscapes): team + EDA notebook conclusions.
- **Xeno-canto policy / keys**: [xeno-canto.org](https://xeno-canto.org/) account and API docs.
- **Model choice** (HF checkpoint vs fine-tune): team ML lead / mentor.

This guide should be **updated** when the CSV schema, default queries, or primary CLAP stack changes.

---

## 10. Web client (`web/`) — pointer for ML / pipeline developers

If you only touch Python and data: you can ignore `web/` until you need to demo retrieval UX. When you do:

- **Onboarding:** start at [`web/docs/README.md`](web/docs/README.md) → [`FEATURES.md`](web/docs/FEATURES.md).
- **Run locally:** `cd web && npm install && npm run dev`.
- **Contract:** UI types live in `web/src/api/types.ts`; mock responses in `web/src/api/mock.ts`. Keeping names/fields aligned with a future REST or gRPC API will reduce UI churn.

Web documentation belongs under **`web/docs/`**; this root guide stays focused on data + training scripts.
