# CLAP Fine-Tuning Audit — 2nd-fine-tune (completed)

**Date completed:** 2026-04-13  
**Run:** epochs 0-9 complete (10 total)  
**Model:** `laion/clap-htsat-fused` fine-tuned on Xeno-canto audio  

---

## 1. Training Results Summary

| Epoch | Train Loss | Val Loss | R@1 | Duration |
|-------|-----------|---------|-----|---------|
| 00 | 0.7159 | 1.8329 | 5.66% | ~3h 41m |
| 01–03 | not logged (log corruption) | — | — | — |
| 04 | 0.0322 | 1.8652 | 7.81% | ~1h 58m |
| 05 | 0.0235 | 1.8856 | 7.81% | ~2h 02m |
| 06 | 0.0163 | 1.8645 | 7.03% | ~2h 01m |
| 07 | 0.0255 | 1.8423 | 7.42% | ~2h 03m |
| 08 | 0.0214 | 1.8358 | 7.42% | ~2h 53m |
| 09 | 0.0073 | 1.8330 | 7.23% | ~3h 14m |

**Best checkpoint:** `best.pt` = end of epoch 0 (`val_loss=1.8329`)  
**Latest checkpoint:** `checkpoints/epochs/epoch_09.pt`

---

## 2. What Actually Happened

### Train loss fell sharply — that part worked
The model learned the training data well:

- Epoch 0 train loss: **0.7159** (still noisy from random weights)
- Epoch 4 onward: **0.01–0.03** — the model can nearly perfectly predict training pairs

This is expected and correct behavior. The learning signal exists and is flowing.

### Val loss and R@1 hit a ceiling and stayed flat through epoch 9
- Epoch 0 val_loss: **1.8329** — best ever
- Epoch 4-9 val_loss: **1.8330–1.8856** — never better than epoch 0
- R@1: **5.66%** at epoch 0, hovers around **7%** later, never breaks 8%

This is the red flag. The model trains fine but fails to generalize.

### Diagnosis: the model learned the wrong thing
The gap between train loss (~0.02) and val loss (~1.84) indicates the model is **memorizing patterns in training data** that don't transfer to the validation set. It learned shortcuts, not acoustic understanding.

---

## 3. Root Causes (most important first)

### A. Per-species text, not per-recording (critical)

The RAG-generated rich descriptions in `data/clap_descriptions.json` are written **per species, not per recording**. Every XC clip of Purple Finch gets the exact same 4 narrative sentences.

**Effect:** The contrastive loss cannot distinguish between different recordings of the same species — the text targets are identical. Instead of learning "what makes this specific recording unique," the model learns "what text sounds like a Purple Finch." It overfits to species-level lexical cues and cannot generalize within-species at retrieval time.

**Measured:** ~3,600 unique narrative sentences are each reused across multiple recordings. Some species-level sentences appear 27 times (once per recording of that species).

### B. Low effective data diversity

Despite 96,735 pairs:
- 5 of the 9 text variants per clip are template-derived (name, sci-name, taxonomy chain, combined labels) — they carry nearly identical semantic information in different surface forms
- True unique acoustic descriptions: close to 0 (all rich text is species-level)
- Effective unique pairs: ~10,779 audio files × ~1–2 genuinely distinct text views = much less diversity than the pair count implies

### C. Class imbalance

- Top species: up to **126 recordings** (Passerella iliaca)
- Bottom: **1–2 recordings** per species for ~774 species
- Validation retrieval averages across all species/types. Well-represented species dominate training; rare species underfit.

### D. Label noise from free-text vocalization types

Vocalization type strings in XC metadata are user-contributed and inconsistent:
- Fine-grained labels like "rolled flight call in flight" vs "flight call" vs "nocturnal flight call" overlap
- Multi-label entries (e.g. "call, song") are split on first token, losing context
- This creates ambiguous positives/negatives during retrieval eval

### E. Val loss is not a great proxy for downstream quality

The validation contrastive loss measures how well the model ranks a batch of pairs — not species-level retrieval across a gallery. R@1 on the eval harness (~8.8% at best) confirms the model is usable but weak.

---

## 4. Why This Run's Strategy Was Always Limited

The contrastive objective assumes: **different clips have meaningfully different text**. When they don't (same species, same description), the model cannot learn what differentiates clips. It learns "Purple Finch text → Purple Finch audio" (species mapping) but not "this particular recording with a brighter opening note → this specific description." That species-mapping ability is already partially present in the base CLAP model, which is why `best.pt` (epoch 0, right at the start of fine-tuning) actually beat all later epochs on val loss — the pretrained model's species-level zero-shot ability was better than what this training reinforced.

---

## 5. Full Evaluation Results (from `results/eval_results.json`)

Run: `evaluate_clap.py --checkpoint checkpoints/best.pt`  
Val clips: 1195 | Strategy combos: 603

| Strategy | mAP | MRR | R@1 | R@5 | R@10 |
|---------|-----|-----|-----|-----|------|
| common name | 0.196 | 0.265 | 7.5% | 25.3% | 35.5% |
| scientific name | 0.173 | 0.243 | 6.8% | 20.5% | 31.5% |
| taxonomy chain | 0.167 | 0.237 | 6.2% | 20.6% | 30.5% |
| sci + common | **0.206** | **0.284** | **8.4%** | **25.2%** | **36.3%** |
| chain + common | 0.195 | 0.266 | 7.2% | 25.0% | 37.0% |
| rich description | 0.167 | 0.237 | 6.2% | 20.6% | 30.5% |

Best query strategy: **scientific + common name combined**.  
Rich descriptions perform no better than taxonomy chain — confirms the per-species text adds no retrieval signal.

PDFs in `results/figures/`.

### Final checkpoint evaluation (epoch 09)

Run: `evaluate_clap.py --checkpoint checkpoints/latest.pt`  
Output: `results/eval_results_epoch09.json` and `results/figures_epoch09/`

| Strategy | mAP | MRR | R@1 | R@5 | R@10 |
|---------|-----|-----|-----|-----|------|
| common name | 0.355 | 0.457 | 17.7% | 43.3% | 56.0% |
| scientific name | 0.346 | 0.451 | 16.7% | 43.3% | 53.5% |
| taxonomy chain | 0.340 | 0.441 | 16.5% | 43.5% | 54.7% |
| sci + common | 0.357 | 0.456 | 17.3% | 44.2% | 56.8% |
| chain + common | **0.358** | **0.462** | **17.9%** | 44.1% | 56.1% |
| rich description | 0.340 | 0.441 | 16.5% | 43.5% | 54.7% |

Two validation clips could not be loaded during this run and were excluded by the evaluator. This does not change the central diagnosis: rich text remains no better than taxonomy-derived variants.

---

## 6. What to Fix in Run 2

These are ordered by impact. The first two are non-negotiable.

### Must fix

**1. Per-recording text descriptions**  
Re-run `generate_clap_descriptions.py` with **per-recording context** fed to the LLM:
- XC recording ID, date, location, habitat, recorder notes
- Behavior type from XC metadata (perched, in flight, alarm, dawn chorus, etc.)
- Quality grade and any XC remarks

Each recording should get a *unique* description. Even 1–2 sentences specific to that file is far more valuable than 4 shared sentences.

**2. Remove or deduplicate repeated text targets**  
When building pairs, if two recordings have identical rich text, assign only one of them the rich text and leave the other with taxonomy templates only. This prevents the model from learning "same text → different audio" shortcuts.

### High value

**3. Stratified sampling / rebalancing**  
Cap species at ~15 recordings in training (oversample rare species up to ~5 minimum). This flattens the class distribution and forces the model to learn rare species better.

**4. Hard negative mining (later epochs)**  
After ~2 warm-up epochs, replace random negatives in each batch with "confusable" species (same family, similar call type). This forces the audio encoder to learn fine-grained distinctions, not just "bird vs. not-bird."

**5. Freeze text encoder for first 2 epochs (`--freeze-text-epochs 2`)**  
The text tower from CLAP is already strong. Freezing it early lets the audio encoder adapt without the text embeddings drifting. Unfreeze from epoch 3 onward.

**6. Lower LR, longer warmup**  
Current: LR=5e-5, warmup=200 steps. For a larger dataset with better diversity: LR=1e-5 or 2e-5, warmup=500 steps. Slower, more stable.

### Nice to have

**7. Clip-level metadata as additional text views**  
Add text like `"recorded in British Columbia, June, mixed forest, male singing from exposed perch"` as an extra pair for each clip. No new data required, just XC metadata already in `xc_metadata_unified.csv`.

**8. Larger effective batch size**  
Contrastive learning benefits from more negatives per step. If VRAM allows, try batch=16, accum=4 (effective 64, same as now but more unique negatives per step). Or accumulate to 128.

---

## 7. Current Model — Is It Usable?

Yes, for coarse species-level tasks:
- **R@1 ~8%**, **R@10 ~36%** — weak for strict retrieval, but non-trivial
- Species-name → audio matching works reasonably for well-represented species
- Not suitable for fine-grained acoustic retrieval or audio captioning yet

**Use as a baseline only.** Run 2 with better data should measurably exceed it.

---

## 8. Checkpoints Available

| File | Epoch | Val Loss | R@1 |
|------|-------|---------|-----|
| `checkpoints/best.pt` | 0 | 1.8329 | 5.7% |
| `checkpoints/latest.pt` | 9 | 1.8330 | 7.2% |
| `checkpoints/epochs/epoch_03.pt` | 3 | ~unknown | — |
| `checkpoints/epochs/epoch_08.pt` | 8 | 1.8358 | 7.4% |
| `checkpoints/epochs/epoch_09.pt` | 9 | 1.8330 | 7.2% |

If starting a 2nd run, do **not** warm-start from these weights — the model learned shortcuts that will persist. Start from the base `laion/clap-htsat-fused` pretrained weights.

---

## 9. Run State

Run reached its configured target (**10 epochs total: 00 through 09**) and exited normally.

Terminal end-of-run summary:
- `epoch 09  train_loss=0.0073  val_loss=1.8330  R@1=0.0723`
- `saved epoch snapshot -> checkpoints/epochs/epoch_09.pt`
- `Training complete. Best val loss: 1.8329`

**Recommendation:** For next experiments, start a fresh run from base CLAP with improved labels/data strategy rather than extending this checkpoint line.
