# CLAP Fine-Tuning Audit Report

**Date:** 2026-04-13  
**Model:** `laion/clap-htsat-fused`  
**Run:** 1st-run / lets-solve-it  

---

## 1. Training Script (`scripts/train_clap.py`) — PASS

| Component | Status | Notes |
|-----------|--------|-------|
| Loss function | PASS | Symmetric InfoNCE (CLIP-style). Correct for audio↔text contrastive alignment |
| Embeddings | PASS | L2-normalised `pooler_output` from both towers, scaled by `model.logit_scale_a` |
| Gradient accumulation | PASS | Loss divided by `accum_steps` before backward; optimizer only steps every 8 steps |
| Mixed precision (AMP) | PASS | Correctly scoped with `GradScaler` and `autocast` |
| Gradient clipping | PASS | `max_norm=1.0` — standard practice |
| LR schedule | PASS | Cosine decay with linear warmup (200 steps). Appropriate for fine-tuning |
| R@1 evaluation | PASS | Pools audio and text embeddings over 64 val batches, builds full similarity matrix, checks rank-1. Valid metric |
| Data shuffling | PASS | Train loader uses `shuffle=True`; val loader does not |
| Checkpoint saving | PASS | `best.pt` saved when val loss improves; `latest.pt` every epoch; per-epoch copies in `checkpoints/epochs/` |

**No correctness bugs found in the training pipeline.**

---

## 2. Training Data Audit

### Pair Counts

| Dataset | Pairs | Unique audio files |
|---------|-------|-------------------|
| Train | 96,735 | 10,779 |
| Val | 10,737 | 1,197 |

### Split Quality

| Check | Result |
|-------|--------|
| Audio overlap between train and val | **0** (clean split) |
| Empty or very short texts (< 5 chars) | **0** |
| Suspicious labels (null / unknown / placeholder) | **0** |
| Duplicate (audio, text) pairs in train | **0** |

### Text Variant Breakdown (per pair)

| Type | Count | Description |
|------|-------|-------------|
| Short name only | ~21,548 | `"Canada Goose call"` |
| Taxonomy path | ~21,558 | `"Animalia > ... > Branta canadensis call"` |
| RAG rich narrative | ~42,711 | Multi-sentence audio descriptions |
| Combined (name + scientific) | ~10,889 | `"Branta canadensis, Canada Goose call"` |
| Other | ~29 | Edge cases |

Each audio file has **5–9 text variants** (median 9). No audio file has fewer than 5. No exact (audio, text) duplicates.

**Text length:** min 9 chars, max 241 chars, avg 102 chars. Well-formed.

### Species Coverage

| Metric | Value |
|--------|-------|
| Unique species in training set | ~1,941 |
| Species with ≥ 10 recordings | 614 |
| Species with only 1–2 recordings | 774 |
| Top represented (by recording count) | Passerella iliaca (126), Setophaga coronata (90) |

---

## 3. Known Data Issue — Narrative Text Duplication

The RAG-generated rich descriptions (`generate_clap_descriptions.py`) are generated **per species**, not per recording. This means all recordings of the same species share the same set of narrative sentences.

**Example:** All 27 Purple Finch recordings use the exact same 4 narrative descriptions:
- *"Listen for the Purple Finch's song, a rich, slurred warble..."*
- *"Discover the territory song of the male Purple Finch..."*
- etc.

**Measured impact:**
- ~3,600 unique narrative sentences are each re-used across multiple recordings
- ~1,092 taxonomy-path strings are also repeated (expected — same species, different recordings)

**Effect on training:**
- The contrastive loss can technically distinguish a species from all *other* species, but cannot distinguish *which recording* of that species matches a given narrative
- Effective learning signal per epoch is lower than the raw pair count suggests
- The model will generalise "species name ↔ species audio" well, but individual-recording-level retrieval will be weaker

**Fix (for a future run):** Generate descriptions per recording using each XC file's individual metadata (behavior tag, location, date, recorder comments). This is a separate data pipeline task.

---

## 4. Training Metrics Captured

Only epoch 0 has a confirmed summary line in the log. Epochs 1–3 completed (checkpoints exist) but their summary lines were lost due to a PowerShell `Tee-Object` encoding issue (UTF-16 content appended to UTF-8 file). The issue has been fixed; future epoch summaries will be readable.

| Epoch | Train Loss | Val Loss | R@1 | Duration |
|-------|-----------|---------|-----|---------|
| 0 | 0.7159 | 1.8329 | 5.66% | ~3h 41m |
| 1 | — (log corrupt) | — | — | — |
| 2 | — (log corrupt) | — | — | — |
| 3 | — (log corrupt) | — | — | — |
| 4+ | In progress | — | — | — |

`best.pt` was saved after epoch 0 (val_loss 1.8329). If any of epochs 1–3 improved on this, `best.pt` was overwritten silently (no log line was captured).

---

## 5. Epoch Prediction to Useful Model

Based on dataset characteristics, model architecture, and epoch-0 metrics:

| Epoch range | Expected behaviour |
|-------------|-------------------|
| 0–2 | Rapid early improvement; model adapts pretrained embeddings to bird domain |
| 3–5 | Steady improvement; species-level discrimination developing |
| 6–8 | Slower gains; plateau approaching; **model usable for retrieval** |
| 9–10 | Final refinement; mild overfitting risk on rare species |

**Predicted R@1 at epoch 10: 40–70%**

The upper bound is capped by the narrative duplication issue (same text for all recordings of a species means the model cannot learn audio-specific features from text alone). Without that issue, the ceiling would be higher.

**Minimum viable model:** ~7–8 epochs. At that point R@1 should exceed 50% and species-level audio↔text retrieval should work reliably for well-represented species.

---

## 6. Hyperparameter Assessment

| Setting | Value | Assessment |
|---------|-------|-----------|
| Model | `laion/clap-htsat-fused` | Correct architecture for this task |
| Batch size | 8 × accum 8 = eff. 64 | Reasonable for RTX 4070 (12 GB VRAM) |
| Epochs | 10 | Sufficient given the dataset size |
| LR | 5e-05 | Standard fine-tuning LR; not too aggressive |
| Warmup | 200 steps | Appropriate for this dataset scale |
| Workers | 4 (default) | Increase to 8 for better GPU utilisation |
| Clip duration | 10 s | Matches CLAP pretraining default; appropriate |

---

## 7. Recommendations for This Run

1. **Continue to epoch 10.** The data and script are sound; let training complete.
2. **Increase `--workers 8`** on next resume to reduce CPU bottleneck.
3. **Check `best.pt` epoch** after training completes — it may be from epoch 1–3 (better than epoch 0) despite missing log lines.

## 8. Recommendations for a Future Run (2nd-run)

1. **Per-recording RAG descriptions** — generate narratives using each XC file's individual metadata tags. Biggest quality improvement available.
2. **Pre-compute processor outputs** (mel spectrograms + tokenized text) to disk before training to remove the remaining CPU bottleneck entirely.
3. **`--freeze-text-epochs 2`** — freeze the text encoder for the first 2 epochs so the audio encoder adapts first; can improve early training stability.
4. **`persistent_workers=True` + higher `prefetch_factor`** in `DataLoader` for better GPU feed rate.
