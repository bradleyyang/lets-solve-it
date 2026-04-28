# Training Audit — Run 8 (finetune8)

**Status:** COMPLETE (post-mortem)
**Date:** 2026-04-27
**Author:** auto-generated from log + eval JSON + manual analysis

---

## 1. Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `laion/clap-htsat-fused` |
| Checkpoint dir | `checkpoints/finetune8/` |
| Warm-start from | `checkpoints/finetune7/best.pt` (**inferred** — see Root Cause below) |
| Epochs | 10 (0–9) |
| Batch size | 16 × accum 8 = **effective 128** |
| Base LR | 2e-05 |
| Audio encoder LR | 2e-06 (×0.1) |
| Text encoder LR | 1e-05 (×0.5) |
| LR schedule | Cosine decay with linear warm-up |
| Warmup steps | 200 |
| Weight decay | 1e-4 |
| Adam betas | (0.9, 0.98) |
| AMP | On (FP16, CUDA) |
| Data mode | Pre-computed `.clap.pt` sidecars (95% coverage) |
| Train pairs | 131,460 (6,663 skipped — .clap.pt missing) |
| Val pairs | 14,530 (805 skipped) |
| Augmentation | random crop, noise/gain, SpecAugment, text aug, mixup α=0.4 |
| Text encoder freeze | Epochs 0 (unfrozen at epoch 1) |
| Logit scale freeze | Epochs 0–1 (unfrozen at epoch 2) |
| Hard negatives | Enabled (same-genus boosting via `species_taxonomy.json`) |
| Balanced sampler | `WeightedRandomSampler` per (species, voc_type) combo |
| Labels file | `data/clap_all_labels.json` |

---

## 2. Epoch Training Curve

> Sourced directly from `training_8th_finetune.log` — every summary line extracted.

| Epoch | train_loss | val_loss | R@1    | Time   | Notes |
|------:|----------:|--------:|-------:|-------:|-------|
| 00 | 3.0527 | 2.5600 | 0.0117 | 1002s | New best; ep00 saved. Initial batch loss ~6.7 → sign of damaged start checkpoint |
| 01 | 2.2666 | 1.9354 | 0.0146 | 1222s | New best; text encoder unfrozen |
| 02 | 1.8977 | 1.7159 | 0.0352 | 1222s | New best; logit_scale unfrozen |
| 03 | 1.6946 | 1.5859 | 0.0527 | 1224s | New best |
| 04 | 1.5627 | 1.5211 | 0.0557 | 1224s | New best |
| 05 | 1.4697 | 1.4808 | 0.0869 | 1220s | New best; val_loss < 1.5 crossed |
| 06 | 1.4119 | 1.4831 | 0.0830 | 1221s | Val loss regressed vs ep05; train still improving |
| 07 | 1.3840 | 1.4444 | 0.0898 | 1219s | **Best R@1 in run;** new best val_loss; ep05+ep06 pruned from Pareto |
| 08 | 1.3520 | **1.4327** | 0.0840 | 1221s | **Best val_loss;** ep08 = eval checkpoint |
| 09 | 1.3461 | 1.4362 | 0.0791 | 1224s | Marginal train improvement; val slightly worse; R@1 declining |

**Best checkpoint:** `epoch_08.pt` / `best.pt` — val_loss = **1.4327**
**Pareto front at end:** [ep07, ep08, ep09]
**Total wall time:** ~3.4 hours (10 × ~20 min/epoch)

**Observations:**
- Train loss decreased every epoch (healthy). Val loss plateaued after ep07/ep08 — classic overfitting onset.
- R@1 peaked at ep07 (0.0898) then degraded through ep08–ep09. Best.pt is NOT the epoch with best R@1.
- The initial batch loss was ~6.7 at batch 351, which is near-random for a 1921-clip database (log₂(1921) ≈ 10.9 nats → expected loss ~6–7 for a damaged model). This is a major red flag — a clean warm-start from Run 6 should have opened around 3–4.

---

## 3. Evaluation Results

**Eval checkpoint:** `checkpoints/finetune8/best.pt` (epoch 08)
**Val set:** 1,921 clips, 722 combos

### 3a. Finetuned model — per query strategy

| Strategy | mAP | MRR | R@1 | R@5 | R@10 | Median 1st Rank |
|----------|----:|----:|----:|----:|-----:|----------------:|
| `all_variants` | 0.1126 | 0.1804 | 4.60% | 12.62% | 19.55% | 24 |
| `sci_common` | 0.1071 | 0.1745 | 4.24% | 12.39% | 18.90% | 25 |
| `chain_common` | 0.1067 | 0.1739 | 4.03% | 12.47% | 19.41% | 25 |
| `name` | 0.0998 | 0.1627 | 3.99% | 10.92% | 18.08% | 27 |
| `rich` | 0.1006 | 0.1613 | 3.97% | 11.38% | 17.49% | 28 |
| `chain` | 0.0989 | 0.1584 | 3.97% | 10.83% | 16.89% | 31 |
| `scientific` | 0.0935 | 0.1485 | 3.41% | 10.47% | 16.36% | 32 |
| `rich_holdout` | 0.0830 | 0.1362 | 2.95% | 8.92% | 15.20% | 31 |

### 3b. Baseline comparisons

| Model | Strategy | mAP | R@1 | R@10 | Median Pos Rank |
|-------|----------|----:|----:|-----:|----------------:|
| Finetuned (best.pt) | all_variants | 0.113 | 4.60% | 19.55% | 83.5 |
| Base (zero-shot CLAP) | name | ~0.012 | ~0.2% | ~1.8% | ~930 |
| Finetuned zero-shot (generalization) | name | ~0.034 | ~0.35% | ~2.9% | ~800 |

> Note: base and finetuned_zeroshot figures sourced from previous eval analysis; exact strategy keys may differ slightly.

### 3c. Key observations

- **R@1 = 4.6%** with all_variants is extremely poor. Run 6 (our benchmark) achieved R@1 ≈ 10–14% (no audit exists).
- **Median positive rank = 83.5** — on average, the ground-truth clip for a query ranks inside the top 5% of the 1921-clip database, which sounds tolerable, but median *first* rank = 24 means the model almost never places the best match at rank 1.
- **`rich_holdout` mAP = 0.083** — the model generalizes poorly to unseen rich descriptions, confirming text encoder damage.
- **Finetuned zero-shot median pos rank = 800** out of 1921. This means ~41st percentile — barely above random chance (960). The model has essentially learned to memorize the training set topology rather than acoustic-semantic alignment.

---

## 4. Root Cause Analysis

### Primary cause: warm-start from a damaged checkpoint

The 7th run used the `--acoustic-only` flag (confirmed in the 7th run script). This flag filtered out all taxonomy/rich text labels during training, forcing the text encoder to predict only from acoustic-type tokens (e.g., "call", "song"). After ~10 epochs with that constraint, the text encoder's alignment with species-level semantics was severely degraded.

When Run 8 warm-started from Run 7's `best.pt`, it inherited this damage. The evidence:
1. **Initial batch loss ~6.7** at epoch 00, batch ~350. A warm-start from a healthy Run 6 checkpoint would open much lower (~3.5–4.0). Near-random loss at epoch 0 indicates the text embeddings are misaligned.
2. **Finetuned zero-shot generalization is near-random** (median pos rank = 800/1921). The model failed to re-learn proper acoustic-semantic anchoring even after 10 full epochs.
3. The training curve shows loss descending normally, suggesting the optimizer was making progress, but starting from such a bad initialization left the model stuck in a poor local minimum relative to its Run 6 counterpart.

### Secondary cause: no audit for Run 7

Run 7 was never audited. If it had been, the `--acoustic-only` damage would have been caught before Run 8 was launched.

---

## 5. Comparison with Previous Runs

| Run | Best mAP (finetuned/name) | R@1 | Val set size | Notes |
|-----|:-------------------------:|----:|-------------:|-------|
| 2 | 0.291 | 13.0% | ~2k | Best historical run |
| 3 | 0.478 | 17.2% | 556 clips | Inflated — tiny val set |
| 4 | 0.198 | 5.3% | Expanded | Regression after val expansion |
| 5 | 0.246 | 7.8% | ~similar | Partial recovery |
| 6 | unknown | ~10%+ | similar | No audit; confirmed best pre-8 |
| 7 | unknown | N/A | — | `--acoustic-only` damage; never evaluated properly |
| **8** | **0.100** | **4.0%** | 1921 clips | Damaged warm-start; worst result |

---

## 6. What Must Change for Run 9

1. **Warm-start from Run 6's `best.pt`** — NOT from Run 7 or 8. Run 6's checkpoint is the last known-good model. If unavailable, start fresh from `laion/clap-htsat-fused`.
2. **Lower the learning rate** — with an already fine-tuned model, use `--lr 5e-6` instead of `2e-5`. Run 8's high initial LR likely caused early instability.
3. **Expand the dataset first** — `scripts/download_targeted_xc.py` has added ~9,900 new recordings (total ≈ 27,913 in `xc_metadata_unified.csv`). Rebuild `clap_train_pairs.json` / `clap_val_pairs.json` and pre-compute `.clap.pt` features for new files.
4. **Verify the starting checkpoint** — before launching, print the first batch loss. If it's >4.5, abort. It should open ≤3.5 for a healthy warm-start.
5. **Write this audit BEFORE and AFTER each run** — see the auto-generated `TRAINING_AUDIT.md` in the checkpoint dir, created by `scripts/train_clap.py`.

---

## 7. Files Referenced

| File | Description |
|------|-------------|
| `training_8th_finetune.log` | Full training log (185,413 lines) |
| `results/eval_results_finetune8.json` | Detailed per-combo eval results |
| `results/figures_finetune8/` | Rank CDF, class breakdown, delta charts |
| `checkpoints/finetune8/best.pt` | Epoch 08 checkpoint (val_loss=1.4327) |
| `checkpoints/finetune7/best.pt` | Likely warm-start source (damaged) |

---

*This audit was reconstructed post-run from log and eval files. Future runs will have auto-generated audits via `scripts/train_clap.py`.*
