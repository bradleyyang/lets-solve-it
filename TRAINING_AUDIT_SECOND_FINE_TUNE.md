# CLAP Fine-Tuning Audit — second fine-tune run

**Date completed:** 2026-04-14  
**Run type:** warm-start into `checkpoints/second-fine-tune/`  
**Model:** `laion/clap-htsat-fused`

---

## 1) Run configuration

- Training command family:
  - `scripts/train_clap.py --checkpoint-dir checkpoints/second-fine-tune --resume checkpoints/second-fine-tune/latest.pt`
- Main settings:
  - epochs: 10 total (00-09)
  - batch: 8, accum: 8 (effective batch 64)
  - LR: `2e-5`
  - warmup: `500`
  - freeze text encoder: first 2 epochs
- Dataset sizes logged:
  - train pairs: 96,627
  - val pairs: 10,728

---

## 2) Training results

| Epoch | Train Loss | Val Loss | R@1 | Duration |
|------:|-----------:|---------:|----:|---------:|
| 00 | 1.8025 | 1.3542 | 0.0879 | 7,910s |
| 01 | 1.1299 | 1.1420 | 0.1843 | 7,718s |
| 02 | 0.3773 | 0.6744 | 0.3320 | 7,804s |
| 03 | 0.1378 | 0.6882 | 0.3836 | 7,793s |
| 04 | 0.0813 | 0.6718 | 0.3867 | 13,841s |
| 05 | 0.0537 | 0.6801 | 0.4235 | 13,834s |
| 06 | 0.0311 | 0.6927 | 0.4297 | 6,551s |
| 07 | 0.0200 | **0.6520** | 0.4501 | 2,603s |
| 08 | 0.0318 | 0.6681 | **0.4814** | 2,632s |
| 09 | 0.0321 | 0.6736 | 0.4471 | 2,623s |

**Best checkpoint:** `checkpoints/second-fine-tune/best.pt` (epoch 07, `val_loss=0.6520`)  
**Latest checkpoint:** `checkpoints/second-fine-tune/latest.pt` (epoch 09)

---

## 3) Retrieval evaluation (fresh run)

Evaluation command:

```bash
python scripts/evaluate_clap.py \
  --checkpoint checkpoints/second-fine-tune/best.pt \
  --also-base \
  --metadata scripts/xc_metadata_unified.csv \
  --audio-root scripts/data/xc_audio \
  --output results/eval_results_second_fine_tune_best.json \
  --figures-dir results/figures_second_fine_tune_best
```

Notes:
- 2 clips failed to load in eval and were excluded.
- 603 queries per strategy were evaluated.

### Fine-tuned (`checkpoints/second-fine-tune/best.pt`)

| Strategy | mAP | MRR | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|
| name | 0.291 | 0.385 | 0.130 | 0.369 | 0.489 |
| scientific | 0.280 | 0.373 | 0.124 | 0.350 | 0.468 |
| chain | 0.281 | 0.370 | 0.123 | 0.348 | 0.466 |
| sci_common | **0.296** | **0.391** | **0.136** | 0.366 | **0.488** |
| chain_common | 0.292 | 0.390 | 0.135 | 0.357 | 0.485 |
| rich | 0.281 | 0.370 | 0.123 | 0.348 | 0.466 |

### Base (`laion/clap-htsat-fused`)

| Strategy | mAP | MRR | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|
| name | 0.017 | 0.024 | 0.004 | 0.015 | 0.030 |
| scientific | 0.008 | 0.011 | 0.002 | 0.004 | 0.005 |
| chain | 0.010 | 0.013 | 0.004 | 0.007 | 0.008 |
| sci_common | 0.012 | 0.018 | 0.000 | 0.011 | 0.019 |
| chain_common | 0.011 | 0.014 | 0.003 | 0.005 | 0.016 |
| rich | 0.010 | 0.013 | 0.004 | 0.007 | 0.008 |

### Delta (fine-tuned - base)

- d mAP: +0.271 to +0.284
- d MRR: +0.357 to +0.376
- d R@1: +0.118 to +0.136
- d R@10: +0.458 to +0.469

---

## 4) Artifacts

- JSON: `results/eval_results_second_fine_tune_best.json`
- Figures: `results/figures_second_fine_tune_best/`
  - `strategy_comparison.pdf`
  - `recall_at_k.pdf`
  - `map_distribution_finetuned.pdf`
  - `map_distribution_base.pdf`
  - `class_breakdown_finetuned.pdf`
  - `class_breakdown_base.pdf`
  - `similarity_distribution_finetuned.pdf`
  - `similarity_distribution_base.pdf`
  - `hardest_easiest_finetuned.pdf`
  - `hardest_easiest_base.pdf`
  - `delta_finetuned_vs_base.pdf`

---

## 5) Summary

- The second fine-tune converged cleanly and substantially improved retrieval quality over base CLAP.
- Best validation checkpoint occurred at epoch 07, while best R@1 occurred at epoch 08.
- `sci_common` is the strongest overall strategy by mAP/MRR/R@1 for this run.
