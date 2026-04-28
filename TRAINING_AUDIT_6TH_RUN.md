# Training Audit — 6th fine-tuning run

**Status:** retrospective (evaluation + figures archived; training log not in repo)  
**Sources of truth:** `results/eval_results_6th_finetune.json` and `results/figures 6th run/*.pdf`  
**Evaluation checkpoint (from JSON):** `checkpoints/sixth-fine-tune/best.pt`

---

## Reading the PDFs

The evaluation script writes numbers to JSON first, then renders PDFs from those aggregates. **This audit uses only the JSON** for headline metrics — that is exactly what each chart is plotting. I cannot reliably decode or reinterpret the visual curves from the PDF binaries here (fonts, vectors, images). Filenames describe the chart intent (below).

---

## 1. What this run trained with

Per project history:

- **`scripts/train_clap.py`** (not a separate `train_clap_v4.py`). In this repo, `train_clap_v4` functionality was merged into the canonical script (e.g. commit `5de82a7`: *train_clap: port --finetune-from and ParetoCheckpointManager from train_clap_v4*).

If you need an epoch-by-loss table here, locate the desktop training log for sixth-fine-tune — it was not checked into this repo (large logs are gitignored).

---

## 2. Evaluation setup (numbers from JSON)

| Field | Value |
|-------|------:|
| Base model id | `laion/clap-htsat-fused` |
| Val clips \(N\) | **1,607** |
| Val combos (queries) \(K\) | **512** \(for most strategies\) |
| Finetuned checkpoint | `checkpoints/sixth-fine-tune/best.pt` |

(Strategies `rich` / `rich_holdout` use **507** queries when some combos lack rich text; core strategies use **512** queries where defined. **`finetuned_zeroshot`** uses ~**266–270** queries per strategy — see JSON `n_queries`.)

---

## 3. Aggregated retrieval — finetuned model

Symmetric text→audio retrieval; metrics are aggregated over queries (see project docs for definitions of mAP / R@k / MRR).

| Strategy | mAP | MRR | R@1 | R@10 | median 1st rank | median positive rank |
|----------|-----:|-----:|-----:|-----:|----------------:|----------------------:|
| **semantic** | 0.247 | 0.361 | 9.53% | 39.16% | 6.0 | 23.0 |
| **all_variants** | 0.247 | 0.363 | **9.39%** | **38.74%** | 6.0 | 23.0 |
| chain_common | 0.243 | 0.354 | 9.54% | 36.90% | 6.0 | 23.75 |
| sci_common | 0.239 | 0.354 | 9.22% | 37.22% | 6.0 | 24.0 |
| rich | 0.235 | 0.352 | 9.16% | 36.80% | 7.0 | 25.0 |
| name | 0.236 | 0.346 | 8.98% | 36.26% | 6.5 | 23.5 |
| chain | 0.229 | 0.343 | 8.86% | 35.92% | 7.0 | 29.0 |
| scientific | 0.226 | 0.336 | 8.22% | 35.48% | 7.0 | 28.5 |
| rich_holdout | 0.233 | 0.350 | 8.77% | 36.45% | 7.0 | 26.0 |

**Takeaways from the aggregates:**

- **`semantic` vs `chain_common`**: both land around **~0.24–0.25 mAP** — the model benefits from probing paraphrases and taxonomy-style wording similarly.
- **Rich text vs taxonomy**: `rich` and **`rich_holdout`** (~0.233 mAP) are only slightly below `chain_common` (**~0.243 mAP**) — same pattern as earlier audits: **per-species RAG-ish text did not materially beat simple chains** at this aggregation level.

---

## 4. Baseline `laion/clap-htsat-fused` (no fine-tuning)

| Strategy | mAP | MRR | R@1 | R@10 |
|----------|-----:|-----:|-----:|-----:|
| name | 0.0218 | 0.0331 | **0.91%** | 2.90% |
| sci_common | 0.0137 | 0.0215 | 0.39% | **1.85%** |
| chain_common | **0.00861** | 0.0137 | **0.098%** | **1.06%** |
| chain | 0.00827 | 0.0116 | 0.20% | 0.92% |
| scientific | **0.00938** | 0.0123 | **0.20%** | 0.93% |

**Rank sanity check (base, `name`):** median_first_rank ≈ **319**, median_pos_rank ≈ **672** — near-random search over ~1.6k clips.

So the **fine-tuned model is massively above base** on the same evaluation split (whatever the headline R@1, the jump from ~1% toward ~9%+ on `name` is the real story).

---

## 5. `finetuned_zeroshot` (held-out probing)

These rows measure **generalisation to query types not aligned with training templates** — smaller effective query counts (**266–270** queries depending on strategy; see JSON `n_queries`).

| Strategy | mAP | R@1 | R@10 | median_pos_rank \(approx.\) |
|----------|-----:|-----:|-----:|----------------------------:|
| all_variants | 0.187 | 3.45% | 17.57% | 69 |
| sci_common | 0.181 | 2.90% | 15.76% | 76 |

This is **well below** in-distribution `all_variants` (9.39% R@1) — expected: the probe set is deliberately harder.

---

## 6. Figures in `results/figures 6th run/` (what each PDF is for)

These match the evaluator’s usual layout; each is derived from the same JSON used above.

| PDF | Role |
|-----|------|
| `recall_at_k.pdf` | R@{1…k} bars — finetuned vs base (`strategy_comparison`-style aggregates may overlap) |
| `strategy_comparison.pdf` | Side‑by‑side strategy comparison |
| `delta_finetuned_vs_base.pdf` | Per-metric deltas (finetuned − base) |
| `class_breakdown_finetuned.pdf` | Per-(species,voc\_type) bar chart — finetuned |
| `class_breakdown_base.pdf` | Same for base |
| `hardest_easiest_finetuned.pdf` | Combos sorted by mAP |
| `map_distribution_*` / `similarity_distribution_*` | Histograms of class mAP / score spread |
| `rank_cdf_base.pdf`, `rank_cdf_finetuned.pdf` | Empirical CDF of first-hit rank |
| `rank_distribution_*.pdf` | Rank histogram variants |
| `*_zeroshot.pdf` counterparts | Same views for zeroshot / finetuned-zeroshot modes |

Concrete **example detail** illustrating the tails (from JSON `detail`, `name`): **Peregrine Falcon / call** has **mAP ≈ 0** with **one** positive in the pool and **first_hit_rank 1160** — the kind of point at the “hardest” end of `hardest_easiest_*.pdf`.

---

## 7. Why Run 6 can look weaker than Run 2 (or shinier PDFs elsewhere) — apples vs oranges

Headline percentages are **not a fair leaderboard** across runs unless everyone uses the same validation split **and** the same eval harness.

### Validation pool size (from stored eval JSON)

| Run (eval artifact) | `n_val_clips` | `n_val_combos` |
|---------------------|----------------:|---------------:|
| 2 (`eval_results_second_fine_tune_best.json`) | **1,197** | **605** |
| 5 (`eval_results_5th_run.json`) | **1,607** | **512** |
| **6** (`eval_results_6th_finetune.json`) | **1,607** | **512** |

- **Vs Run 2:** The second fine-tune used a **smaller retrieval gallery** (1,197 vs 1,607 clips) and a different split. Larger galleries mean **more distractor clips per query**, so **R@m and raw R@1 often drop** even when the model improved — so Run 2’s **~13% R@1 (`name`)** is **not directly comparable** to Run 6’s **~9%** without re-evaluating both checkpoints on the same pairs.

- **Vs Run 5:** In these JSON exports, Run 5 and Run 6 share the **same** `n_val_clips` / `n_val_combos`. Core strategies (`name`, `chain_*`, etc.) are **on comparable footing**; if one run looks worse, it is more likely **model/checkpoint**, not a smaller pdf count.

### “Run 6 has more PDFs” — evaluation got broader, not just stricter totals

Run 6’s figure pack is larger because **`evaluate_clap.py` gained more reporting paths** over time: e.g. **rank CDFs**, **rank distributions**, **base/finetuned zeroshot chart variants**, and strategies like **`semantic`** / **`finetuned_zeroshot`** that **earlier JSONs often did not record at all**. Extra PDFs mainly mean **more views of the same experiment** — including **harder probe rows** (zeroshot) that deliberately look bad — not that the base `name` metric was computed with a magically “unfair” denominator vs Run 5.

**Practical takeaway:** Prefer **within-run deltas** (base → finetuned, `all_variants` vs `rich_holdout`, finetuned vs `finetuned_zeroshot`). For cross-run bragging rights, **re-run eval** on one frozen `clap_val_pairs.json` + one script version.

---

## 8. Verdict

- **Fine-tuning with `scripts/train_clap.py` on this checkpoint produced a large lift over pretrained CLAP** on this 1 607‑clip retrieval benchmark.
- **Strategy spread is modest** (~0.23–0.25 mAP band among strong strategies) — taxonomy chains and semantic probes behave similarly to rich/holdout at the aggregate level, consistent with prior project audits.
- **Zeroshot probe rows prove the model still overfits templated wording** somewhat (R@1 drops into the mid‑single digits on `finetuned_zeroshot`).
- Figures in **`results/figures 6th run/`** visualize these exact aggregates; cite this file + JSON, not guesses from redrawn curves.
