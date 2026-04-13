"""
BirdCLAP Evaluation Script
===========================
Evaluates a CLAP model (fine-tuned or base) on the held-out val set using
text-to-audio retrieval. Tests six query strategies so you can see exactly
which label types drive performance.

Evaluation protocol:
  - Database   : all unique val audio clips (from clap_val_pairs.json)
  - Queries    : one text query per (species, type) combo, per strategy
  - Positives  : all val clips that belong to the queried (species, type)
  - Metrics    : mAP, MRR, R@1, R@5, R@10  (multi-label retrieval)

Six query strategies:
  1. name          "Northern Cardinal call"
  2. scientific    "Cardinalis cardinalis call"
  3. chain         "Animalia > ... > Cardinalis cardinalis call"
  4. sci_common    "Cardinalis cardinalis, Northern Cardinal call"
  5. chain_common  "Animalia > ... > Cardinalis cardinalis, Northern Cardinal call"
  6. rich          LLM-generated acoustic description (first variant)

Plots saved to results/figures/:
  1. strategy_comparison.pdf   — mAP / MRR / R@K bar chart per strategy
  2. recall_at_k.pdf           — R@K curve (K=1..20) per strategy
  3. map_distribution.pdf      — violin plot of per-combo mAP per strategy
  4. class_breakdown.pdf       — mAP by taxonomic class × strategy
  5. similarity_distribution.pdf — KDE of positive vs negative cosine sim
  6. hardest_easiest.pdf       — top-20 hardest / easiest species by mAP
  7. delta_finetuned_vs_base.pdf — Δmetric bar chart  (only with --also-base)

Usage:
    conda activate birdclap

    python scripts/evaluate_clap.py --checkpoint checkpoints/best.pt
    python scripts/evaluate_clap.py --checkpoint checkpoints/best.pt --also-base
    python scripts/evaluate_clap.py          # base model only
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "laion/clap-htsat-fused"
DEFAULT_VAL_PAIRS  = Path("data/clap_val_pairs.json")
DEFAULT_METADATA   = Path("data/xc_metadata_unified.csv")
DEFAULT_LABELS     = Path("data/clap_all_labels.json")
DEFAULT_TAXONOMY   = Path("data/species_taxonomy.json")
DEFAULT_AUDIO_ROOT = Path(".")
DEFAULT_OUTPUT     = Path("results/eval_results.json")
DEFAULT_FIGURES    = Path("results/figures")
AUDIO_SR           = 48_000
CLIP_SECONDS       = 10
MIN_DURATION_S     = 0.5  # match scripts/train_clap.py
STRATEGY_ORDER     = ["name", "scientific", "chain", "sci_common", "chain_common", "rich"]
STRATEGY_LABELS    = {
    "name":         "Common name",
    "scientific":   "Scientific name",
    "chain":        "Taxonomy chain",
    "sci_common":   "Sci + common",
    "chain_common": "Chain + common",
    "rich":         "Rich description",
}


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: Path, sr: int = AUDIO_SR, clip_s: float = CLIP_SECONDS):
    """Load audio like train_clap.py: prefer pre-clipped .wav sibling, else librosa."""
    import librosa
    import soundfile as sf

    target_len = int(clip_s * sr)
    min_len = int(MIN_DURATION_S * sr)
    wav_path = path.with_suffix(".wav")
    if wav_path.is_file():
        try:
            y, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if file_sr == sr and len(y) == target_len:
                return y
            if len(y) < min_len:
                return None
            if len(y) >= target_len:
                start = (len(y) - target_len) // 2
                y = y[start : start + target_len]
            else:
                y = np.pad(y, (0, target_len - len(y)))
            return y.astype(np.float32)
        except Exception:
            pass
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception as e:
        print(f"  [audio error] {path}: {e}")
        return None
    if len(y) < min_len:
        return None
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    return y.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_audio_batch(wavs, processor, model, device):
    inputs = processor(
        audio=wavs,
        return_tensors="pt",
        sampling_rate=AUDIO_SR,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feat = model.get_audio_features(
        input_features=inputs.get("input_features"),
        is_longer=inputs.get("is_longer"),
    )
    return F.normalize(feat.pooler_output, dim=-1).cpu()


@torch.no_grad()
def encode_text_batch(texts, processor, model, device):
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feat = model.get_text_features(
        input_ids=inputs.get("input_ids"),
        attention_mask=inputs.get("attention_mask"),
    )
    return F.normalize(feat.pooler_output, dim=-1).cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_metrics(sim_row: np.ndarray,
                      positive_indices: list[int],
                      max_k: int = 20) -> dict:
    """
    Compute mAP, MRR, R@1..max_k for a single query.
    Also returns raw positive and negative similarity scores for
    distribution analysis.
    """
    ranked    = np.argsort(sim_row)[::-1]
    pos_set   = set(positive_indices)
    n_pos     = len(pos_set)
    n_total   = len(sim_row)

    # R@k for k = 1..max_k
    recall_at_k = {}
    for k in range(1, max_k + 1):
        hits = sum(1 for idx in ranked[:k] if idx in pos_set)
        recall_at_k[k] = hits / n_pos

    # MRR
    mrr = 0.0
    for rank, idx in enumerate(ranked, 1):
        if idx in pos_set:
            mrr = 1.0 / rank
            break

    # mAP
    hits, ap_sum = 0, 0.0
    for rank, idx in enumerate(ranked, 1):
        if idx in pos_set:
            hits += 1
            ap_sum += hits / rank
    ap = ap_sum / n_pos if n_pos else 0.0

    # Positive / negative similarity distributions
    pos_sims = sim_row[positive_indices].tolist()
    neg_mask = np.ones(n_total, dtype=bool)
    neg_mask[positive_indices] = False
    neg_sims = sim_row[neg_mask].tolist()

    return {
        "mAP": ap,
        "MRR": mrr,
        "recall_at_k": recall_at_k,
        "pos_sims": pos_sims,
        "neg_sims": neg_sims,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Query builders
# ─────────────────────────────────────────────────────────────────────────────

def build_queries(combos, tax_db, all_labels):
    strategies = {s: {} for s in STRATEGY_ORDER}
    for name, vtype in combos:
        key = f"{name}||{vtype}"
        tax = tax_db.get(name, {})
        sci     = tax.get("scientific", "")
        kingdom = tax.get("kingdom", "Animalia")
        phylum  = tax.get("phylum",  "Chordata")
        cls     = tax.get("class",   "")
        order   = tax.get("order",   "")
        family  = tax.get("family",  "")
        genus   = tax.get("genus",   "")
        chain   = " > ".join(p for p in [kingdom, phylum, cls, order, family, genus, sci] if p)
        vt = vtype.strip()

        strategies["name"][(name, vtype)] = f"{name} {vt}"
        if sci:
            strategies["scientific"][(name, vtype)]   = f"{sci} {vt}"
            strategies["chain"][(name, vtype)]         = f"{chain} {vt}"
            strategies["sci_common"][(name, vtype)]    = f"{sci}, {name} {vt}"
            strategies["chain_common"][(name, vtype)]  = f"{chain}, {name} {vt}"

        rich_variants = [v for v in all_labels.get(key, []) if len(v.split()) > 8]
        if rich_variants:
            strategies["rich"][(name, vtype)] = rich_variants[0]

    return strategies


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint, base_model, device):
    print(f"  Loading base model: {base_model}")
    model     = ClapModel.from_pretrained(base_model)
    processor = ClapProcessor.from_pretrained(base_model)
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = state.get(
            "model_state",
            state.get("model_state_dict", state.get("state_dict", state)),
        )
        inc = model.load_state_dict(sd, strict=False)
        missing = list(getattr(inc, "missing_keys", ()))
        unexpected = list(getattr(inc, "unexpected_keys", ()))
        if missing:
            print(f"  [warn] {len(missing)} missing keys in checkpoint")
        if unexpected:
            print(f"  [warn] {len(unexpected)} unexpected keys in checkpoint")
        print(f"  Loaded fine-tuned weights from {ckpt_path}")
    else:
        print(f"  No checkpoint - evaluating base model (zero-shot)")
    model.eval().to(device)
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(model, processor, val_clips, clip_to_combo,
             strategies, audio_root, device, batch_size=16):
    """
    Returns:
        agg     : {strategy: {metric: float}}   — macro-averaged metrics
        detail  : {strategy: [{combo, mAP, MRR, R@k, tax_class, ...}]}
        sim_data: {strategy: {"pos": [...], "neg": [...]}}
    """
    # Encode audio
    print(f"\n  Encoding {len(val_clips)} val audio clips ...")
    audio_embs, failed = [], []
    for i in range(0, len(val_clips), batch_size):
        batch_paths = val_clips[i: i + batch_size]
        wavs, valid = [], []
        for p in batch_paths:
            wav = load_audio(audio_root / p)
            if wav is not None:
                wavs.append(wav); valid.append(p)
            else:
                failed.append(p)
        if wavs:
            audio_embs.append(encode_audio_batch(wavs, processor, model, device))
        if (i // batch_size) % 10 == 0:
            print(f"    {min(i + batch_size, len(val_clips))}/{len(val_clips)}", end="\r")

    if failed:
        print(f"\n  [warn] {len(failed)} clips failed to load - excluded")
    valid_clips = [c for c in val_clips if c not in set(failed)]
    print(f"\n  Audio encoding done.")
    audio_matrix = torch.cat(audio_embs, dim=0).numpy()

    combo_to_indices = defaultdict(list)
    for idx, clip in enumerate(valid_clips):
        combo = clip_to_combo.get(clip)
        if combo:
            combo_to_indices[combo].append(idx)

    agg, detail, sim_data = {}, {}, {}

    for strategy, combo_queries in strategies.items():
        eval_combos = [(c, q) for c, q in combo_queries.items()
                       if c in combo_to_indices and combo_to_indices[c]]
        if not eval_combos:
            print(f"  [{strategy}] no valid combos - skipping")
            continue

        print(f"  [{strategy}] encoding {len(eval_combos)} queries ...")
        query_texts = [q for _, q in eval_combos]
        text_embs = []
        for i in range(0, len(query_texts), 64):
            text_embs.append(encode_text_batch(query_texts[i:i+64], processor, model, device))
        text_matrix = torch.cat(text_embs, dim=0).numpy()
        sim_matrix  = text_matrix @ audio_matrix.T

        per_combo, all_pos_sims, all_neg_sims = [], [], []
        for q_idx, (combo, _) in enumerate(eval_combos):
            m = retrieval_metrics(sim_matrix[q_idx], combo_to_indices[combo])
            per_combo.append({
                "species":      combo[0],
                "voc_type":     combo[1],
                "mAP":          m["mAP"],
                "MRR":          m["MRR"],
                "recall_at_k":  m["recall_at_k"],
                "n_positives":  len(combo_to_indices[combo]),
            })
            all_pos_sims.extend(m["pos_sims"])
            all_neg_sims.extend(m["neg_sims"])

        # Macro-average
        a = {}
        for metric in ["mAP", "MRR"]:
            a[metric] = float(np.mean([c[metric] for c in per_combo]))
        for k in range(1, 21):
            a[f"R@{k}"] = float(np.mean([c["recall_at_k"][k] for c in per_combo]))
        a["n_queries"] = len(eval_combos)
        agg[strategy]      = a
        detail[strategy]   = per_combo
        sim_data[strategy] = {"pos": all_pos_sims, "neg": all_neg_sims}

        print(f"    mAP={a['mAP']:.3f}  MRR={a['MRR']:.3f}  "
              f"R@1={a['R@1']:.3f}  R@5={a['R@5']:.3f}  R@10={a['R@10']:.3f}  "
              f"({a['n_queries']} queries)")

    return agg, detail, sim_data


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(all_results: dict, tax_db: dict, figures_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Style
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        10,
        "axes.titlesize":   11,
        "axes.labelsize":   10,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "axes.grid.axis":   "y",
        "grid.alpha":       0.3,
        "figure.dpi":       150,
        "savefig.bbox":     "tight",
        "savefig.dpi":      200,
    })

    MODEL_COLORS = {"finetuned": "#2166ac", "base": "#d6604d"}
    STRATEGY_COLORS = plt.cm.tab10(np.linspace(0, 0.6, len(STRATEGY_ORDER)))
    strategies_present = [s for s in STRATEGY_ORDER
                          if any(s in r["agg"] for r in all_results.values())]
    s_labels = [STRATEGY_LABELS.get(s, s) for s in strategies_present]

    # ── 1. Strategy comparison bar chart ──────────────────────────────────────
    metrics_to_show = ["mAP", "MRR", "R@1", "R@5", "R@10"]
    n_metrics = len(metrics_to_show)
    n_strat   = len(strategies_present)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4.5), sharey=False)
    fig.suptitle("Retrieval Performance by Query Strategy", fontweight="bold", y=1.02)

    for ax, metric in zip(axes, metrics_to_show):
        for model_key, res in all_results.items():
            vals = [res["agg"].get(s, {}).get(metric, np.nan) for s in strategies_present]
            x    = np.arange(n_strat)
            offset = 0.2 * (list(all_results.keys()).index(model_key) - (len(all_results) - 1) / 2)
            bars = ax.bar(x + offset, vals, width=0.35 / len(all_results),
                          color=MODEL_COLORS.get(model_key, "#555"),
                          label=model_key, alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(metric)
        ax.set_xticks(np.arange(n_strat))
        ax.set_xticklabels(s_labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    handles = [Patch(color=MODEL_COLORS.get(k, "#555"), label=k) for k in all_results]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "strategy_comparison.pdf")
    plt.close(fig)
    print(f"  Saved: strategy_comparison.pdf")

    # ── 2. Recall@K curve ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(all_results),
                             figsize=(5.5 * len(all_results), 4), sharey=True, squeeze=False)
    fig.suptitle("Recall@K Curve (K = 1 … 20)", fontweight="bold")
    ks = list(range(1, 21))

    for ax, (model_key, res) in zip(axes[0], all_results.items()):
        for i, s in enumerate(strategies_present):
            if s not in res["agg"]:
                continue
            vals = [res["agg"][s].get(f"R@{k}", np.nan) for k in ks]
            ax.plot(ks, vals, marker="o", markersize=3, linewidth=1.5,
                    color=STRATEGY_COLORS[i], label=STRATEGY_LABELS.get(s, s))
        ax.set_title(model_key)
        ax.set_xlabel("K")
        ax.set_ylabel("Recall@K")
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(figures_dir / "recall_at_k.pdf")
    plt.close(fig)
    print(f"  Saved: recall_at_k.pdf")

    # ── 3. mAP distribution (violin) ─────────────────────────────────────────
    for model_key, res in all_results.items():
        fig, ax = plt.subplots(figsize=(max(6, n_strat * 1.4), 4.5))
        data  = [np.array([c["mAP"] for c in res["detail"].get(s, [])]) for s in strategies_present]
        parts = ax.violinplot(data, positions=np.arange(n_strat), showmedians=True,
                              showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(STRATEGY_COLORS[i])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        # Overlay median value text
        for i, d in enumerate(data):
            if len(d):
                ax.text(i, np.median(d) + 0.02, f"{np.median(d):.2f}",
                        ha="center", va="bottom", fontsize=8)

        ax.set_xticks(np.arange(n_strat))
        ax.set_xticklabels(s_labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Average Precision (per query)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-query mAP Distribution — {model_key}")
        fig.tight_layout()
        fig.savefig(figures_dir / f"map_distribution_{model_key}.pdf")
        plt.close(fig)
        print(f"  Saved: map_distribution_{model_key}.pdf")

    # ── 4. Taxonomic class breakdown ──────────────────────────────────────────
    for model_key, res in all_results.items():
        # Collect mAP per (class, strategy)
        class_map: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for s in strategies_present:
            for c in res["detail"].get(s, []):
                tax_class = tax_db.get(c["species"], {}).get("class", "Unknown")
                class_map[tax_class][s].append(c["mAP"])

        classes = sorted(class_map.keys())
        n_cls   = len(classes)
        fig, ax = plt.subplots(figsize=(max(6, n_cls * n_strat * 0.4 + 2), 4.5))
        group_w = 0.8
        bar_w   = group_w / n_strat
        x       = np.arange(n_cls)

        for i, s in enumerate(strategies_present):
            means = [np.mean(class_map[cls][s]) if class_map[cls][s] else np.nan
                     for cls in classes]
            sems  = [np.std(class_map[cls][s]) / np.sqrt(len(class_map[cls][s]))
                     if len(class_map[cls][s]) > 1 else 0 for cls in classes]
            ax.bar(x + i * bar_w - group_w / 2 + bar_w / 2, means,
                   bar_w * 0.9, yerr=sems, color=STRATEGY_COLORS[i], alpha=0.85,
                   label=STRATEGY_LABELS.get(s, s), capsize=3, error_kw={"linewidth": 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=20, ha="right")
        ax.set_ylabel("Mean AP")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"mAP by Taxonomic Class — {model_key}")
        ax.legend(fontsize=8, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(figures_dir / f"class_breakdown_{model_key}.pdf")
        plt.close(fig)
        print(f"  Saved: class_breakdown_{model_key}.pdf")

    # ── 5. Similarity score distribution (positive vs negative) ──────────────
    for model_key, res in all_results.items():
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
        fig.suptitle(f"Positive vs Negative Similarity Distributions — {model_key}",
                     fontweight="bold")
        axes_flat = axes.flatten()

        for i, s in enumerate(strategies_present[:6]):
            ax = axes_flat[i]
            sims = res["sim_data"].get(s)
            if not sims:
                ax.set_visible(False)
                continue
            pos = np.array(sims["pos"])
            neg = np.array(sims["neg"])
            # KDE via histogram approximation
            bins = np.linspace(-0.1, 1.0, 60)
            ax.hist(neg, bins=bins, density=True, alpha=0.55, color="#d6604d", label="Negative")
            ax.hist(pos, bins=bins, density=True, alpha=0.75, color="#2166ac", label="Positive")
            ax.axvline(np.mean(pos), color="#2166ac", linestyle="--", linewidth=1)
            ax.axvline(np.mean(neg), color="#d6604d", linestyle="--", linewidth=1)
            ax.set_title(STRATEGY_LABELS.get(s, s), fontsize=9)
            ax.set_xlabel("Cosine similarity", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=7, frameon=False)
            # Annotate separation
            sep = np.mean(pos) - np.mean(neg)
            ax.text(0.97, 0.95, f"Δμ = {sep:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        fig.savefig(figures_dir / f"similarity_distribution_{model_key}.pdf")
        plt.close(fig)
        print(f"  Saved: similarity_distribution_{model_key}.pdf")

    # ── 6. Hardest / easiest species (using 'name' strategy) ─────────────────
    for model_key, res in all_results.items():
        detail = res["detail"].get("name") or res["detail"].get(strategies_present[0], [])
        if not detail:
            continue
        sorted_by_map = sorted(detail, key=lambda x: x["mAP"])
        hardest = sorted_by_map[:20]
        easiest = sorted_by_map[-20:][::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"20 Hardest / Easiest Species by mAP — {model_key}", fontweight="bold")

        def bar_species(ax, items, color, title):
            labels = [f"{c['species']} ({c['voc_type']})" for c in items]
            vals   = [c["mAP"] for c in items]
            y = np.arange(len(labels))
            bars = ax.barh(y, vals, color=color, alpha=0.8, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{v:.2f}", va="center", fontsize=7.5)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlim(0, 1.1)
            ax.set_xlabel("mAP")
            ax.set_title(title)
            ax.invert_yaxis()

        bar_species(ax1, hardest, "#d6604d", "20 Hardest (lowest mAP)")
        bar_species(ax2, easiest, "#2166ac", "20 Easiest (highest mAP)")
        fig.tight_layout()
        fig.savefig(figures_dir / f"hardest_easiest_{model_key}.pdf")
        plt.close(fig)
        print(f"  Saved: hardest_easiest_{model_key}.pdf")

    # ── 7. Delta chart (fine-tuned − base) ────────────────────────────────────
    if "finetuned" in all_results and "base" in all_results:
        metrics = ["mAP", "MRR", "R@1", "R@5", "R@10"]
        n_m     = len(metrics)
        fig, axes = plt.subplots(1, n_m, figsize=(3.5 * n_m, 4.5), sharey=False)
        fig.suptitle("Δ Metric: Fine-tuned − Base (zero-shot)", fontweight="bold", y=1.02)

        for ax, metric in zip(axes, metrics):
            deltas = []
            for s in strategies_present:
                ft_val   = all_results["finetuned"]["agg"].get(s, {}).get(metric, np.nan)
                base_val = all_results["base"]["agg"].get(s, {}).get(metric, np.nan)
                deltas.append(ft_val - base_val if not (np.isnan(ft_val) or np.isnan(base_val)) else np.nan)

            colors = ["#2166ac" if (not np.isnan(d) and d >= 0) else "#d6604d" for d in deltas]
            bars   = ax.bar(np.arange(n_strat), deltas, color=colors, alpha=0.85, edgecolor="white")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            for bar, v in zip(bars, deltas):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (0.003 if v >= 0 else -0.008),
                            f"{v:+.3f}", ha="center",
                            va="bottom" if v >= 0 else "top", fontsize=7)
            ax.set_title(f"Δ{metric}")
            ax.set_xticks(np.arange(n_strat))
            ax.set_xticklabels(s_labels, rotation=35, ha="right", fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.2f"))

        fig.tight_layout()
        fig.savefig(figures_dir / "delta_finetuned_vs_base.pdf")
        plt.close(fig)
        print(f"  Saved: delta_finetuned_vs_base.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Console table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(label, agg):
    strats = [s for s in STRATEGY_ORDER if s in agg]
    print(f"\n{'-'*72}")
    print(f"  {label}")
    print(f"{'-'*72}")
    print(f"  {'Strategy':<16} {'mAP':>7} {'MRR':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7}  Queries")
    print(f"  {'-'*68}")
    for s in strats:
        r = agg[s]
        print(f"  {s:<16} {r['mAP']:>7.3f} {r['MRR']:>7.3f} "
              f"{r['R@1']:>7.3f} {r['R@5']:>7.3f} {r['R@10']:>7.3f}  {r['n_queries']}")
    print(f"{'-'*72}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--also-base",   action="store_true")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--val-pairs",   default=str(DEFAULT_VAL_PAIRS))
    parser.add_argument("--metadata",    default=str(DEFAULT_METADATA))
    parser.add_argument("--labels",      default=str(DEFAULT_LABELS))
    parser.add_argument("--taxonomy",    default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--audio-root",  default=str(DEFAULT_AUDIO_ROOT))
    parser.add_argument("--output",      default=str(DEFAULT_OUTPUT))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES))
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--no-plots",    action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load data
    val_pairs  = json.loads(Path(args.val_pairs).read_text(encoding="utf-8"))
    all_labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    tax_db     = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))

    df       = pd.read_csv(args.metadata, encoding="utf-8")
    name_col = next((c for c in ("common_name", "species", "name") if c in df.columns), None)
    type_col = next((c for c in ("vocalization_type", "type") if c in df.columns), None)
    path_col = next((c for c in ("filepath", "file_path", "path", "filename") if c in df.columns), None)
    clip_to_combo = {
        str(row[path_col]).strip(): (
            str(row[name_col]).strip(),
            str(row[type_col]).split(",")[0].strip().lower()
        )
        for _, row in df.iterrows()
    }

    seen, val_clips = set(), []
    for p in val_pairs:
        if p["audio"] not in seen:
            seen.add(p["audio"]); val_clips.append(p["audio"])

    val_combos = list({clip_to_combo[c] for c in val_clips if c in clip_to_combo})
    print(f"Val clips: {len(val_clips)}  |  (species, type) combos: {len(val_combos)}")

    strategies = build_queries(val_combos, tax_db, all_labels)
    for s, q in strategies.items():
        print(f"  {s:<16}: {len(q)} queries")

    audio_root  = Path(args.audio_root)
    all_results = {}   # {model_key: {"agg": ..., "detail": ..., "sim_data": ...}}

    def evaluate(checkpoint, label):
        model, processor = load_model(checkpoint, args.model, device)
        agg, detail, sim_data = run_eval(
            model, processor, val_clips, clip_to_combo,
            strategies, audio_root, device, args.batch_size)
        del model
        print_table(label, agg)
        return {"agg": agg, "detail": detail, "sim_data": sim_data}

    if args.checkpoint:
        print(f"\n{'='*72}\n  Evaluating: fine-tuned  ({args.checkpoint})")
        all_results["finetuned"] = evaluate(args.checkpoint, f"Fine-tuned: {args.checkpoint}")

    if args.also_base or not args.checkpoint:
        print(f"\n{'='*72}\n  Evaluating: base model  ({args.model})")
        all_results["base"] = evaluate(None, f"Base model (zero-shot): {args.model}")

    # Delta table
    if "finetuned" in all_results and "base" in all_results:
        print(f"\n{'='*72}\n  Delta (fine-tuned - base)")
        print(f"{'-'*72}")
        print(f"  {'Strategy':<16} {'d mAP':>7} {'d MRR':>7} {'d R@1':>7} {'d R@5':>7} {'d R@10':>7}")
        print(f"  {'-'*68}")
        for s in STRATEGY_ORDER:
            if s not in all_results["finetuned"]["agg"]: continue
            ft   = all_results["finetuned"]["agg"][s]
            base = all_results["base"]["agg"][s]
            def d(k): return ft[k] - base[k]
            print(f"  {s:<16} {d('mAP'):>+7.3f} {d('MRR'):>+7.3f} "
                  f"{d('R@1'):>+7.3f} {d('R@5'):>+7.3f} {d('R@10'):>+7.3f}")
        print(f"{'-'*72}")

    # Save JSON (exclude raw sim arrays to keep file small)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for k, v in all_results.items():
        save_data[k] = {
            "agg":    v["agg"],
            "detail": v["detail"],   # per-combo metrics, no raw sims
        }
    out_path.write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "base_model": args.model,
                "n_val_clips": len(val_clips),
                "n_val_combos": len(val_combos),
                "results": save_data,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nResults saved -> {out_path}")

    # Generate plots
    if not args.no_plots:
        print(f"\nGenerating plots -> {args.figures_dir}/")
        generate_plots(all_results, tax_db, Path(args.figures_dir))
        print(f"All plots saved.")


if __name__ == "__main__":
    main()
