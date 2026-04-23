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
  1. strategy_comparison.pdf          — mAP / MRR / R@K bar chart per strategy
  2. class_breakdown_{model}.pdf      — mAP by taxonomic class × strategy
  3. hardest_easiest_{model}.pdf      — top-20 hardest / easiest species by mAP
  4. rank_cdf_{model}.pdf             — CDF of first-hit rank (full gallery range)
  5. delta_finetuned_vs_base.pdf      — Δmetric bar chart  (only with --also-base)
  Semantic (opt-in, one file per model):
  6. semantic_probe_{model}.pdf       — R@K per hand-written acoustic query
  7. acoustic_coherence_{model}.pdf   — genus/family R@K vs random baseline
  8. cross_species_transfer_{m}.pdf   — description transfer R@K by genus

Semantic search evaluations (opt-in):
  --semantic-queries data/my_queries.json
      Evaluate hand-written purely-acoustic queries (no species names) against
      the full gallery. See data/semantic_queries_example.json for format.
  --acoustic-coherence
      Encodes one rich acoustic description per combo, measures whether top-K
      nearest neighbours in text-embedding space share the same genus/family.
      Genus R@K >> random baseline proves the model learned acoustic structure.
  --cross-species-eval
      Uses species A's acoustic description to retrieve species B's audio
      (same genus, different species). Above-random R@K means the model
      learned transferable acoustic properties beyond species identity.

Query strategies now return lists of texts; run_eval averages embeddings before
computing similarity. The "all_variants" strategy uses all 8 label variants per
combo (5 taxonomy + 3 rich) as an ensemble — free improvement at inference time.

Audio gallery is encoded once per model and reused across all eval functions
(encode_gallery is called once inside evaluate(), not separately per eval).

Usage:
    conda activate birdclap

    python scripts/evaluate_clap.py --checkpoint checkpoints/best.pt
    python scripts/evaluate_clap.py --checkpoint checkpoints/best.pt --also-base
    python scripts/evaluate_clap.py          # base model only

    # Full semantic eval suite
    python scripts/evaluate_clap.py --checkpoint checkpoints/best.pt \
        --semantic-queries data/semantic_queries_example.json \
        --acoustic-coherence \
        --cross-species-eval
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL         = "laion/clap-htsat-fused"
DEFAULT_VAL_PAIRS     = Path("data/clap_val_pairs.json")
DEFAULT_HOLDOUT_PAIRS = Path("data/clap_holdout_pairs.json")
DEFAULT_HOLDOUT_DESCS = Path("data/clap_descriptions_holdout.json")
DEFAULT_LABELS        = Path("data/clap_all_labels.json")
DEFAULT_TAXONOMY      = Path("data/species_taxonomy.json")
DEFAULT_AUDIO_ROOT    = Path(".")
DEFAULT_OUTPUT        = Path("results/eval_results.json")
DEFAULT_FIGURES       = Path("results/figures")
AUDIO_SR              = 48_000
CLIP_SECONDS          = 10
MIN_DURATION_S        = 0.5  # match scripts/train_clap.py
STRATEGY_ORDER        = ["name", "scientific", "chain", "sci_common", "chain_common",
                         "rich", "rich_holdout", "all_variants"]
STRATEGY_LABELS       = {
    "name":         "Common name",
    "scientific":   "Scientific name",
    "chain":        "Taxonomy chain",
    "sci_common":   "Sci + common",
    "chain_common": "Chain + common",
    "rich":         "Rich description",
    "rich_holdout": "Rich (held-out)",
    "all_variants": "Ensemble (all)",
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

    # Rank of first positive hit (1-indexed); 0 if no positives
    first_hit_rank = 0
    for rank, idx in enumerate(ranked, 1):
        if idx in pos_set:
            first_hit_rank = rank
            break

    # Median rank of ALL positives in the gallery
    pos_ranks = [rank for rank, idx in enumerate(ranked, 1) if idx in pos_set]
    median_pos_rank = float(np.median(pos_ranks)) if pos_ranks else float(n_total)

    # Positive / negative similarity distributions
    pos_sims = sim_row[positive_indices].tolist()
    neg_mask = np.ones(n_total, dtype=bool)
    neg_mask[positive_indices] = False
    neg_sims = sim_row[neg_mask].tolist()

    return {
        "mAP": ap,
        "MRR": mrr,
        "recall_at_k": recall_at_k,
        "first_hit_rank": first_hit_rank,
        "median_pos_rank": median_pos_rank,
        "pos_sims": pos_sims,
        "neg_sims": neg_sims,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Query builders
# ─────────────────────────────────────────────────────────────────────────────

def build_queries(combos, tax_db, all_labels, holdout_descs=None):
    """
    Returns strategies dict where each value is a list of query texts.
    Single-text strategies use a one-element list; the ensemble strategy
    uses all available variants. run_eval encodes all texts and averages
    their embeddings before computing the similarity matrix.
    """
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

        # Individual strategies — single text wrapped in list
        strategies["name"][(name, vtype)] = [f"{name} {vt}"]
        if sci:
            strategies["scientific"][(name, vtype)]   = [f"{sci} {vt}"]
            strategies["chain"][(name, vtype)]         = [f"{chain} {vt}"]
            strategies["sci_common"][(name, vtype)]    = [f"{sci}, {name} {vt}"]
            strategies["chain_common"][(name, vtype)]  = [f"{chain}, {name} {vt}"]

        all_label_variants = all_labels.get(key, [])
        tax_templates = all_label_variants[:5]   # indices 0-4: taxonomy
        rich_variants = all_label_variants[5:]   # indices 5+: LLM descriptions

        # rich: single rotating acoustic description (trained on)
        if rich_variants:
            idx = hash(key) % len(rich_variants)
            strategies["rich"][(name, vtype)] = [rich_variants[idx]]

        # rich_holdout: description held back from training — unseen acoustic query
        if holdout_descs:
            held = holdout_descs.get(key, [])
            if held:
                idx = hash(key) % len(held)
                strategies["rich_holdout"][(name, vtype)] = [held[idx]]

        # all_variants: ensemble of ALL available texts for this combo.
        # At query time, all texts are encoded and their embeddings are averaged
        # before computing similarity — free improvement, no retraining needed.
        ensemble_texts = (tax_templates if sci else [f"{name} {vt}"]) + rich_variants
        if ensemble_texts:
            strategies["all_variants"][(name, vtype)] = ensemble_texts

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
# Gallery encoder (shared across all eval functions)
# ─────────────────────────────────────────────────────────────────────────────

def encode_gallery(model, processor, val_clips, audio_root, device, batch_size=16):
    """
    Encode all val audio clips to normalised embeddings once.
    Call this once per model and pass the result to run_eval / semantic evals
    to avoid redundant encoding.

    Returns:
        audio_matrix : np.ndarray (N, D), L2-normalised
        valid_clips  : list[str] — clips that loaded successfully (same order)
    """
    print(f"\n  Encoding {len(val_clips)} val audio clips ...")
    audio_embs, failed = [], []
    for i in range(0, len(val_clips), batch_size):
        batch_paths = val_clips[i : i + batch_size]
        wavs = []
        for p in batch_paths:
            wav = load_audio(audio_root / p)
            if wav is not None:
                wavs.append(wav)
            else:
                failed.append(p)
        if wavs:
            audio_embs.append(encode_audio_batch(wavs, processor, model, device))
        if (i // batch_size) % 10 == 0:
            print(f"    {min(i + batch_size, len(val_clips))}/{len(val_clips)}", end="\r")
    if failed:
        print(f"\n  [warn] {len(failed)} clips failed to load - excluded")
    valid_clips = [c for c in val_clips if c not in set(failed)]
    print(f"\n  Audio encoding done. ({len(valid_clips)} clips)")
    audio_matrix = torch.cat(audio_embs, dim=0).numpy()
    return audio_matrix, valid_clips


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(model, processor, val_clips, clip_to_combo,
             strategies, audio_root, device, batch_size=16,
             audio_matrix=None, valid_clips_in=None):
    """
    Returns:
        agg     : {strategy: {metric: float}}   — macro-averaged metrics
        detail  : {strategy: [{combo, mAP, MRR, R@k, tax_class, ...}]}
        sim_data: {strategy: {"pos": [...], "neg": [...]}}

    If audio_matrix + valid_clips_in are provided (from encode_gallery), audio
    encoding is skipped. Otherwise encode_gallery is called internally.
    """
    # Encode audio (or reuse pre-encoded gallery)
    if audio_matrix is not None and valid_clips_in is not None:
        valid_clips = valid_clips_in
        print(f"\n  [run_eval] using pre-encoded gallery ({len(valid_clips)} clips)")
    else:
        audio_matrix, valid_clips = encode_gallery(
            model, processor, val_clips, audio_root, device, batch_size)

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

        n_variants_avg = sum(len(q) for _, q in eval_combos) / len(eval_combos)
        print(f"  [{strategy}] encoding {len(eval_combos)} queries "
              f"(avg {n_variants_avg:.1f} texts/combo) ...")

        # Flatten all per-combo texts into one list, encode in batches of 64,
        # then fold back and average per-combo embeddings (re-normalised).
        flat_texts = [t for _, q in eval_combos for t in q]
        n_per_combo = [len(q) for _, q in eval_combos]
        flat_embs = []
        for i in range(0, len(flat_texts), 64):
            flat_embs.append(encode_text_batch(flat_texts[i:i+64], processor, model, device))
        flat_embs = torch.cat(flat_embs, dim=0)   # (total_texts, D)

        # Average and re-normalise per combo
        combo_embs = []
        offset = 0
        for n in n_per_combo:
            mean_emb = flat_embs[offset : offset + n].mean(dim=0)
            combo_embs.append(F.normalize(mean_emb, dim=0))
            offset += n
        text_matrix = torch.stack(combo_embs).numpy()   # (n_combos, D)
        sim_matrix  = text_matrix @ audio_matrix.T

        per_combo, all_pos_sims, all_neg_sims = [], [], []
        n_gallery = len(valid_clips)
        for q_idx, (combo, _) in enumerate(eval_combos):
            m = retrieval_metrics(sim_matrix[q_idx], combo_to_indices[combo])
            per_combo.append({
                "species":         combo[0],
                "voc_type":        combo[1],
                "mAP":             m["mAP"],
                "MRR":             m["MRR"],
                "recall_at_k":     m["recall_at_k"],
                "n_positives":     len(combo_to_indices[combo]),
                "first_hit_rank":  m["first_hit_rank"],
                "median_pos_rank": m["median_pos_rank"],
            })
            all_pos_sims.extend(m["pos_sims"])
            all_neg_sims.extend(m["neg_sims"])

        # Macro-average
        a = {}
        for metric in ["mAP", "MRR"]:
            a[metric] = float(np.mean([c[metric] for c in per_combo]))
        for k in range(1, 21):
            a[f"R@{k}"] = float(np.mean([c["recall_at_k"][k] for c in per_combo]))
        a["n_queries"]        = len(eval_combos)
        a["median_first_rank"]= float(np.median([c["first_hit_rank"] for c in per_combo]))
        a["mean_first_rank"]  = float(np.mean([c["first_hit_rank"] for c in per_combo]))
        a["median_pos_rank"]  = float(np.median([c["median_pos_rank"] for c in per_combo]))
        # Percentile of first hit: how far into the gallery must you scroll on average?
        a["first_rank_pct"]   = float(np.mean(
            [c["first_hit_rank"] / n_gallery for c in per_combo]
        ))
        agg[strategy]      = a
        detail[strategy]   = per_combo
        sim_data[strategy] = {"pos": all_pos_sims, "neg": all_neg_sims}

        print(f"    mAP={a['mAP']:.3f}  MRR={a['MRR']:.3f}  "
              f"R@1={a['R@1']:.3f}  R@5={a['R@5']:.3f}  R@10={a['R@10']:.3f}  "
              f"median_rank={a['median_first_rank']:.0f}/{n_gallery}  "
              f"({a['n_queries']} queries)")

    return agg, detail, sim_data


# ─────────────────────────────────────────────────────────────────────────────
# Semantic search evaluations
# ─────────────────────────────────────────────────────────────────────────────

def run_semantic_probe(audio_matrix, valid_clips, clip_to_combo,
                       model, processor, probe_queries, device, batch_size=64):
    """
    Evaluate hand-written purely-acoustic text queries against the full gallery.

    probe_queries: list of dicts loaded from --semantic-queries JSON:
        [{"query": "rising two-note whistle with a descending slur",
          "positive_combos": ["American Robin||song"]}, ...]
    The queries must NOT contain species names — they should describe only
    acoustic properties so the eval is a true test of acoustic understanding.

    Returns:
        results : list of per-query metric dicts
        summary : macro-averaged metric dict
    """
    if not probe_queries:
        return [], {}

    combo_to_indices: dict[tuple, list[int]] = defaultdict(list)
    for idx, clip in enumerate(valid_clips):
        combo = clip_to_combo.get(clip)
        if combo:
            combo_to_indices[combo].append(idx)

    # Encode all probe texts in one pass
    texts = [q["query"] for q in probe_queries]
    flat_embs = []
    for i in range(0, len(texts), batch_size):
        flat_embs.append(encode_text_batch(texts[i : i + batch_size], processor, model, device))
    text_matrix = torch.cat(flat_embs, dim=0).numpy()   # (n_queries, D)
    sim_matrix  = text_matrix @ audio_matrix.T           # (n_queries, n_gallery)

    results = []
    for q_idx, probe in enumerate(probe_queries):
        pos_indices: list[int] = []
        for combo_str in probe.get("positive_combos", []):
            if "||" in combo_str:
                sp, vt = combo_str.split("||", 1)
                pos_indices.extend(combo_to_indices.get((sp.strip(), vt.strip().lower()), []))
        if not pos_indices:
            print(f"    [semantic_probe] '{probe['query'][:60]}' — no positives found, skipped")
            continue
        m = retrieval_metrics(sim_matrix[q_idx], pos_indices)
        results.append({
            "query":          probe["query"],
            "n_positives":    len(pos_indices),
            "mAP":            m["mAP"],
            "MRR":            m["MRR"],
            "recall_at_k":    m["recall_at_k"],
            "first_hit_rank": m["first_hit_rank"],
        })

    if not results:
        return results, {}

    fhr = [r["first_hit_rank"] for r in results if r["first_hit_rank"] > 0]
    summary = {
        "mAP":               float(np.mean([r["mAP"] for r in results])),
        "MRR":               float(np.mean([r["MRR"] for r in results])),
        "R@1":               float(np.mean([r["recall_at_k"][1] for r in results])),
        "R@5":               float(np.mean([r["recall_at_k"][5] for r in results])),
        "R@10":              float(np.mean([r["recall_at_k"][10] for r in results])),
        "median_first_rank": float(np.median(fhr)) if fhr else float(len(valid_clips)),
        "n_queries":         len(results),
    }
    n_gal = len(valid_clips)
    print(f"  [semantic_probe] mAP={summary['mAP']:.3f}  "
          f"R@1={summary['R@1']:.3f}  R@5={summary['R@5']:.3f}  "
          f"median_rank={summary['median_first_rank']:.0f}/{n_gal}  "
          f"({summary['n_queries']} queries)")
    return results, summary


def run_acoustic_coherence(model, processor, combos, tax_db, all_labels,
                            device, batch_size=64, k_vals=(1, 5, 10)):
    """
    Measure whether the text embedding space has learned acoustic structure.

    Encodes one rich acoustic description per (species, type) combo.  For each
    combo, finds its K nearest neighbours in text-embedding space and reports
    what fraction share the same genus / family vs a random-pick baseline.

    If the model learned acoustics from descriptions, similar-sounding species
    (congeneric ones are a decent proxy) should cluster together in embedding
    space — genus_R@K >> random baseline.

    Returns:
        results : list of per-combo dicts
        summary : {"genus_R@{k}": ..., "family_R@{k}": ...,
                   "random_genus_R@{k}": ..., ...}
    """
    combo_texts: dict[tuple, str] = {}
    for name, vtype in combos:
        key   = f"{name}||{vtype}"
        rich  = all_labels.get(key, [])[5:]
        if rich:
            combo_texts[(name, vtype)] = rich[hash(key) % len(rich)]

    if len(combo_texts) < 10:
        print(f"  [acoustic_coherence] only {len(combo_texts)} combos have rich descriptions — skipping")
        return [], {}

    combos_list = list(combo_texts.keys())
    texts       = [combo_texts[c] for c in combos_list]
    print(f"  [acoustic_coherence] encoding {len(texts)} acoustic descriptions ...")

    flat_embs = []
    for i in range(0, len(texts), batch_size):
        flat_embs.append(encode_text_batch(texts[i : i + batch_size], processor, model, device))
    emb_matrix = torch.cat(flat_embs, dim=0).numpy()   # (N, D) normalised

    sim_matrix = emb_matrix @ emb_matrix.T             # (N, N) cosine similarity
    np.fill_diagonal(sim_matrix, -2.0)                  # exclude self

    def genus(name):  return tax_db.get(name, {}).get("genus",  "")
    def family(name): return tax_db.get(name, {}).get("family", "")

    # Pre-count genus/family sizes for normalisation
    genus_counts:  dict[str, int] = defaultdict(int)
    family_counts: dict[str, int] = defaultdict(int)
    for name, _ in combos_list:
        if genus(name):  genus_counts[genus(name)]   += 1
        if family(name): family_counts[family(name)] += 1

    n = len(combos_list)
    results = []
    for i, (name, vtype) in enumerate(combos_list):
        ranked = np.argsort(sim_matrix[i])[::-1]
        row    = {"species": name, "voc_type": vtype,
                  "genus": genus(name), "family": family(name)}
        for k in k_vals:
            top_k = ranked[:k]
            row[f"genus_hits@{k}"]  = sum(1 for j in top_k
                                          if genus(combos_list[j][0])  == genus(name)  and genus(name))
            row[f"family_hits@{k}"] = sum(1 for j in top_k
                                          if family(combos_list[j][0]) == family(name) and family(name))
        results.append(row)

    summary: dict = {"n_combos": n}
    for k in k_vals:
        genus_recalls, family_recalls = [], []
        genus_randoms, family_randoms = [], []
        for i, (name, _) in enumerate(combos_list):
            g = genus(name); f = family(name)
            max_g = min(k, max(genus_counts[g]  - 1, 0))
            max_f = min(k, max(family_counts[f] - 1, 0))
            if max_g > 0:
                genus_recalls.append(results[i][f"genus_hits@{k}"]  / max_g)
                genus_randoms.append((genus_counts[g] - 1) / (n - 1))
            if max_f > 0:
                family_recalls.append(results[i][f"family_hits@{k}"] / max_f)
                family_randoms.append((family_counts[f] - 1) / (n - 1))
        summary[f"genus_R@{k}"]         = float(np.mean(genus_recalls))  if genus_recalls  else 0.0
        summary[f"family_R@{k}"]        = float(np.mean(family_recalls)) if family_recalls else 0.0
        summary[f"random_genus_R@{k}"]  = float(np.mean(genus_randoms))  if genus_randoms  else 0.0
        summary[f"random_family_R@{k}"] = float(np.mean(family_randoms)) if family_randoms else 0.0

    print(f"  [acoustic_coherence] text-embedding neighbour analysis:")
    for k in k_vals:
        print(f"    K={k:2d}  "
              f"genus_R@K={summary[f'genus_R@{k}']:.3f} (random={summary[f'random_genus_R@{k}']:.3f})  "
              f"family_R@K={summary[f'family_R@{k}']:.3f} (random={summary[f'random_family_R@{k}']:.3f})")
    return results, summary


def run_cross_species_transfer(audio_matrix, valid_clips, clip_to_combo,
                                model, processor, tax_db, all_labels,
                                device, batch_size=64):
    """
    Cross-species description transfer evaluation.

    For each (species_A, voc_type) combo, takes A's acoustic description and
    queries the gallery for species B's audio clips, where B is a different
    species in the same genus as A.

    Interpretation: if the model learned transferable acoustic features,
    a description of A's call should retrieve B's call better than random
    (both A and B share genus-level acoustic similarities).

    Returns:
        results : list of per-query-pair dicts
        summary : transfer metrics including random baseline
    """
    combo_to_indices: dict[tuple, list[int]] = defaultdict(list)
    for idx, clip in enumerate(valid_clips):
        combo = clip_to_combo.get(clip)
        if combo:
            combo_to_indices[combo].append(idx)

    # Group combos by genus, keep only those in gallery
    genus_to_combos: dict[str, list[tuple]] = defaultdict(list)
    for combo in combo_to_indices:
        name, vtype = combo
        g = tax_db.get(name, {}).get("genus", "")
        if g:
            genus_to_combos[g].append(combo)

    # Genera with multiple species in gallery
    multi_genus = {g: cs for g, cs in genus_to_combos.items()
                   if len({c[0] for c in cs}) >= 2}

    if not multi_genus:
        print("  [cross_species_transfer] no genera with ≥2 species in gallery — skipping")
        return [], {}

    # Build transfer pairs: (query_text, target_combo, source_species, genus)
    transfer_pairs = []
    for genus_name, genus_combos in multi_genus.items():
        species_set = {c[0] for c in genus_combos}
        for name_a, vtype_a in genus_combos:
            key_a = f"{name_a}||{vtype_a}"
            rich_a = all_labels.get(key_a, [])[5:]
            if not rich_a:
                continue
            query_text = rich_a[hash(key_a) % len(rich_a)]
            for name_b in species_set:
                if name_b == name_a:
                    continue
                target = (name_b, vtype_a)
                if combo_to_indices.get(target):
                    transfer_pairs.append((query_text, target, name_a, genus_name))

    if not transfer_pairs:
        print("  [cross_species_transfer] no valid cross-species pairs — skipping")
        return [], {}

    n_genera_used = len({p[3] for p in transfer_pairs})
    print(f"  [cross_species_transfer] {len(transfer_pairs)} query pairs "
          f"across {n_genera_used} genera ...")

    texts = [p[0] for p in transfer_pairs]
    flat_embs = []
    for i in range(0, len(texts), batch_size):
        flat_embs.append(encode_text_batch(texts[i : i + batch_size], processor, model, device))
    text_matrix = torch.cat(flat_embs, dim=0).numpy()
    sim_matrix  = text_matrix @ audio_matrix.T

    n_gallery = len(valid_clips)
    results = []
    for q_idx, (query_text, target_combo, src_species, genus_name) in enumerate(transfer_pairs):
        pos_indices = combo_to_indices[target_combo]
        m = retrieval_metrics(sim_matrix[q_idx], pos_indices)
        results.append({
            "source_species":  src_species,
            "target_species":  target_combo[0],
            "voc_type":        target_combo[1],
            "genus":           genus_name,
            "n_positives":     len(pos_indices),
            "mAP":             m["mAP"],
            "MRR":             m["MRR"],
            "recall_at_k":     m["recall_at_k"],
            "first_hit_rank":  m["first_hit_rank"],
        })

    avg_n_pos  = np.mean([r["n_positives"] for r in results])
    random_r1  = avg_n_pos / n_gallery
    fhr = [r["first_hit_rank"] for r in results if r["first_hit_rank"] > 0]
    summary = {
        "mAP":               float(np.mean([r["mAP"] for r in results])),
        "MRR":               float(np.mean([r["MRR"] for r in results])),
        "R@1":               float(np.mean([r["recall_at_k"][1] for r in results])),
        "R@5":               float(np.mean([r["recall_at_k"][5] for r in results])),
        "R@10":              float(np.mean([r["recall_at_k"][10] for r in results])),
        "median_first_rank": float(np.median(fhr)) if fhr else float(n_gallery),
        "random_R@1":        float(random_r1),
        "n_queries":         len(results),
        "n_genera":          n_genera_used,
    }
    print(f"  [cross_species_transfer] "
          f"R@1={summary['R@1']:.3f} (random={summary['random_R@1']:.4f})  "
          f"R@5={summary['R@5']:.3f}  mAP={summary['mAP']:.3f}  "
          f"({summary['n_queries']} pairs, {summary['n_genera']} genera)")
    return results, summary


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

    # ── 2. Taxonomic class breakdown ──────────────────────────────────────────
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

    # ── 3. Hardest / easiest species (using 'name' strategy) ─────────────────
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

    # ── 4. Rank CDF ───────────────────────────────────────────────────────────
    # Cumulative fraction of queries where first hit ≤ k — complements R@K curve
    # but on the full rank range so you can see the long tail.
    for model_key, res in all_results.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title(f"Rank CDF (fraction of queries with first hit ≤ rank) — {model_key}",
                     fontweight="bold")
        max_rank_cdf = 0
        for i, s in enumerate(strategies_present):
            ranks = sorted(c["first_hit_rank"] for c in res["detail"].get(s, [])
                           if c["first_hit_rank"] > 0)
            if not ranks:
                continue
            max_rank_cdf = max(max_rank_cdf, ranks[-1])
            n = len(ranks)
            ax.step([0] + ranks, np.linspace(0, 1, n + 1),
                    color=STRATEGY_COLORS[i], linewidth=1.5,
                    label=STRATEGY_LABELS.get(s, s))
        ax.axvline(1,  color="gray", linewidth=0.7, linestyle=":")
        ax.axvline(5,  color="gray", linewidth=0.7, linestyle=":")
        ax.axvline(10, color="gray", linewidth=0.7, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlim(1, max(max_rank_cdf, 10))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Rank (log scale)")
        ax.set_ylabel("Cumulative fraction of queries")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / f"rank_cdf_{model_key}.pdf")
        plt.close(fig)
        print(f"  Saved: rank_cdf_{model_key}.pdf")

    # ── 5. Delta chart (fine-tuned − base) ────────────────────────────────────
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


def generate_semantic_plots(semantic_results: dict, figures_dir: Path):
    """
    Generate plots for the three semantic search evaluations:
      10. semantic_probe_results.pdf   — per-query R@K bars
      11. acoustic_coherence.pdf       — genus/family R@K vs random baseline
      12. cross_species_transfer.pdf   — transfer R@K by genus, vs random
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.3,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 200,
    })

    for model_key, sem in semantic_results.items():
        # ── 10. Semantic probe: per-query R@K bar chart ───────────────────────
        probe_results = sem.get("semantic_probe_results", [])
        if probe_results:
            ks     = [1, 5, 10]
            labels = [r["query"][:45] + "…" if len(r["query"]) > 45 else r["query"]
                      for r in probe_results]
            n_q    = len(probe_results)
            x      = np.arange(n_q)
            width  = 0.25
            colors = ["#2166ac", "#5aae61", "#d6604d"]

            fig, ax = plt.subplots(figsize=(max(8, n_q * 1.5 + 2), 4.5))
            for i, k in enumerate(ks):
                vals = [r["recall_at_k"][k] for r in probe_results]
                ax.bar(x + (i - 1) * width, vals, width, label=f"R@{k}",
                       color=colors[i], alpha=0.85, edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Recall@K")
            ax.set_title(f"Semantic Probe: Purely-Acoustic Query Results — {model_key}",
                         fontweight="bold")
            ax.legend(fontsize=9, frameon=False)
            fig.tight_layout()
            fig.savefig(figures_dir / f"semantic_probe_{model_key}.pdf")
            plt.close(fig)
            print(f"  Saved: semantic_probe_{model_key}.pdf")

        # ── 11. Acoustic coherence: genus/family R@K vs random ───────────────
        coh_summary = sem.get("acoustic_coherence_summary", {})
        if coh_summary and "n_combos" in coh_summary:
            k_vals = sorted({int(key.split("@")[1]) for key in coh_summary
                             if key.startswith("genus_R@")})
            if k_vals:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
                fig.suptitle(f"Acoustic Coherence: Embedding-Space Neighbour Analysis — {model_key}",
                             fontweight="bold")
                for ax, level, color, rand_color in [
                    (ax1, "genus",  "#2166ac", "#abd9e9"),
                    (ax2, "family", "#5aae61", "#a6d96a"),
                ]:
                    model_vals  = [coh_summary.get(f"{level}_R@{k}", 0) for k in k_vals]
                    random_vals = [coh_summary.get(f"random_{level}_R@{k}", 0) for k in k_vals]
                    x = np.arange(len(k_vals))
                    ax.bar(x - 0.2, model_vals,  0.38, label="Model",  color=color,      alpha=0.9)
                    ax.bar(x + 0.2, random_vals, 0.38, label="Random", color=rand_color, alpha=0.75)
                    for xi, (mv, rv) in enumerate(zip(model_vals, random_vals)):
                        ax.text(xi - 0.2, mv + 0.01, f"{mv:.2f}", ha="center", fontsize=8)
                        ax.text(xi + 0.2, rv + 0.01, f"{rv:.2f}", ha="center", fontsize=8)
                    ax.set_xticks(x)
                    ax.set_xticklabels([f"K={k}" for k in k_vals])
                    ax.set_ylim(0, 1.1)
                    ax.set_ylabel("R@K (normalised)")
                    ax.set_title(f"Same-{level} neighbour recall")
                    ax.legend(fontsize=9, frameon=False)
                fig.tight_layout()
                fig.savefig(figures_dir / f"acoustic_coherence_{model_key}.pdf")
                plt.close(fig)
                print(f"  Saved: acoustic_coherence_{model_key}.pdf")

        # ── 12. Cross-species transfer: R@K by genus ─────────────────────────
        xfer_results = sem.get("cross_species_transfer_results", [])
        xfer_summary = sem.get("cross_species_transfer_summary", {})
        if xfer_results and xfer_summary:
            # Aggregate R@K per genus
            genus_map: dict[str, list] = defaultdict(list)
            for r in xfer_results:
                genus_map[r["genus"]].append(r)
            genera = sorted(genus_map.keys(),
                            key=lambda g: -np.mean([r["recall_at_k"][1] for r in genus_map[g]]))[:20]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Cross-Species Description Transfer — {model_key}", fontweight="bold")

            # Left: R@1 per genus (top-20 by R@1)
            r1_per_genus = [np.mean([r["recall_at_k"][1] for r in genus_map[g]]) for g in genera]
            y = np.arange(len(genera))
            colors_bar = ["#2166ac" if v > xfer_summary.get("random_R@1", 0) else "#d6604d"
                          for v in r1_per_genus]
            ax1.barh(y, r1_per_genus, color=colors_bar, alpha=0.85, edgecolor="white")
            ax1.axvline(xfer_summary.get("random_R@1", 0), color="gray",
                        linewidth=1.2, linestyle="--", label="Random baseline")
            ax1.set_yticks(y); ax1.set_yticklabels(genera, fontsize=8)
            ax1.set_xlabel("R@1"); ax1.set_title("R@1 per genus (top-20 by R@1)")
            ax1.invert_yaxis(); ax1.legend(fontsize=8, frameon=False)

            # Right: overall R@K curve vs random baseline
            ks_xfer = [1, 2, 3, 5, 10]
            model_rk  = [xfer_summary.get(f"R@{k}", 0) for k in ks_xfer]
            ax2.plot(ks_xfer, model_rk, "o-", color="#2166ac", linewidth=2, label="Transfer R@K")
            ax2.axhline(xfer_summary.get("random_R@1", 0), color="gray",
                        linewidth=1.2, linestyle="--", label="Random baseline (R@1)")
            ax2.set_xlabel("K"); ax2.set_ylabel("Recall@K")
            ax2.set_title("Overall transfer R@K vs random")
            ax2.set_ylim(0, max(max(model_rk) * 1.2, 0.05))
            ax2.legend(fontsize=9, frameon=False)

            fig.tight_layout()
            fig.savefig(figures_dir / f"cross_species_transfer_{model_key}.pdf")
            plt.close(fig)
            print(f"  Saved: cross_species_transfer_{model_key}.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Console table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(label, agg):
    strats = [s for s in STRATEGY_ORDER if s in agg]
    print(f"\n{'-'*90}")
    print(f"  {label}")
    print(f"{'-'*90}")
    print(f"  {'Strategy':<16} {'mAP':>7} {'MRR':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7}"
          f"  {'MedRank':>8} {'MeanRank':>9}  Queries")
    print(f"  {'-'*86}")
    for s in strats:
        r = agg[s]
        med_rank  = r.get("median_first_rank", float("nan"))
        mean_rank = r.get("mean_first_rank",   float("nan"))
        print(f"  {s:<16} {r['mAP']:>7.3f} {r['MRR']:>7.3f} "
              f"{r['R@1']:>7.3f} {r['R@5']:>7.3f} {r['R@10']:>7.3f}"
              f"  {med_rank:>8.0f} {mean_rank:>9.1f}  {r['n_queries']}")
    print(f"{'-'*90}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     default=None)
    parser.add_argument("--also-base",      action="store_true")
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--val-pairs",      default=str(DEFAULT_VAL_PAIRS))
    parser.add_argument("--holdout-pairs",  default=str(DEFAULT_HOLDOUT_PAIRS),
                        help="Path to clap_holdout_pairs.json for zero-shot species eval")
    parser.add_argument("--holdout-descs",  default=str(DEFAULT_HOLDOUT_DESCS),
                        help="Path to clap_descriptions_holdout.json for held-out acoustic query eval")
    parser.add_argument("--labels",         default=str(DEFAULT_LABELS))
    parser.add_argument("--taxonomy",       default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--audio-root",     default=str(DEFAULT_AUDIO_ROOT))
    parser.add_argument("--output",         default=str(DEFAULT_OUTPUT))
    parser.add_argument("--figures-dir",    default=str(DEFAULT_FIGURES))
    parser.add_argument("--batch-size",     type=int, default=16)
    parser.add_argument("--device",         default=None)
    parser.add_argument("--no-plots",       action="store_true", help="Skip plot generation")
    # Semantic search evaluations
    parser.add_argument("--semantic-queries",  default=None, metavar="JSON",
                        help="Path to hand-written acoustic probe queries JSON "
                             "(see data/semantic_queries_example.json for format). "
                             "Queries must NOT contain species names.")
    parser.add_argument("--acoustic-coherence", action="store_true",
                        help="Run acoustic coherence eval: measures whether same-genus "
                             "species cluster together in text-embedding space")
    parser.add_argument("--cross-species-eval", action="store_true",
                        help="Run cross-species description transfer: tests if species A's "
                             "acoustic description retrieves species B's audio (same genus)")
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

    # Load held-out descriptions if available
    holdout_descs: dict = {}
    holdout_desc_path = Path(args.holdout_descs)
    if holdout_desc_path.exists():
        holdout_descs = json.loads(holdout_desc_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(holdout_descs)} held-out description keys from {holdout_desc_path}")
    else:
        print(f"[info] No held-out descriptions at {holdout_desc_path} — rich_holdout strategy skipped")

    # Build clip→combo mapping from the pairs JSON itself (every pair has a
    # "combo" field like "Virginia Rail||call"), so we don't need the metadata CSV.
    # Merge all pair sources so holdout clips are also covered.
    all_pair_sources = list(val_pairs)
    holdout_pairs_path = Path(args.holdout_pairs)
    if holdout_pairs_path.exists():
        all_pair_sources += json.loads(holdout_pairs_path.read_text(encoding="utf-8"))
    clip_to_combo: dict[str, tuple[str, str]] = {}
    for p in all_pair_sources:
        audio = p["audio"]
        combo_str = p.get("combo", "")
        if audio not in clip_to_combo and "||" in combo_str:
            sp, vt = combo_str.split("||", 1)
            clip_to_combo[audio] = (sp.strip(), vt.strip().lower())

    seen, val_clips = set(), []
    for p in val_pairs:
        if p["audio"] not in seen:
            seen.add(p["audio"]); val_clips.append(p["audio"])

    val_combos = list({clip_to_combo[c] for c in val_clips if c in clip_to_combo})
    print(f"Val clips: {len(val_clips)}  |  (species, type) combos: {len(val_combos)}")

    strategies = build_queries(val_combos, tax_db, all_labels, holdout_descs)
    for s, q in strategies.items():
        print(f"  {s:<16}: {len(q)} queries")

    audio_root  = Path(args.audio_root)
    all_results = {}        # {model_key: {"agg": ..., "detail": ..., "sim_data": ...}}
    semantic_results = {}   # {model_key: {"semantic_probe_*": ..., ...}}

    # Load hand-written acoustic probe queries if provided
    probe_queries: list = []
    if args.semantic_queries:
        sq_path = Path(args.semantic_queries)
        if sq_path.exists():
            probe_queries = json.loads(sq_path.read_text(encoding="utf-8"))
            print(f"Loaded {len(probe_queries)} semantic probe queries from {sq_path}")
        else:
            print(f"[warn] --semantic-queries path not found: {sq_path}")

    run_sem_evals = probe_queries or args.acoustic_coherence or args.cross_species_eval

    def evaluate(checkpoint, label, model_key):
        model, processor = load_model(checkpoint, args.model, device)

        # Encode gallery once — reused by all evals for this model
        audio_matrix, vc = encode_gallery(
            model, processor, val_clips, audio_root, device, args.batch_size)

        agg, detail, sim_data = run_eval(
            model, processor, val_clips, clip_to_combo,
            strategies, audio_root, device, args.batch_size,
            audio_matrix=audio_matrix, valid_clips_in=vc)
        print_table(label, agg)

        # ── Semantic search evaluations ───────────────────────────────────────
        sem: dict = {}
        if probe_queries:
            print(f"\n  [semantic_probe] evaluating {len(probe_queries)} acoustic queries ...")
            sp_results, sp_summary = run_semantic_probe(
                audio_matrix, vc, clip_to_combo,
                model, processor, probe_queries, device)
            sem["semantic_probe_results"] = sp_results
            sem["semantic_probe_summary"] = sp_summary

        if args.acoustic_coherence:
            print(f"\n  [acoustic_coherence] ...")
            coh_results, coh_summary = run_acoustic_coherence(
                model, processor, val_combos, tax_db, all_labels, device)
            sem["acoustic_coherence_results"] = coh_results
            sem["acoustic_coherence_summary"] = coh_summary

        if args.cross_species_eval:
            print(f"\n  [cross_species_transfer] ...")
            xfer_results, xfer_summary = run_cross_species_transfer(
                audio_matrix, vc, clip_to_combo,
                model, processor, tax_db, all_labels, device)
            sem["cross_species_transfer_results"] = xfer_results
            sem["cross_species_transfer_summary"] = xfer_summary

        if sem:
            semantic_results[model_key] = sem

        del model
        return {"agg": agg, "detail": detail, "sim_data": sim_data}

    if args.checkpoint:
        print(f"\n{'='*72}\n  Evaluating: fine-tuned  ({args.checkpoint})")
        all_results["finetuned"] = evaluate(
            args.checkpoint, f"Fine-tuned: {args.checkpoint}", "finetuned")

    if args.also_base or not args.checkpoint:
        print(f"\n{'='*72}\n  Evaluating: base model  ({args.model})")
        all_results["base"] = evaluate(
            None, f"Base model (zero-shot): {args.model}", "base")

    # ── Zero-shot holdout species eval ────────────────────────────────────────
    # Evaluates on species that were NEVER in training — tests whether the model
    # learned acoustic features that generalise, vs memorising a species lookup.
    holdout_pairs_path = Path(args.holdout_pairs)
    if holdout_pairs_path.exists():
        print(f"\n{'='*72}")
        print(f"  Zero-shot eval: unseen species ({holdout_pairs_path})")
        holdout_pairs_data = json.loads(holdout_pairs_path.read_text(encoding="utf-8"))
        seen_h, holdout_clips_list = set(), []
        for p in holdout_pairs_data:
            if p["audio"] not in seen_h:
                seen_h.add(p["audio"]); holdout_clips_list.append(p["audio"])
        holdout_combos = list({
            clip_to_combo[c] for c in holdout_clips_list if c in clip_to_combo
        })
        print(f"  Holdout clips: {len(holdout_clips_list)}  |  combos: {len(holdout_combos)}")

        # Only taxonomy queries for holdout (no rich descriptions for unseen species)
        holdout_strats = build_queries(holdout_combos, tax_db, all_labels)

        def evaluate_holdout(checkpoint, label):
            model, processor = load_model(checkpoint, args.model, device)
            agg, detail, _ = run_eval(
                model, processor, holdout_clips_list, clip_to_combo,
                holdout_strats, audio_root, device, args.batch_size)
            del model
            print_table(f"{label} [ZERO-SHOT unseen species]", agg)
            return {"agg": agg, "detail": detail}

        if args.checkpoint:
            all_results["finetuned_zeroshot"] = evaluate_holdout(
                args.checkpoint, f"Fine-tuned: {args.checkpoint}")
        if args.also_base or not args.checkpoint:
            all_results["base_zeroshot"] = evaluate_holdout(
                None, f"Base model (zero-shot): {args.model}")
    else:
        print(f"\n[info] No holdout pairs at {holdout_pairs_path} — zero-shot species eval skipped")

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
    # Attach semantic eval summaries (exclude large per-item result lists)
    semantic_save: dict = {}
    for mk, sem in semantic_results.items():
        semantic_save[mk] = {
            key: val for key, val in sem.items() if key.endswith("_summary")
        }
        # Include probe per-query results (compact) since there are few queries
        if "semantic_probe_results" in sem:
            semantic_save[mk]["semantic_probe_results"] = [
                {k2: v2 for k2, v2 in r.items() if k2 != "recall_at_k"}
                | {"R@1": r["recall_at_k"].get(1), "R@5": r["recall_at_k"].get(5),
                   "R@10": r["recall_at_k"].get(10)}
                for r in sem["semantic_probe_results"]
            ]

    out_path.write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "base_model": args.model,
                "n_val_clips": len(val_clips),
                "n_val_combos": len(val_combos),
                "results": save_data,
                "semantic_eval": semantic_save,
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
        figs = Path(args.figures_dir)
        print(f"\nGenerating plots -> {figs}/")
        generate_plots(all_results, tax_db, figs)
        if semantic_results:
            print(f"\nGenerating semantic eval plots ...")
            generate_semantic_plots(semantic_results, figs)
        print(f"All plots saved.")


if __name__ == "__main__":
    main()
