#!/usr/bin/env python3
"""
CLAP Fine-tuning Script
=======================
Fine-tunes laion/clap-htsat-fused (or any compatible HF ClapModel) on local
Xeno-canto audio using the pre-built pair JSONs produced by
build_clap_training_pairs.py.

Training objective
------------------
Symmetric InfoNCE / CLIP-style contrastive loss.  For each mini-batch of N
(audio, text) pairs the model produces two embedding matrices (N × D each).
Both are L2-normalised; the similarity matrix (N × N) is scaled by a learned
log_scale parameter and trained with cross-entropy over rows and columns so
each audio attends to its own text and vice versa.

Inputs
------
  data/clap_train_pairs.json   list of {"audio": "audio/xc/<id>.mp3", "text": "..."}
  data/clap_val_pairs.json     same format, held-out validation split
  scripts/data/xc_audio/       audio root; pair["audio"] is resolved relative to it

Outputs
-------
  checkpoints/best.pt          state_dict with lowest val loss
  checkpoints/latest.pt        overwritten every epoch — safe resume point
  checkpoints/epochs/epoch_NN.pt   full copy after each epoch (same payload as latest;
                               use --no-epoch-checkpoints to skip — each file is ~1.8 GB)

Prerequisites
-------------
  pip install -r requirements-ml.txt
  ffmpeg on PATH  (needed by librosa for MP3 decoding on most systems)

Usage (from repo root, lets-solve-it/):
  python scripts/train_clap.py
  python scripts/train_clap.py --epochs 5 --batch-size 8 --accum 8
  python scripts/train_clap.py --resume checkpoints/latest.pt
  python scripts/train_clap.py --audio-root D:/xc_mirror --workers 8

Tips
----
- RTX 4070 (12 GB VRAM): batch-size 8, accum 8  → effective batch 64, FP16
- If OOM, lower --batch-size to 4 and raise --accum proportionally.
- Freeze the text tower early in training:
    python scripts/train_clap.py --freeze-text-epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Missing librosa / soundfile.  Run: pip install -r requirements-ml.txt", file=sys.stderr)
    raise

# ── constants ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL    = "laion/clap-htsat-fused"
TARGET_SR        = 48_000          # Hz required by laion/clap-htsat-fused
CLIP_DURATION_S  = 10.0            # seconds; long recordings are centre-cropped
MIN_DURATION_S   = 0.5             # recordings shorter than this are skipped


# ── helpers ────────────────────────────────────────────────────────────────────

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_workers() -> int:
    """
    Conservative worker default for Windows + 16 GB RAM systems.
    Keeps enough parallelism for throughput while avoiding host-memory spikes.
    """
    cpu_count = os.cpu_count() or 6
    return max(2, min(6, cpu_count // 2))


def load_audio(path: Path, sr: int = TARGET_SR, clip_s: float = CLIP_DURATION_S,
               augment: bool = False) -> np.ndarray | None:
    """
    Load MP3/WAV at `sr` Hz, mono.  Centre-crops or pads to `clip_s` seconds.
    Returns None when the file cannot be decoded or is too short.

    Fast path: if convert_to_wav.py has already produced a pre-clipped .wav
    sibling (same filename, .wav extension), that file is read directly with
    soundfile — no MP3 decoding, no resampling, no crop/pad needed since the
    WAV was written at exactly (sr, clip_s) by the converter.
    """
    target_len = int(clip_s * sr)
    min_len    = int(MIN_DURATION_S * sr)

    # --- fast path: pre-clipped WAV produced by convert_to_wav.py ---
    wav_path = path.with_suffix(".wav")
    if wav_path.is_file():
        try:
            y, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            # If the WAV has the right length it's pre-clipped — return immediately.
            if file_sr == sr and len(y) == target_len:
                return y
            # WAV exists but was not pre-clipped (e.g. older conversion); fall
            # through to the normal path using the WAV instead of MP3.
            if len(y) < min_len:
                return None
            if len(y) >= target_len:
                max_start = len(y) - target_len
                start = random.randint(0, max_start) if augment else max_start // 2
                y = y[start : start + target_len]
            else:
                y = np.pad(y, (0, target_len - len(y)))
            return y.astype(np.float32)
        except Exception:
            pass  # corrupt WAV — fall through to librosa with the original file

    # --- slow path: decode MP3 (or any format) via librosa ---
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception:
        return None

    if len(y) < min_len:
        return None

    if len(y) >= target_len:
        # Random crop during training, centre-crop otherwise
        max_start = len(y) - target_len
        start = random.randint(0, max_start) if augment else max_start // 2
        y = y[start : start + target_len]
    else:
        # Zero-pad on the right
        y = np.pad(y, (0, target_len - len(y)))

    return y.astype(np.float32)


# ── dataset ────────────────────────────────────────────────────────────────────

class ClapPairDataset(Dataset):
    """
    Reads a flat list of {"audio": relative_path, "text": "..."} pairs and
    resolves each audio path against `audio_root`.

    Bad rows (missing file, decode failure, too short) are silently skipped
    during __init__ so the DataLoader never sees a None sample.
    """

    def __init__(
        self,
        pairs_path: Path,
        audio_root: Path,
        sr: int = TARGET_SR,
        clip_s: float = CLIP_DURATION_S,
        verbose: bool = True,
        augment: bool = False,
    ) -> None:
        raw: list[dict[str, str]] = json.loads(
            pairs_path.read_text(encoding="utf-8")
        )

        self.sr      = sr
        self.clip_s  = clip_s
        self.root    = audio_root
        self.augment = augment

        # Pre-filter to rows whose audio file actually exists on disk.
        # The waveform is loaded lazily in __getitem__ to keep RAM low.
        self.pairs: list[dict[str, str]] = []
        missing = 0
        for p in raw:
            fp = audio_root / p["audio"]
            if fp.is_file():
                self.pairs.append(p)
            else:
                missing += 1

        if verbose:
            print(
                f"  [{pairs_path.name}]  {len(self.pairs):,} usable pairs "
                f"({missing:,} skipped - audio file not found)"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        pair  = self.pairs[idx]
        audio = load_audio(self.root / pair["audio"], self.sr, self.clip_s, self.augment)
        if audio is None:
            return None
        if self.augment:
            # Random gain: scale volume ±40% so model is invariant to recording level
            audio = audio * random.uniform(0.6, 1.4)
            # Gaussian noise: simulate low-SNR field recordings
            audio = audio + np.random.randn(len(audio)).astype(np.float32) * 0.002
        return {
            "audio":      audio,
            "text":       pair["text"],
            "combo":      pair.get("combo", ""),   # species||type — may be absent in old pair files
            "audio_path": pair["audio"],
        }


def spec_augment(
    features: "Tensor",
    time_mask_max: int = 100,
    freq_mask_max: int = 16,
    n_time_masks: int = 2,
    n_freq_masks: int = 2,
) -> "Tensor":
    """
    SpecAugment: randomly zero out time and frequency bands on a mel spectrogram.
    Operates on the last two dims (..., freq, time) so it works regardless of
    whether the leading batch/channel dims are present.
    Compatible with ClapPrecomputedDataset (applied after loading .clap.pt).
    """
    x = features.clone()
    freq_dim = x.shape[-2]
    time_dim = x.shape[-1]
    for _ in range(n_time_masks):
        t = random.randint(1, max(1, min(time_mask_max, time_dim - 1)))
        t0 = random.randint(0, time_dim - t)
        x[..., :, t0 : t0 + t] = 0.0
    for _ in range(n_freq_masks):
        f = random.randint(1, max(1, min(freq_mask_max, freq_dim - 1)))
        f0 = random.randint(0, freq_dim - f)
        x[..., f0 : f0 + f, :] = 0.0
    return x


class ClapPrecomputedDataset(Dataset):
    """
    Fast variant of ClapPairDataset that loads pre-computed mel-spectrogram
    tensors (.clap.pt sidecars) instead of raw waveforms.

    Run scripts/precompute_clap_features.py once to generate the sidecars.
    Each sidecar contains {"input_features": Tensor, "is_longer": Tensor}
    as saved by ClapFeatureExtractor with batch_size=1.

    Reduces per-batch CPU cost from ~14 s (ClapProcessor on 8 samples) to
    ~0.1 s (torch.load + RobertaTokenizer).
    """

    def __init__(
        self,
        pairs_path: Path,
        audio_root: Path,
        verbose: bool = True,
        augment: bool = False,
    ) -> None:
        raw: list[dict[str, str]] = json.loads(
            pairs_path.read_text(encoding="utf-8")
        )
        self.root    = audio_root
        self.augment = augment
        self.pairs: list[dict[str, str]] = []
        missing = 0
        for p in raw:
            clap_pt = (audio_root / p["audio"]).with_suffix(".clap.pt")
            if clap_pt.is_file():
                self.pairs.append(p)
            else:
                missing += 1
        if verbose:
            print(
                f"  [{pairs_path.name}]  {len(self.pairs):,} pre-computed pairs "
                f"({missing:,} skipped — .clap.pt not found)"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        pair    = self.pairs[idx]
        clap_pt = (self.root / pair["audio"]).with_suffix(".clap.pt")
        try:
            feat = torch.load(str(clap_pt), map_location="cpu", weights_only=True)
        except Exception:
            return None
        feats = feat["input_features"]
        if self.augment:
            feats = spec_augment(feats)
        return {
            # keep the batch dim (1, …) — collate_precomputed_fn cats along dim 0
            "input_features": feats,
            "is_longer":      feat["is_longer"],
            "text":       pair["text"],
            "combo":      pair.get("combo", ""),
            "audio_path": pair["audio"],
        }


def collate_precomputed_fn(
    batch: list[dict[str, Any] | None],
    tokenizer: Any,
) -> dict[str, Any] | None:
    """
    Collate for ClapPrecomputedDataset.

    Stacks pre-computed audio tensors and tokenises text on-the-fly (fast).
    Produces the same dict structure as collate_fn so the rest of the
    training loop is unchanged.
    """
    valid = [b for b in batch if b is not None]
    if not valid:
        return None

    input_features = torch.cat([b["input_features"] for b in valid], dim=0)
    is_longer      = torch.cat([b["is_longer"]      for b in valid], dim=0)
    texts          = [b["text"]       for b in valid]
    combos         = [b["combo"]      for b in valid]
    paths          = [b["audio_path"] for b in valid]

    text_enc = tokenizer(
        texts,
        return_tensors = "pt",
        padding        = True,
        truncation     = True,
    )

    return {
        "input_features": input_features,
        "is_longer":      is_longer,
        "input_ids":      text_enc["input_ids"],
        "attention_mask": text_enc["attention_mask"],
        "meta": {"combos": combos, "audio_paths": paths},
    }


def collate_fn(
    batch: list[dict[str, Any] | None],
    processor: ClapProcessor,
    sr: int,
) -> dict[str, Any] | None:
    """
    Filters None items, calls ClapProcessor, returns CPU tensors (pin_memory-safe).
    Non-tensor metadata (combos, audio_paths) is stored under the "meta" key so
    batch_to_device can skip it.
    Returns None when the entire batch is empty (shouldn't happen in practice).
    """
    valid = [b for b in batch if b is not None]
    if not valid:
        return None

    waveforms = [b["audio"]      for b in valid]
    texts     = [b["text"]       for b in valid]
    combos    = [b["combo"]      for b in valid]
    paths     = [b["audio_path"] for b in valid]

    inputs = processor(
        audio           = waveforms,
        text            = texts,
        sampling_rate   = sr,
        return_tensors  = "pt",
        padding         = True,
        truncation      = True,
    )
    result = {k: v for k, v in inputs.items()}
    result["meta"] = {"combos": combos, "audio_paths": paths}
    return result


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    # Move tensors to device; leave non-tensor values (e.g. "meta" dict) as-is.
    return {k: v.to(device, non_blocking=True) if isinstance(v, Tensor) else v
            for k, v in batch.items()}


# ── loss ───────────────────────────────────────────────────────────────────────

def contrastive_loss(
    audio_emb: Tensor,
    text_emb: Tensor,
    log_scale: Tensor,
    combos: list[str],
    audio_paths: list[str],
    combo_weights: dict[str, float] | None = None,
) -> Tensor:
    """
    Multi-positive symmetric InfoNCE with optional dataset-level combo weighting.

    Both embeddings are assumed already L2-normalised. Instead of treating only
    the diagonal as positive (which causes false negatives when two clips from the
    same species+type land in the same batch), we build a positive mask:

      positive[i][j] = 1  if audio[i] and text[j] share the same audio file
                           OR the same (species, type) combo

    Soft cross-entropy with these targets avoids penalising the model for
    correctly aligning clips that are genuinely interchangeable.

    combo_weights: pre-computed {combo: inverse_frequency} from the full training
    set (not from the batch). When provided, rare combos get higher per-sample
    weight so they contribute equally to the gradient regardless of class size.
    Weights are normalised within the batch to keep the loss scale stable.

    Falls back to standard diagonal loss when combo/path info is absent
    (e.g. old pair files without the "combo" field).
    """
    N = len(audio_emb)
    scale  = log_scale.exp().clamp(max=100.0)
    logits = scale * audio_emb @ text_emb.T   # (N, N)

    # Fast path: no metadata → standard diagonal CLIP loss
    if not combos or not any(combos):
        labels = torch.arange(N, device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0

    # Build positive mask on CPU then move to device (small N, negligible cost)
    mask = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if audio_paths[i] == audio_paths[j]:   # same recording, different text variant
                mask[i][j] = 1.0
            elif combos[i] and combos[i] == combos[j]:  # same species+type, different clip
                mask[i][j] = 1.0
    mask = mask.to(logits.device)

    # Normalise rows → soft targets that sum to 1
    targets = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)

    loss_per_row_a = -(targets   * F.log_softmax(logits,   dim=1)).sum(dim=1)
    loss_per_row_t = -(targets.T * F.log_softmax(logits.T, dim=1)).sum(dim=1)

    if combo_weights is not None:
        # Dataset-level inverse-frequency weights: rare combos weighted up.
        # Normalise within the batch so the loss scale is independent of weight magnitude.
        w = torch.tensor(
            [combo_weights.get(c, 1.0) for c in combos],
            dtype=torch.float32, device=logits.device,
        )
        w = w / w.sum().clamp(min=1e-8)
        loss_a = (w * loss_per_row_a).sum()
        loss_t = (w * loss_per_row_t).sum()
    else:
        loss_a = loss_per_row_a.mean()
        loss_t = loss_per_row_t.mean()

    return (loss_a + loss_t) / 2.0


# ── retrieval metric ───────────────────────────────────────────────────────────

@torch.no_grad()
def recall_at_1(
    model: ClapModel,
    loader: DataLoader,
    processor: ClapProcessor,
    device: torch.device,
    max_batches: int = 64,
) -> float:
    """
    Audio→text R@1 on a subset of the validation loader.

    A retrieval is counted correct if the top-ranked text comes from the same
    (species, type) combo as the query audio — not just the exact paired row.
    This avoids false negatives when two clips of the same species land in the
    same evaluation window.
    """
    model.eval()
    audio_embs: list[Tensor] = []
    text_embs:  list[Tensor] = []
    all_combos: list[str]    = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        if batch is None:
            continue
        combos = batch.get("meta", {}).get("combos", [])
        batch  = batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            a_feat = model.get_audio_features(
                input_features = batch.get("input_features"),
                is_longer      = batch.get("is_longer"),
            )
            t_feat = model.get_text_features(
                input_ids      = batch.get("input_ids"),
                attention_mask = batch.get("attention_mask"),
            )
        audio_embs.append(F.normalize(a_feat.pooler_output, dim=-1).cpu())
        text_embs.append(F.normalize(t_feat.pooler_output,  dim=-1).cpu())
        all_combos.extend(combos)

    if not audio_embs:
        return 0.0

    A    = torch.cat(audio_embs)    # (N, D)
    T    = torch.cat(text_embs)     # (N, D)
    sims = A @ T.T                  # (N, N)
    top1 = sims.argmax(dim=-1)      # (N,) — index of highest-scoring text per audio

    if all_combos and len(all_combos) == len(top1):
        # Correct if top-ranked text shares the same combo as the query audio
        correct = sum(
            all_combos[top1[i].item()] == all_combos[i]
            for i in range(len(top1))
        )
        return correct / len(top1)
    else:
        # Fallback for old pair files without combo info
        labels = torch.arange(len(top1))
        return (top1 == labels).float().mean().item()


# ── checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: ClapModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: Any,
    epoch: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optim_state":    optimizer.state_dict(),
            "scaler_state":   scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss":  best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: ClapModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: Any,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["best_val_loss"]


# ── training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: ClapModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: Any,
    device: torch.device,
    accum_steps: int,
    epoch: int,
    use_amp: bool,
    combo_weights: dict[str, float] | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"epoch {epoch:02d} train", unit="batch", dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        if batch is None:
            continue
        batch = batch_to_device(batch, device)

        meta   = batch.get("meta", {})
        combos = meta.get("combos", [])
        paths  = meta.get("audio_paths", [])

        with torch.autocast(device_type=device.type, enabled=use_amp):
            audio_feat = model.get_audio_features(
                input_features = batch.get("input_features"),
                is_longer      = batch.get("is_longer"),
            )
            text_feat = model.get_text_features(
                input_ids      = batch.get("input_ids"),
                attention_mask = batch.get("attention_mask"),
            )

            a_emb = F.normalize(audio_feat.pooler_output, dim=-1)
            t_emb = F.normalize(text_feat.pooler_output,  dim=-1)
            loss  = contrastive_loss(a_emb, t_emb, model.logit_scale_a, combos, paths, combo_weights)
            loss  = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        n_batches  += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: ClapModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    combo_weights: dict[str, float] | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc="val", unit="batch", dynamic_ncols=True):
        if batch is None:
            continue
        meta   = batch.get("meta", {})
        combos = meta.get("combos", [])
        paths  = meta.get("audio_paths", [])
        batch  = batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            audio_feat = model.get_audio_features(
                input_features = batch.get("input_features"),
                is_longer      = batch.get("is_longer"),
            )
            text_feat = model.get_text_features(
                input_ids      = batch.get("input_ids"),
                attention_mask = batch.get("attention_mask"),
            )
            a_emb = F.normalize(audio_feat.pooler_output, dim=-1)
            t_emb = F.normalize(text_feat.pooler_output,  dim=-1)
            loss  = contrastive_loss(a_emb, t_emb, model.logit_scale_a, combos, paths, combo_weights)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── cache warming ─────────────────────────────────────────────────────────────

def warm_file_cache(
    train_ds: "ClapPairDataset",
    val_ds: "ClapPairDataset",
    workers: int = 4,
) -> None:
    """
    Read every unique WAV (or MP3) file in the training and validation datasets
    into the OS page cache before the first epoch begins.

    On a 16 GB system the full ~8.9 GB of WAV siblings fits in RAM.  After this
    pass every subsequent DataLoader read is served from memory at ~20 GB/s
    instead of from disk at ~500 MB/s, making epoch 1 as fast as epoch 9.

    Uses a thread pool for parallel I/O (I/O-bound, so threads are fine).
    Skips files that cannot be opened rather than aborting training.
    """
    # Collect unique file paths.
    # ClapPrecomputedDataset → warm the .clap.pt sidecars.
    # ClapPairDataset        → warm the .wav siblings (or .mp3 fallback).
    seen: set[Path] = set()
    for ds in (train_ds, val_ds):
        for pair in ds.pairs:
            base   = ds.root / pair["audio"]
            clap_pt = base.with_suffix(".clap.pt")
            wav     = base.with_suffix(".wav")
            if clap_pt.is_file():
                seen.add(clap_pt)
            elif wav.is_file():
                seen.add(wav)
            else:
                seen.add(base)

    paths = sorted(seen)
    total_files = len(paths)
    total_mb    = sum(p.stat().st_size for p in paths if p.exists()) / 1024 ** 2

    print(
        f"\nWarming OS page cache: {total_files:,} unique files  "
        f"({total_mb / 1024:.2f} GB) ..."
    )

    n_ok = n_err = 0
    bytes_read = 0

    def _read(p: Path) -> int:
        return len(p.read_bytes())

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_read, p): p for p in paths}
        pbar = tqdm(
            as_completed(futs),
            total=total_files,
            desc="cache warm",
            unit="file",
            dynamic_ncols=True,
        )
        for fut in pbar:
            try:
                bytes_read += fut.result()
                n_ok += 1
            except Exception:
                n_err += 1
            pbar.set_postfix(ok=n_ok, err=n_err, mb=f"{bytes_read/1024**2:.0f}")

    elapsed = time.time() - t0
    speed   = bytes_read / 1024 ** 2 / max(elapsed, 1e-3)
    print(
        f"Cache warm done: {n_ok:,} files read  ({bytes_read/1024**3:.2f} GB)  "
        f"{speed:.0f} MB/s  ({elapsed:.1f}s)"
        + (f"  [{n_err} errors]" if n_err else "")
    )


# ── argument parsing ───────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = repo_root()
    ap = argparse.ArgumentParser(
        description="Fine-tune CLAP on Xeno-canto (audio, text) pairs."
    )
    ap.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace ClapModel id (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--audio-root",
        default=str(root / "scripts" / "data" / "xc_audio"),
        help="Root directory containing audio/xc/*.mp3 (default: scripts/data/xc_audio)",
    )
    ap.add_argument(
        "--train-pairs",
        default=str(root / "data" / "clap_train_pairs.json"),
        help="Path to clap_train_pairs.json",
    )
    ap.add_argument(
        "--val-pairs",
        default=str(root / "data" / "clap_val_pairs.json"),
        help="Path to clap_val_pairs.json",
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=str(root / "checkpoints"),
        help="Directory for checkpoints (default: checkpoints/)",
    )
    ap.add_argument("--epochs",       type=int,   default=10,    help="Training epochs (default 10)")
    ap.add_argument("--batch-size",   type=int,   default=8,     help="Per-GPU batch size (default 8)")
    ap.add_argument("--accum",        type=int,   default=8,     help="Gradient accumulation steps (default 8 -> effective batch 64)")
    ap.add_argument("--lr",           type=float, default=2e-5,  help="Peak learning rate for projection heads (default 2e-5)")
    ap.add_argument("--lr-audio-mult", type=float, default=0.1,
                    help="Audio encoder LR multiplier relative to --lr (default 0.1 → audio gets 2e-6)")
    ap.add_argument("--lr-text-mult",  type=float, default=0.5,
                    help="Text encoder LR multiplier relative to --lr (default 0.5 → text gets 1e-5)")
    ap.add_argument("--warmup-steps", type=int,   default=200,   help="LR warmup steps (default 200)")
    ap.add_argument("--workers",      type=int,   default=default_workers(), help="DataLoader worker processes (default auto)")
    ap.add_argument("--prefetch-factor", type=int, default=2,    help="DataLoader prefetch factor when workers > 0 (default 2)")
    ap.add_argument("--clip-s",       type=float, default=CLIP_DURATION_S, help="Audio clip length in seconds (default 10.0)")
    ap.add_argument(
        "--freeze-text-epochs", type=int, default=1,
        help="Freeze the text encoder for this many epochs at the start (default 1)",
    )
    ap.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed-precision training (use full FP32 — slower, more VRAM)",
    )
    ap.add_argument(
        "--resume", default=None, metavar="CHECKPOINT",
        help="Resume from a checkpoint file produced by this script",
    )
    ap.add_argument(
        "--no-epoch-checkpoints",
        action="store_true",
        help="Do not write checkpoints/<epoch-checkpoints-subdir>/epoch_NN.pt each epoch",
    )
    ap.add_argument(
        "--epoch-checkpoints-subdir",
        default="epochs",
        help="Subfolder under --checkpoint-dir for per-epoch copies (default: epochs)",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default 42)",
    )
    ap.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help="Disable DataLoader persistent workers (enabled by default when workers > 0)",
    )
    ap.add_argument(
        "--no-cache-warm",
        action="store_true",
        help="Skip the OS page-cache warming pass before epoch 0 (not recommended on cold starts)",
    )
    ap.add_argument(
        "--no-precomputed",
        action="store_true",
        help="Force raw-audio path even if .clap.pt sidecars exist (for debugging)",
    )
    ap.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training augmentation (random crop, noise/gain, SpecAugment). "
             "Val set is never augmented regardless.",
    )
    return ap.parse_args(argv)


# ── main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; force UTF-8 so tqdm/log output is not
    # mixed with UTF-16 when PowerShell redirects (see checkpoints/RESUME.md).
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args    = parse_args(argv)
    root    = repo_root()
    ckpt_dir = Path(args.checkpoint_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp   = (not args.no_amp) and (device.type == "cuda")
    if device.type == "cuda":
        # Throughput-focused defaults for Ampere/Ada cards such as RTX 4070.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print(f"Device:       {device}  (AMP {'on' if use_amp else 'off'})")
    print(f"Model:        {args.model}")
    print(f"Audio root:   {args.audio_root}")
    print(f"Batch size:   {args.batch_size}  x  accum {args.accum}  =  effective {args.batch_size * args.accum}")
    print(f"Epochs:       {args.epochs}  |  LR: {args.lr} "
          f"(audio ×{args.lr_audio_mult}={args.lr*args.lr_audio_mult:.2e}, "
          f"text ×{args.lr_text_mult}={args.lr*args.lr_text_mult:.2e})  |  warmup: {args.warmup_steps}")
    print(
        f"Workers:      {args.workers}  |  prefetch: {args.prefetch_factor}  |  "
        f"persistent: {'off' if args.no_persistent_workers else 'on'}"
    )
    print(f"Data path:    {'pre-computed .clap.pt (fast)' if not args.no_precomputed else 'raw audio (slow)'}")

    # ── processor + model ─────────────────────────────────────────────────────
    print("\nLoading processor and model weights ...")
    processor = ClapProcessor.from_pretrained(args.model)
    model: ClapModel = ClapModel.from_pretrained(args.model)
    model.to(device)

    # ── datasets ──────────────────────────────────────────────────────────────
    audio_root = Path(args.audio_root)
    print("\nBuilding datasets ...")

    # Auto-detect whether pre-computed .clap.pt sidecars are available.
    # If ≥ 95 % of training pairs have a sidecar, use the fast pre-computed
    # path (ClapPrecomputedDataset + collate_precomputed_fn) which reduces
    # per-batch CPU cost from ~14 s to ~0.1 s.  Fall back to the raw-audio
    # path (ClapPairDataset + collate_fn) otherwise.
    use_precomputed = False
    if not args.no_precomputed:
        pre_train = ClapPrecomputedDataset(Path(args.train_pairs), audio_root, verbose=False)
        raw_train = ClapPairDataset(Path(args.train_pairs), audio_root, verbose=False)
        coverage  = len(pre_train) / max(len(raw_train), 1)
        if coverage >= 0.95:
            use_precomputed = True
            print(
                f"  Pre-computed features detected ({coverage*100:.0f}% coverage) — "
                f"using fast ClapPrecomputedDataset."
            )
        else:
            print(
                f"  Pre-computed features coverage {coverage*100:.0f}% < 95% — "
                f"falling back to raw-audio path.  "
                f"Run scripts/precompute_clap_features.py to enable the fast path."
            )

    do_augment = not args.no_augment
    if use_precomputed:
        train_ds = ClapPrecomputedDataset(Path(args.train_pairs), audio_root, augment=do_augment)
        val_ds   = ClapPrecomputedDataset(Path(args.val_pairs),   audio_root, augment=False)
        _collate = partial(collate_precomputed_fn, tokenizer=processor.tokenizer)
    else:
        train_ds = ClapPairDataset(Path(args.train_pairs), audio_root, clip_s=args.clip_s, augment=do_augment)
        val_ds   = ClapPairDataset(Path(args.val_pairs),   audio_root, clip_s=args.clip_s, augment=False)
        _collate = partial(collate_fn, processor=processor, sr=TARGET_SR)

    print(f"  Augmentation: {'on (random crop, noise/gain, SpecAugment)' if do_augment else 'off'}")

    # Warm the OS page cache so epoch 0 is served from RAM, not disk.
    # For the pre-computed path this warms the .clap.pt files instead of WAVs.
    if not args.no_cache_warm:
        warm_file_cache(train_ds, val_ds, workers=max(args.workers, 4))

    # ── dataset-level combo frequencies for weighted loss + sampler ───────────
    # Count how many pairs each combo has in the training set.
    from collections import Counter
    combo_freq: Counter = Counter(
        p.get("combo", "") for p in train_ds.pairs if p.get("combo")
    )
    total_combos = sum(combo_freq.values())
    # Inverse-frequency weights, normalised so the mean weight == 1.
    n_combos = len(combo_freq)
    combo_weights: dict[str, float] = {
        combo: (total_combos / (n_combos * count))
        for combo, count in combo_freq.items()
    } if combo_freq else {}

    # WeightedRandomSampler: each training pair is sampled with probability
    # proportional to its combo's inverse frequency, so rare combos appear as
    # often as common ones in expectation each epoch.
    sample_weights = torch.tensor(
        [combo_weights.get(p.get("combo", ""), 1.0) for p in train_ds.pairs],
        dtype=torch.float64,
    )
    weighted_sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(train_ds),
        replacement = True,
    )
    print(
        f"  WeightedRandomSampler: {len(combo_freq)} combos, "
        f"weight range [{sample_weights.min():.3f}, {sample_weights.max():.3f}]"
    )

    loader_kwargs: dict[str, Any] = {}
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
        loader_kwargs["persistent_workers"] = (not args.no_persistent_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        sampler     = weighted_sampler,   # replaces shuffle=True
        num_workers = args.workers,
        collate_fn  = _collate,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.workers,
        collate_fn  = _collate,
        pin_memory  = (device.type == "cuda"),
        **loader_kwargs,
    )

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    # Differential learning rates: audio encoder learns slower (already pretrained
    # on AudioSet), text encoder learns at half speed, projection heads at full LR.
    audio_params = list(model.audio_model.parameters())
    text_params  = list(model.text_model.parameters())
    other_ids    = {id(p) for p in audio_params} | {id(p) for p in text_params}
    proj_params  = [p for p in model.parameters() if id(p) not in other_ids]

    param_groups = [
        {"params": audio_params, "lr": args.lr * args.lr_audio_mult},
        {"params": text_params,  "lr": args.lr * args.lr_text_mult},
        {"params": proj_params,  "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay = 1e-4,
        betas        = (0.9, 0.98),
    )

    total_steps = math.ceil(len(train_ds) / args.batch_size / args.accum) * args.epochs
    warmup      = min(args.warmup_steps, total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── optional resume ───────────────────────────────────────────────────────
    start_epoch    = 0
    best_val_loss  = float("inf")
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            print(f"\nResuming from {resume_path} ...")
            start_epoch, best_val_loss = load_checkpoint(
                resume_path, model, optimizer, scaler, scheduler
            )
            start_epoch += 1
            if not math.isfinite(best_val_loss):
                best_sidecar = ckpt_dir / "best.pt"
                if best_sidecar.is_file():
                    ck = torch.load(
                        best_sidecar, map_location="cpu", weights_only=False
                    )
                    bvl = ck.get("best_val_loss")
                    if bvl is not None and math.isfinite(float(bvl)):
                        best_val_loss = float(bvl)
                        print(
                            f"  -> best_val_loss from {best_sidecar.name}: "
                            f"{best_val_loss:.4f} (resume ckpt had non-finite best)"
                        )
            print(f"  -> starting at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        else:
            print(f"Warning: checkpoint {resume_path} not found, starting fresh.", file=sys.stderr)

    # ── training ──────────────────────────────────────────────────────────────
    print(f"\n{'-' * 60}")
    print(f"Starting training   ({len(train_ds):,} train  |  {len(val_ds):,} val pairs)")
    print(f"{'-' * 60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Optional text encoder freeze
        if args.freeze_text_epochs > 0:
            freeze = epoch < args.freeze_text_epochs
            for p in model.text_model.parameters():
                p.requires_grad_(not freeze)
            if freeze and epoch == 0:
                print(f"Text encoder frozen for first {args.freeze_text_epochs} epoch(s).")
            elif not freeze and epoch == args.freeze_text_epochs:
                print(f"Epoch {epoch}: text encoder unfrozen.")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, scheduler,
            device, args.accum, epoch, use_amp, combo_weights,
        )
        val_loss = validate(model, val_loader, device, use_amp, combo_weights)
        r1       = recall_at_1(model, val_loader, processor, device)

        elapsed = time.time() - t0
        print(
            f"epoch {epoch:02d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"R@1={r1:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # Update best before saving latest (latest.pt must carry correct best_val_loss for resume)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt",
                model, optimizer, scaler, scheduler, epoch, best_val_loss,
            )
            print(f"  [OK] new best val_loss={best_val_loss:.4f}  ->  {ckpt_dir}/best.pt")

        save_checkpoint(
            ckpt_dir / "latest.pt",
            model, optimizer, scaler, scheduler, epoch, best_val_loss,
        )

        if not args.no_epoch_checkpoints:
            epoch_dir = ckpt_dir / args.epoch_checkpoints_subdir
            epoch_path = epoch_dir / f"epoch_{epoch:02d}.pt"
            save_checkpoint(
                epoch_path,
                model, optimizer, scaler, scheduler, epoch, best_val_loss,
            )
            print(f"  saved epoch snapshot  ->  {epoch_path}")

    print(f"\nTraining complete.  Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
