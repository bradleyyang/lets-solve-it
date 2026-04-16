#!/usr/bin/env python3
"""
CLAP Fine-tuning Script — v4 (4th-run improvements)
====================================================
Builds on train_clap.py with four targeted improvements that squeeze more
performance out of the *same dataset*, without any architectural changes.

What's new vs. train_clap.py
-----------------------------
1. SpecAugment  (--no-spec-augment to disable)
   Random time-masking and frequency-masking applied to the pre-computed
   mel-spectrogram tensors *during training*, per sample, per batch.
   The .clap.pt files on disk are never modified.
   Default: 2 time masks of up to 80 frames (~0.8 s) and 2 frequency masks
   of up to 16 mel bins.  Controlled via --time-mask-max / --freq-mask-max /
   --num-time-masks / --num-freq-masks.

2. Warm-start  (--finetune-from)
   Loads only the model weights from a prior checkpoint and resets the
   optimiser + scheduler.  Unlike --resume (which continues the exact same
   training run), --finetune-from is for continuing from another run's best.pt
   at a lower learning rate without carrying over stale Adam momentum.

3. Differential learning rates  (--lr-audio-mult / --lr-text-mult)
   The audio encoder is already well-adapted from CLAP pre-training and two
   prior fine-tunes, so it gets a much lower LR to avoid regressing.
   Defaults:
     audio_model parameters : lr * 0.10
     text_model  parameters : lr * 0.50
     projection + logit_scale: lr * 1.00
   No weight-decay on bias / LayerNorm params (standard AdamW practice).

4. Larger effective batch  (default accum 16 → effective 128)
   More in-batch negatives per optimisation step.  With the pre-computed path
   this costs almost nothing in wall time.

Other defaults changed from train_clap.py
   --epochs         20  (was 10;  ~10 min/epoch × 20 ≈ 3.5 h)
   --lr             5e-6 (was 2e-5; appropriate for warm-start continuation)
   --warmup-steps   50  (was 200; shorter ramp when starting near convergence)
   --freeze-text-epochs 0 (was 1;  text encoder already adapted)
   Per-epoch checkpoints use Pareto-optimal snapshotting: an epoch is saved
   only if it is better than all kept snapshots on at least one metric
   (train_loss, val_loss, R@1).  Epochs that are strictly worse on every
   metric are automatically discarded and their files deleted.

Recommended launch for the 4th fine-tune (from repo root):

    $env:TRAINING_LOG="training_run_fourth_fine_tune.log"
    .venv/Scripts/python scripts/launch_training_v4.py \\
        --checkpoint-dir checkpoints/fourth-fine-tune \\
        --finetune-from  checkpoints/third-fine-tune/best.pt \\
        --workers 4

    (All other flags use the new defaults above.)
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Missing librosa / soundfile.  Run: pip install -r requirements-ml.txt",
          file=sys.stderr)
    raise

# ── constants ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "laion/clap-htsat-fused"
TARGET_SR       = 48_000
CLIP_DURATION_S = 10.0
MIN_DURATION_S  = 0.5


# ── helpers ────────────────────────────────────────────────────────────────────

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_workers() -> int:
    cpu_count = os.cpu_count() or 6
    return max(2, min(6, cpu_count // 2))


def load_audio(path: Path, sr: int = TARGET_SR,
               clip_s: float = CLIP_DURATION_S) -> np.ndarray | None:
    """Load MP3/WAV at `sr` Hz, mono.  Centre-crop or pad to `clip_s` seconds."""
    target_len = int(clip_s * sr)
    min_len    = int(MIN_DURATION_S * sr)

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
    except Exception:
        return None

    if len(y) < min_len:
        return None
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    return y.astype(np.float32)


# ── datasets ───────────────────────────────────────────────────────────────────

class ClapPairDataset(Dataset):
    def __init__(self, pairs_path: Path, audio_root: Path,
                 sr: int = TARGET_SR, clip_s: float = CLIP_DURATION_S,
                 verbose: bool = True) -> None:
        raw = json.loads(pairs_path.read_text(encoding="utf-8"))
        self.sr     = sr
        self.clip_s = clip_s
        self.root   = audio_root
        self.pairs: list[dict] = []
        missing = 0
        for p in raw:
            if (audio_root / p["audio"]).is_file():
                self.pairs.append(p)
            else:
                missing += 1
        if verbose:
            print(f"  [{pairs_path.name}]  {len(self.pairs):,} usable pairs "
                  f"({missing:,} skipped - audio file not found)")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict | None:
        pair  = self.pairs[idx]
        audio = load_audio(self.root / pair["audio"], self.sr, self.clip_s)
        if audio is None:
            return None
        return {
            "audio":      audio,
            "text":       pair["text"],
            "combo":      pair.get("combo", ""),
            "audio_path": pair["audio"],
        }


class ClapPrecomputedDataset(Dataset):
    """Loads pre-computed .clap.pt mel-spectrogram sidecars (fast path)."""

    def __init__(self, pairs_path: Path, audio_root: Path,
                 verbose: bool = True) -> None:
        raw = json.loads(pairs_path.read_text(encoding="utf-8"))
        self.root  = audio_root
        self.pairs: list[dict] = []
        missing = 0
        for p in raw:
            if (audio_root / p["audio"]).with_suffix(".clap.pt").is_file():
                self.pairs.append(p)
            else:
                missing += 1
        if verbose:
            print(f"  [{pairs_path.name}]  {len(self.pairs):,} pre-computed pairs "
                  f"({missing:,} skipped — .clap.pt not found)")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict | None:
        pair    = self.pairs[idx]
        clap_pt = (self.root / pair["audio"]).with_suffix(".clap.pt")
        try:
            feat = torch.load(str(clap_pt), map_location="cpu", weights_only=True)
        except Exception:
            return None
        return {
            "input_features": feat["input_features"],
            "is_longer":      feat["is_longer"],
            "text":       pair["text"],
            "combo":      pair.get("combo", ""),
            "audio_path": pair["audio"],
        }


# ── SpecAugment ────────────────────────────────────────────────────────────────

def spec_augment(
    features: Tensor,
    time_mask_max: int = 80,
    freq_mask_max: int = 16,
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
) -> Tensor:
    """
    Apply SpecAugment (Park et al., 2019) to a batch of mel spectrograms.

    features : (B, 1, F, T)  — pre-computed input_features tensor where
                                F ≈ 64 mel bins and T ≈ 1001 time frames.

    Each sample in the batch receives independent random masks, so the model
    cannot rely on the batch's shared structure to bypass the masking.

    time_mask_max  : max consecutive time frames zeroed per mask  (~0.8 s @ 10 ms/frame)
    freq_mask_max  : max consecutive mel bins zeroed per mask
    num_time_masks : how many time masks per sample
    num_freq_masks : how many frequency masks per sample

    Returns a new tensor (no in-place modification of the original).
    """
    if features.dim() != 4:
        return features  # guard against unexpected shapes

    result = features.clone()
    B, _, F, T = result.shape

    for b in range(B):
        for _ in range(num_time_masks):
            t = random.randint(0, time_mask_max)
            if t > 0:
                t0 = random.randint(0, max(T - t, 0))
                result[b, :, :, t0 : t0 + t] = 0.0

        for _ in range(num_freq_masks):
            f = random.randint(0, freq_mask_max)
            if f > 0:
                f0 = random.randint(0, max(F - f, 0))
                result[b, :, f0 : f0 + f, :] = 0.0

    return result


# ── collate functions ──────────────────────────────────────────────────────────

def collate_precomputed_fn(
    batch: list[dict | None],
    tokenizer: Any,
    augment: bool = False,
    time_mask_max: int = 80,
    freq_mask_max: int = 16,
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
) -> dict | None:
    """
    Collate for ClapPrecomputedDataset.

    When augment=True (training), SpecAugment is applied to input_features
    before returning the batch.  augment=False (validation) returns features
    unmodified — critical so val_loss reflects real performance.
    """
    valid = [b for b in batch if b is not None]
    if not valid:
        return None

    input_features = torch.cat([b["input_features"] for b in valid], dim=0)
    is_longer      = torch.cat([b["is_longer"]      for b in valid], dim=0)

    if augment:
        input_features = spec_augment(
            input_features,
            time_mask_max=time_mask_max,
            freq_mask_max=freq_mask_max,
            num_time_masks=num_time_masks,
            num_freq_masks=num_freq_masks,
        )

    texts  = [b["text"]       for b in valid]
    combos = [b["combo"]      for b in valid]
    paths  = [b["audio_path"] for b in valid]

    text_enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    return {
        "input_features": input_features,
        "is_longer":      is_longer,
        "input_ids":      text_enc["input_ids"],
        "attention_mask": text_enc["attention_mask"],
        "meta": {"combos": combos, "audio_paths": paths},
    }


def collate_fn(batch: list[dict | None], processor: ClapProcessor,
               sr: int) -> dict | None:
    """Raw-audio collate (fallback when pre-computed sidecars are missing)."""
    valid = [b for b in batch if b is not None]
    if not valid:
        return None
    waveforms = [b["audio"]      for b in valid]
    texts     = [b["text"]       for b in valid]
    combos    = [b["combo"]      for b in valid]
    paths     = [b["audio_path"] for b in valid]
    inputs = processor(
        audio=waveforms, text=texts, sampling_rate=sr,
        return_tensors="pt", padding=True, truncation=True,
    )
    result = dict(inputs)
    result["meta"] = {"combos": combos, "audio_paths": paths}
    return result


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, Tensor) else v
            for k, v in batch.items()}


# ── differential learning-rate param groups ────────────────────────────────────

def make_param_groups(
    model: ClapModel,
    base_lr: float,
    lr_audio_mult: float,
    lr_text_mult: float,
    weight_decay: float,
) -> list[dict]:
    """
    Split model parameters into six AdamW groups based on which sub-module
    they belong to and whether they should receive weight decay.

    Rationale:
      audio_model  : already well-adapted by CLAP pre-training + prior runs.
                     Low LR avoids catastrophic forgetting of audio features.
      text_model   : more room for improvement (domain-specific bird terminology).
                     Moderate LR.
      projections / logit_scale : highest LR — these benefit most from further
                     alignment tuning.

    Bias, LayerNorm, and layer_norm parameters never receive weight decay
    (standard AdamW practice; decaying these harms training stability).
    """
    no_decay_keywords = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    buckets: dict[str, tuple[list, list]] = {
        "audio": ([], []),   # (decay_params, nodecay_params)
        "text":  ([], []),
        "other": ([], []),
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = any(kw in name for kw in no_decay_keywords)
        if name.startswith("audio_model."):
            key = "audio"
        elif name.startswith("text_model."):
            key = "text"
        else:
            key = "other"
        bucket = buckets[key]
        (bucket[1] if no_decay else bucket[0]).append(param)

    lr_mults = {"audio": lr_audio_mult, "text": lr_text_mult, "other": 1.0}
    groups = []
    for key, (decay_p, nodecay_p) in buckets.items():
        lr = base_lr * lr_mults[key]
        if decay_p:
            groups.append({"params": decay_p,   "lr": lr, "weight_decay": weight_decay})
        if nodecay_p:
            groups.append({"params": nodecay_p, "lr": lr, "weight_decay": 0.0})

    total = sum(p.numel() for g in groups for p in g["params"])
    print(f"  Param groups: "
          f"audio×{lr_audio_mult} | text×{lr_text_mult} | proj×1.0 "
          f"| total {total/1e6:.1f}M params")
    return groups


# ── loss ───────────────────────────────────────────────────────────────────────

def contrastive_loss(
    audio_emb: Tensor,
    text_emb: Tensor,
    log_scale: Tensor,
    combos: list[str],
    audio_paths: list[str],
) -> Tensor:
    """
    Multi-positive symmetric InfoNCE.

    Positive mask: audio[i] and text[j] are positives when they share the
    same audio file OR the same (species, type) combo.  Soft targets prevent
    penalising the model for correctly aligning genuinely interchangeable clips.
    """
    N      = len(audio_emb)
    scale  = log_scale.exp().clamp(max=100.0)
    logits = scale * audio_emb @ text_emb.T

    if not combos or not any(combos):
        labels = torch.arange(N, device=logits.device)
        return (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)) / 2.0

    mask = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if audio_paths[i] == audio_paths[j]:
                mask[i][j] = 1.0
            elif combos[i] and combos[i] == combos[j]:
                mask[i][j] = 1.0
    mask    = mask.to(logits.device)
    targets = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)

    loss_a = -(targets   * F.log_softmax(logits,   dim=1)).sum(dim=1).mean()
    loss_t = -(targets.T * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
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
    model.eval()
    audio_embs, text_embs, all_combos = [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        if batch is None:
            continue
        combos = batch.get("meta", {}).get("combos", [])
        batch  = batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            a_feat = model.get_audio_features(
                input_features=batch.get("input_features"),
                is_longer=batch.get("is_longer"),
            )
            t_feat = model.get_text_features(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
            )
        audio_embs.append(F.normalize(a_feat.pooler_output, dim=-1).cpu())
        text_embs.append(F.normalize(t_feat.pooler_output,  dim=-1).cpu())
        all_combos.extend(combos)

    if not audio_embs:
        return 0.0

    A    = torch.cat(audio_embs)
    T    = torch.cat(text_embs)
    sims = A @ T.T
    top1 = sims.argmax(dim=-1)

    if all_combos and len(all_combos) == len(top1):
        correct = sum(
            all_combos[top1[i].item()] == all_combos[i]
            for i in range(len(top1))
        )
        return correct / len(top1)
    labels = torch.arange(len(top1))
    return (top1 == labels).float().mean().item()


# ── Pareto-optimal epoch checkpointing ────────────────────────────────────────

class ParetoCheckpointManager:
    """
    Maintains a Pareto-optimal set of per-epoch snapshots across three metrics:
      - train_loss  (lower is better)
      - val_loss    (lower is better)
      - R@1         (higher is better)

    An epoch snapshot is kept only if no already-kept snapshot is at least as
    good on every metric AND strictly better on at least one.

    When a new epoch is added:
      1. If it is dominated by any kept snapshot → discard it (no file written).
      2. Otherwise → save it, then delete any kept snapshots now dominated by
         the new one.

    Result: the snapshots directory contains exactly the Pareto front of all
    epochs seen so far, using no more disk space than necessary.
    """

    def __init__(self, checkpoint_dir: Path, subdir: str = "epochs") -> None:
        self.ep_dir = checkpoint_dir / subdir
        self.ep_dir.mkdir(parents=True, exist_ok=True)
        # Each entry: {epoch, train_loss, val_loss, r1, path}
        self._kept: list[dict] = []

    # ------------------------------------------------------------------
    def _dominates(self, a: dict, b: dict) -> bool:
        """True if a is at least as good as b on every metric and strictly
        better on at least one (i.e. a dominates b)."""
        at_least = (
            a["train_loss"] <= b["train_loss"] and
            a["val_loss"]   <= b["val_loss"]   and
            a["r1"]         >= b["r1"]
        )
        strictly = (
            a["train_loss"] < b["train_loss"] or
            a["val_loss"]   < b["val_loss"]   or
            a["r1"]         > b["r1"]
        )
        return at_least and strictly

    # ------------------------------------------------------------------
    def consider(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        r1: float,
        model: "ClapModel",
        optimizer: "torch.optim.Optimizer",
        scaler: "torch.cuda.amp.GradScaler",
        scheduler: Any,
        best_val_loss: float,
    ) -> bool:
        """
        Evaluate epoch against the current Pareto front and save/discard.
        Returns True if the snapshot was saved.
        """
        new = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, r1=r1)

        # Dominated? → skip entirely
        if any(self._dominates(k, new) for k in self._kept):
            print(f"  [epoch {epoch:02d}] snapshot skipped — dominated on all metrics")
            return False

        # Not dominated → save
        path = self.ep_dir / f"epoch_{epoch:02d}.pt"
        save_checkpoint(path, model, optimizer, scaler, scheduler, epoch,
                        best_val_loss)
        new["path"] = path

        # Remove any previously kept epochs now dominated by this new one
        dominated = [k for k in self._kept if self._dominates(new, k)]
        for d in dominated:
            try:
                d["path"].unlink(missing_ok=True)
                print(f"  [epoch {epoch:02d}] removed dominated snapshot "
                      f"epoch_{d['epoch']:02d}.pt  "
                      f"(tl={d['train_loss']:.4f} vl={d['val_loss']:.4f} "
                      f"r1={d['r1']:.4f})")
            except Exception:
                pass
        self._kept = [k for k in self._kept if k not in dominated]
        self._kept.append(new)

        kept_str = ", ".join(f"ep{k['epoch']:02d}" for k in sorted(self._kept, key=lambda x: x["epoch"]))
        print(f"  [epoch {epoch:02d}] snapshot saved  ->  {path.name}  "
              f"| Pareto front: [{kept_str}]")
        return True


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
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optim_state":     optimizer.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss":   best_val_loss,
    }, path)


def load_checkpoint(
    path: Path,
    model: ClapModel,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    scheduler: Any | None,
    weights_only_mode: bool = False,
) -> tuple[int, float]:
    """
    Load a checkpoint.

    weights_only_mode=True  (used with --finetune-from):
        Loads only the model weights.  Optimizer, scaler, and scheduler are
        left at their initial states.  Always starts from epoch 0 with fresh
        momentum — correct behaviour when continuing from a *different* run.

    weights_only_mode=False (used with --resume):
        Restores everything, continuing the exact same training run.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Model weights — tolerate key mismatches from different-version checkpoints
    sd = ckpt.get("model_state",
         ckpt.get("model_state_dict",
         ckpt.get("state_dict", ckpt)))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys in checkpoint (fine if "
              f"architecture is identical)")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys in checkpoint")

    if weights_only_mode:
        return 0, float("inf")   # fresh start: epoch 0, no known best_val_loss

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scaler is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt.get("best_val_loss", float("inf"))


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
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"epoch {epoch:02d} train", unit="batch",
                dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        if batch is None:
            continue
        batch  = batch_to_device(batch, device)
        meta   = batch.get("meta", {})
        combos = meta.get("combos", [])
        paths  = meta.get("audio_paths", [])

        with torch.autocast(device_type=device.type, enabled=use_amp):
            audio_feat = model.get_audio_features(
                input_features=batch.get("input_features"),
                is_longer=batch.get("is_longer"),
            )
            text_feat = model.get_text_features(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
            )
            a_emb = F.normalize(audio_feat.pooler_output, dim=-1)
            t_emb = F.normalize(text_feat.pooler_output,  dim=-1)
            loss  = contrastive_loss(a_emb, t_emb, model.logit_scale_a,
                                     combos, paths)
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
                input_features=batch.get("input_features"),
                is_longer=batch.get("is_longer"),
            )
            text_feat = model.get_text_features(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
            )
            a_emb = F.normalize(audio_feat.pooler_output, dim=-1)
            t_emb = F.normalize(text_feat.pooler_output,  dim=-1)
            loss  = contrastive_loss(a_emb, t_emb, model.logit_scale_a,
                                     combos, paths)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── cache warming ──────────────────────────────────────────────────────────────

def warm_file_cache(
    train_ds: ClapPairDataset | ClapPrecomputedDataset,
    val_ds:   ClapPairDataset | ClapPrecomputedDataset,
    workers: int = 4,
) -> None:
    seen: set[Path] = set()
    for ds in (train_ds, val_ds):
        for pair in ds.pairs:
            base    = ds.root / pair["audio"]
            clap_pt = base.with_suffix(".clap.pt")
            wav     = base.with_suffix(".wav")
            if clap_pt.is_file():
                seen.add(clap_pt)
            elif wav.is_file():
                seen.add(wav)
            else:
                seen.add(base)

    paths      = sorted(seen)
    total_mb   = sum(p.stat().st_size for p in paths if p.exists()) / 1024**2
    print(f"\nWarming OS page cache: {len(paths):,} unique files "
          f"({total_mb/1024:.2f} GB) ...")

    n_ok = n_err = 0
    bytes_read = 0

    def _read(p: Path) -> int:
        return len(p.read_bytes())

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_read, p): p for p in paths}
        pbar = tqdm(as_completed(futs), total=len(paths), desc="cache warm",
                    unit="file", dynamic_ncols=True)
        for fut in pbar:
            try:
                bytes_read += fut.result()
                n_ok += 1
            except Exception:
                n_err += 1
            pbar.set_postfix(ok=n_ok, err=n_err, mb=f"{bytes_read/1024**2:.0f}")

    elapsed = time.time() - t0
    speed   = bytes_read / 1024**2 / max(elapsed, 1e-3)
    print(f"Cache warm done: {n_ok:,} files  ({bytes_read/1024**3:.2f} GB)  "
          f"{speed:.0f} MB/s  ({elapsed:.1f}s)"
          + (f"  [{n_err} errors]" if n_err else ""))


# ── argument parsing ───────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = repo_root()
    ap   = argparse.ArgumentParser(
        description="Fine-tune CLAP v4: SpecAugment + warm-start + differential LR."
    )
    # paths
    ap.add_argument("--model",       default=DEFAULT_MODEL)
    ap.add_argument("--audio-root",  default=str(root / "scripts" / "data" / "xc_audio"))
    ap.add_argument("--train-pairs", default=str(root / "data" / "clap_train_pairs.json"))
    ap.add_argument("--val-pairs",   default=str(root / "data" / "clap_val_pairs.json"))
    ap.add_argument("--checkpoint-dir", default=str(root / "checkpoints"))
    # checkpointing
    ap.add_argument("--resume",        default=None, metavar="CKPT",
                    help="Full resume: restore model + optimiser + epoch counter")
    ap.add_argument("--finetune-from", default=None, metavar="CKPT",
                    help="Warm-start: load model weights only, reset optimiser. "
                         "Use for continuing from a *different* run's best.pt.")
    ap.add_argument("--no-epoch-checkpoints", action="store_true",
                    help="Disable Pareto epoch snapshotting entirely "
                         "(only best.pt and latest.pt will be written)")
    ap.add_argument("--epoch-checkpoints-subdir", default="epochs")
    # training hyper-params  (new defaults suitable for warm-start continuation)
    ap.add_argument("--epochs",       type=int,   default=20)
    ap.add_argument("--batch-size",   type=int,   default=8)
    ap.add_argument("--accum",        type=int,   default=16,
                    help="Gradient accumulation steps (default 16 → effective 128)")
    ap.add_argument("--lr",           type=float, default=5e-6,
                    help="Base learning rate (default 5e-6 for warm-start)")
    ap.add_argument("--warmup-steps", type=int,   default=50,
                    help="LR warmup steps (default 50; short ramp near convergence)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--freeze-text-epochs", type=int, default=0,
                    help="Freeze text encoder for this many epochs (default 0; "
                         "text encoder is already adapted from prior runs)")
    # differential LR
    ap.add_argument("--lr-audio-mult", type=float, default=0.1,
                    help="Scale LR for audio_model params (default 0.1 — conservative)")
    ap.add_argument("--lr-text-mult",  type=float, default=0.5,
                    help="Scale LR for text_model params (default 0.5 — moderate)")
    # SpecAugment
    ap.add_argument("--no-spec-augment", action="store_true",
                    help="Disable SpecAugment (enabled by default)")
    ap.add_argument("--time-mask-max",  type=int, default=80,
                    help="Max consecutive time frames per mask (default 80 ≈ 0.8 s)")
    ap.add_argument("--freq-mask-max",  type=int, default=16,
                    help="Max consecutive mel bins per mask (default 16)")
    ap.add_argument("--num-time-masks", type=int, default=2)
    ap.add_argument("--num-freq-masks", type=int, default=2)
    # dataloader
    ap.add_argument("--workers",         type=int, default=default_workers())
    ap.add_argument("--prefetch-factor",  type=int, default=2)
    ap.add_argument("--no-persistent-workers", action="store_true")
    ap.add_argument("--no-cache-warm",   action="store_true")
    ap.add_argument("--no-precomputed",  action="store_true")
    ap.add_argument("--clip-s",          type=float, default=CLIP_DURATION_S)
    # misc
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed",   type=int, default=42)
    return ap.parse_args(argv)


# ── main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args     = parse_args(argv)
    root     = repo_root()
    ckpt_dir = Path(args.checkpoint_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True
        torch.set_float32_matmul_precision("high")

    use_augment = not args.no_spec_augment

    print(f"Device:         {device}  (AMP {'on' if use_amp else 'off'})")
    print(f"Model:          {args.model}")
    print(f"Audio root:     {args.audio_root}")
    print(f"Batch size:     {args.batch_size}  x  accum {args.accum}  "
          f"=  effective {args.batch_size * args.accum}")
    print(f"Epochs:         {args.epochs}  |  LR: {args.lr}  "
          f"|  warmup: {args.warmup_steps}")
    print(f"LR multipliers: audio×{args.lr_audio_mult}  "
          f"text×{args.lr_text_mult}  proj×1.0")
    print(f"SpecAugment:    {'ON' if use_augment else 'OFF'}"
          + (f"  (time≤{args.time_mask_max} ×{args.num_time_masks}  "
             f"freq≤{args.freq_mask_max} ×{args.num_freq_masks})" if use_augment else ""))
    print(f"Workers:        {args.workers}  |  prefetch: {args.prefetch_factor}  "
          f"|  persistent: {'off' if args.no_persistent_workers else 'on'}")

    # ── load processor + model ────────────────────────────────────────────────
    print("\nLoading processor and model weights ...")
    processor = ClapProcessor.from_pretrained(args.model)
    model: ClapModel = ClapModel.from_pretrained(args.model)
    model.to(device)

    # ── datasets ──────────────────────────────────────────────────────────────
    audio_root = Path(args.audio_root)
    print("\nBuilding datasets ...")

    use_precomputed = False
    if not args.no_precomputed:
        pre_check = ClapPrecomputedDataset(Path(args.train_pairs), audio_root, verbose=False)
        raw_check = ClapPairDataset(Path(args.train_pairs), audio_root, verbose=False)
        coverage  = len(pre_check) / max(len(raw_check), 1)
        if coverage >= 0.95:
            use_precomputed = True
            print(f"  Pre-computed features detected ({coverage*100:.0f}% coverage) — "
                  f"using fast ClapPrecomputedDataset.")
        else:
            print(f"  Pre-computed coverage {coverage*100:.0f}% < 95% — "
                  f"falling back to raw-audio path.")

    if use_precomputed:
        train_ds = ClapPrecomputedDataset(Path(args.train_pairs), audio_root)
        val_ds   = ClapPrecomputedDataset(Path(args.val_pairs),   audio_root)
        _collate_train = partial(
            collate_precomputed_fn,
            tokenizer=processor.tokenizer,
            augment=use_augment,
            time_mask_max=args.time_mask_max,
            freq_mask_max=args.freq_mask_max,
            num_time_masks=args.num_time_masks,
            num_freq_masks=args.num_freq_masks,
        )
        _collate_val = partial(
            collate_precomputed_fn,
            tokenizer=processor.tokenizer,
            augment=False,   # always clean for val
        )
    else:
        train_ds       = ClapPairDataset(Path(args.train_pairs), audio_root, clip_s=args.clip_s)
        val_ds         = ClapPairDataset(Path(args.val_pairs),   audio_root, clip_s=args.clip_s)
        _collate_train = partial(collate_fn, processor=processor, sr=TARGET_SR)
        _collate_val   = _collate_train   # no augment available on raw path

    if not args.no_cache_warm:
        warm_file_cache(train_ds, val_ds, workers=max(args.workers, 4))

    loader_kwargs: dict[str, Any] = {}
    if args.workers > 0:
        loader_kwargs["prefetch_factor"]    = max(1, args.prefetch_factor)
        loader_kwargs["persistent_workers"] = not args.no_persistent_workers

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=_collate_train,
        pin_memory=(device.type == "cuda"), drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=_collate_val,
        pin_memory=(device.type == "cuda"),
        **loader_kwargs,
    )

    # ── optimiser with differential LR ────────────────────────────────────────
    print("\nSetting up optimiser ...")
    param_groups = make_param_groups(
        model,
        base_lr       = args.lr,
        lr_audio_mult = args.lr_audio_mult,
        lr_text_mult  = args.lr_text_mult,
        weight_decay  = args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98))

    total_steps = math.ceil(len(train_ds) / args.batch_size / args.accum) * args.epochs
    warmup      = min(args.warmup_steps, total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── checkpoint loading ────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume and args.finetune_from:
        print("WARNING: both --resume and --finetune-from supplied.  "
              "--resume takes precedence.", file=sys.stderr)

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            print(f"\nFull resume from {resume_path} ...")
            start_epoch, best_val_loss = load_checkpoint(
                resume_path, model, optimizer, scaler, scheduler,
                weights_only_mode=False,
            )
            start_epoch += 1
            print(f"  -> epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        else:
            print(f"Warning: --resume checkpoint not found: {resume_path}",
                  file=sys.stderr)
    elif args.finetune_from:
        ft_path = Path(args.finetune_from)
        if ft_path.is_file():
            print(f"\nWarm-start: loading model weights from {ft_path} ...")
            load_checkpoint(ft_path, model, None, None, None,
                            weights_only_mode=True)
            print(f"  -> model weights loaded; optimiser reset to fresh state.")
            print(f"  -> training will start at epoch 0 with lr={args.lr:.1e}")
        else:
            print(f"Warning: --finetune-from checkpoint not found: {ft_path}",
                  file=sys.stderr)

    # ── training ──────────────────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print(f"Starting training  ({len(train_ds):,} train  |  {len(val_ds):,} val pairs)")
    print(f"Epoch snapshots:   "
          + ("disabled (--no-epoch-checkpoints)" if args.no_epoch_checkpoints
             else "Pareto-optimal (kept only if best on ≥1 metric)"))
    print(f"{'-'*60}\n")

    pareto = (None if args.no_epoch_checkpoints
              else ParetoCheckpointManager(ckpt_dir, args.epoch_checkpoints_subdir))

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Optional text encoder freeze (off by default in v4)
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
            device, args.accum, epoch, use_amp,
        )
        val_loss = validate(model, val_loader, device, use_amp)
        r1       = recall_at_1(model, val_loader, processor, device)

        # Report current LR for the projection group (index -2 = other_decay)
        cur_lr = optimizer.param_groups[-2]["lr"] if len(optimizer.param_groups) >= 2 else args.lr

        elapsed = time.time() - t0
        print(
            f"epoch {epoch:02d}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"R@1={r1:.4f}  lr={cur_lr:.2e}  ({elapsed:.0f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scaler,
                            scheduler, epoch, best_val_loss)
            print(f"  [OK] new best val_loss={best_val_loss:.4f}  "
                  f"->  {ckpt_dir}/best.pt")

        save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scaler,
                        scheduler, epoch, best_val_loss)

        if pareto is not None:
            pareto.consider(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                r1=r1,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
            )

    print(f"\nTraining complete.  Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
