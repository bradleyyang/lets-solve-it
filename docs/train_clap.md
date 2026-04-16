# `train_clap.py` — CLAP Fine-tuning

Fine-tunes `laion/clap-htsat-fused` on your Xeno-canto corpus using the
`(audio, text)` pairs built by the data pipeline.

---

## Quick start

```bash
# from repo root (lets-solve-it/)
pip install -r requirements-ml.txt   # torch, transformers, librosa, soundfile …
# ffmpeg must be on PATH for librosa MP3 decoding

python scripts/train_clap.py
```

This uses all defaults: auto workers (typically 3-6), batch 8 x accum 8, 10 epochs, FP16 on GPU.

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| **Python 3.10+** | Same venv as the rest of the repo |
| **PyTorch ≥ 2.2** | CUDA build strongly recommended |
| **transformers ≥ 4.40** | `ClapModel` + `ClapProcessor` |
| **librosa ≥ 0.10** | MP3 decoding + resampling |
| **soundfile** | WAV fallback |
| **ffmpeg** | On system PATH; needed by librosa for MP3 on most platforms |
| **GPU** | RTX 4070 (12 GB) or better recommended; CPU training is functional but very slow |

All Python deps are in `requirements-ml.txt`:

```bash
pip install -r requirements-ml.txt
```

---

## Inputs

| Path | Description |
|------|-------------|
| `data/clap_train_pairs.json` | 96,735 `{"audio": "audio/xc/<id>.mp3", "text": "..."}` pairs |
| `data/clap_val_pairs.json` | 10,737 held-out pairs |
| `scripts/data/xc_audio/` | Audio root; `audio/xc/*.mp3` sit under here |

Pair JSONs are produced by `scripts/build_clap_training_pairs.py`.  
Audio files are downloaded by `scripts/download_xc_audio.py`.

---

## Outputs

```
checkpoints/
  best.pt      ← lowest validation loss seen so far
  latest.pt    ← overwritten every epoch (safe resume point)
```

Each checkpoint is a `torch.save` dict with:

```python
{
    "epoch":           int,
    "model_state":     OrderedDict,   # ClapModel.state_dict()
    "optim_state":     ...,
    "scaler_state":    ...,           # GradScaler (AMP)
    "scheduler_state": ...,           # LambdaLR cosine schedule
    "best_val_loss":   float,
}
```

---

## Training objective

**Symmetric InfoNCE (CLIP-style contrastive loss).**

For a batch of N pairs the model produces:

- `A` — N × D audio embeddings (L2-normalised)
- `T` — N × D text embeddings (L2-normalised)

Similarity matrix: `S = logit_scale × A @ T.T` (N × N).  
Loss: cross-entropy over rows (audio→text) + columns (text→audio), averaged.

`logit_scale` is the model's own learned temperature parameter.

---

## Architecture

Uses `laion/clap-htsat-fused` from Hugging Face:

- **Audio tower:** HTSAT (Hierarchical Token-Semantic Audio Transformer)
- **Text tower:** RoBERTa-based
- Both towers project to the **same D-dimensional embedding space**

The script fine-tunes **all parameters** by default (full fine-tune).  
Use `--freeze-text-epochs N` to keep the text tower frozen for the first N epochs
while the audio tower adapts faster.

---

## CLI reference

```
python scripts/train_clap.py [OPTIONS]

--model           HuggingFace model ID          default: laion/clap-htsat-fused
--audio-root      Root dir of downloaded audio  default: scripts/data/xc_audio
--train-pairs     Path to train JSON            default: data/clap_train_pairs.json
--val-pairs       Path to val JSON              default: data/clap_val_pairs.json
--checkpoint-dir  Where to write checkpoints    default: checkpoints/
--epochs          Number of epochs              default: 10
--batch-size      Per-GPU samples               default: 8
--accum           Gradient accumulation steps   default: 8  → eff. batch 64
--lr              Peak learning rate            default: 2e-5
--warmup-steps    LR warmup steps               default: 200
--workers         DataLoader processes          default: auto (2-6 by CPU)
--prefetch-factor DataLoader prefetch factor    default: 2
--clip-s          Audio clip length (seconds)   default: 10.0
--freeze-text-epochs  Freeze text tower epochs  default: 1
--no-amp          Disable mixed precision       flag, off by default
--no-persistent-workers Disable persistent workers flag, off by default
--resume          Resume from checkpoint path   optional
--seed            Random seed                   default: 42
```

---

## VRAM / batch size guide

| GPU VRAM | Recommended settings |
|----------|----------------------|
| 8 GB | `--batch-size 4 --accum 16` |
| 12 GB (RTX 4070) | `--batch-size 8 --accum 8` ← default |
| 16 GB | `--batch-size 16 --accum 4` |
| 24 GB+ | `--batch-size 32 --accum 2` |

All combinations give effective batch = 64.  Larger effective batches (128+)
often improve contrastive learning stability; tune if you have spare VRAM.

---

## LR schedule

Linear warm-up for `--warmup-steps` optimizer steps, then **cosine decay** to
near-zero over the remaining steps.  AdamW with `weight_decay=1e-4`,
`betas=(0.9, 0.98)`, gradient clipping at `max_norm=1.0`.

---

## Validation metric

After each epoch the script computes:

- **Val loss** — InfoNCE on the full validation loader
- **R@1** — audio→text Recall@1 over up to 64 batches (fast proxy for retrieval quality)

```
epoch 03  train_loss=1.2831  val_loss=1.1047  R@1=0.6250  (312s)
  ✓  new best val_loss=1.1047  →  checkpoints/best.pt
```

R@1 of 1.0 = every query retrieves its correct text at rank 1.

---

## Resuming a run

```bash
python scripts/train_clap.py --resume checkpoints/latest.pt
```

The optimizer, scheduler, and GradScaler states are all restored; training
continues from the next epoch.

---

## Common issues

| Problem | Fix |
|---------|-----|
| `libsndfile` / MP3 error | Install **ffmpeg** and put it on PATH |
| CUDA out of memory | Halve `--batch-size`, double `--accum` |
| `No module named librosa` | `pip install -r requirements-ml.txt` |
| Very slow on CPU | Use a CUDA GPU; CPU training can take days |
| All pairs skipped | Check `--audio-root` points to the folder containing `audio/xc/` |

---

## Using the fine-tuned checkpoint

Load `best.pt` for inference:

```python
import torch
from transformers import ClapModel, ClapProcessor

model_id  = "laion/clap-htsat-fused"
ckpt_path = "checkpoints/best.pt"

processor = ClapProcessor.from_pretrained(model_id)
model     = ClapModel.from_pretrained(model_id)
model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state"])
model.eval()
```

The model can then be used exactly like in `scripts/mini_clap_xc_sample.py`.
