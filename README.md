
# Survival Baselines for TCGA Patch‑level Embeddings

Four aggregators, one repo:

| key | Aggregator | Survival head |
|-----|------------|---------------|
| **mean** | Mean‑pool | Cox PH |
| **max** | Max‑pool | Cox PH |
| **attn** | Gated attention (Ilse 2018) | Cox PH |
| **transmil** | Full TransMIL (PPEG + top‑k corr + Transformer) | Cox PH |

Features
* Works with CHIEF (768‑D) or UNI/H‑Optimus (1 536‑D) tiles.
* Rectangular padding + **top‑k=256** correlation keeps VRAM O(N).
* AMP + GradScaler for mixed precision.
* Multi‑GPU via PyTorch **DistributedDataParallel** (8 GPU friendly).
* Logs Harrell C‑index to stdout; plug your W&B logger if desired.

Quick start
```bash
python -m torch.distributed.launch --nproc_per_node=8 train.py        --config configs/default.yaml
```
