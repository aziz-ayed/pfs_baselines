
# Survival Baselines for TCGA Patch‑level Embeddings

Four aggregators, one repo:

| key | Aggregator | Survival head |
|-----|------------|---------------|
| **MeanPool** | Mean‑pool | Cox PH |
| **MaxPool** | Max‑pool | Cox PH |
| **AttnMIL** | Gated attention (Ilse 2018) | Cox PH |
| **TransMIL** | Full TransMIL (PPEG + top‑k corr + Transformer) | Cox PH |

Features
* Works with CHIEF (768‑D) or UNI/H‑Optimus (1 536‑D) tiles.
* Rectangular padding + **top‑k=256** correlation keeps VRAM O(N).
* AMP + GradScaler for mixed precision.
* Multi‑GPU via PyTorch **DistributedDataParallel** (8 GPU friendly).
* Logs Harrell C‑index to stdout; plug your W&B logger if desired.

Quick start
```bash
torchrun --standalone --nproc_per_node=7 \
         train.py --config configs/default.yaml
```
