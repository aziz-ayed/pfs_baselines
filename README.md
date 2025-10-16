---

# Survival Baselines for TCGA Patch-level Embeddings (with Multimodal RNA Fusion)

A modular research framework for **weakly supervised survival prediction** using **multiple instance learning (MIL)** on whole-slide embeddings and **RNA-guided multimodal fusion**.
Built for scalable PyTorch training, attention-based interpretability, and reproducible benchmarking across TCGA cohorts.

---

## What’s inside

* **Aggregators (unimodal pathology)**

  * `MeanPoolCox`, `MaxPoolCox`, `AttnMILCox`, `TransMILCox`, `AttnMILNewCox` — from simple pooling to gated attention and transformer-based MIL. 
* **Multimodal (pathology + RNA)**

  * `MultimodalCox`: combines AttnMILNew slide embeddings with a 640-D RNA latent via late fusion into a Cox head. 
* **Training**: DDP-ready, AMP-enabled (autocast + GradScaler), patient-level stratified splits, per-organ metrics, W&B logging, and full checkpointing.  
* **Evaluation**: Harrell’s C-index + time-dependent AUC (t-AUC). 
* **Utilities**:

  * `prepare_data_splits.py`: patient-level 70/15/15 splits from features + clinical CSV. 
  * `dump_attentions.py`: exports gated-attention weights aligned to patch coordinates. 

---

## Repo layout

```
.
├── train.py
├── eval.py
├── dump_attentions.py
├── prepare_data_splits.py
├── configs/
│   └── default.yaml
└── src/
    ├── dataset.py
    ├── models.py
    ├── metrics.py
    ├── collate.py
    └── seed.py
```

---

## Data & assumptions

* **Patch features**: one HDF5 per slide with dataset `features` shaped `(N_patches, D)`.
* **Clinical CSV**: must include `patient_id`, `project_id`, `progression_recurrence_event`, `days_to_progression_recurrence`, `max_follow_up_days`.
  Event time is computed as `days_to_progression_recurrence` if the event occurred, otherwise `max_follow_up_days`.  
* **Organs / cohorts**: current mapping groups TCGA projects into `{Lung, Colon, Breast}`; single-organ filtering supported. 
* **RNA latents (multimodal)**: per-patient 640-D latents (`latent_0 … latent_639`), intersected across **features ∩ clinical ∩ RNA**.  

> 💡 If not using RNA, set `model: AttnMILNewCox` (or any unimodal variant). For multimodal setups, use `model: MultimodalCox`. 

---

## Configuration

Edit `configs/default.yaml`:

* **Paths**

  ```yaml
  feature_dir: "/path/to/your/features"
  split_file:  "data/clean_splits.pt"
  clinical_csv: "data/outcomes/clinical.csv"
  ckpt_dir: "checkpoints/multimodal_latest"
  ```



* **Model**

  ```yaml
  model: MultimodalCox
  aggregator: "AttnMILNew"
  feature_dim: null
  topk_corr: 256
  dropout_p: 0.25
  ```

   

* **Training**

  ```yaml
  epochs: 500
  batch_size: 8
  learning_rate: 5e-5
  weight_decay: 1e-4
  gpus: [0,1,2,3,4,5,6]
  num_workers: 4
  save_checkpoints: true
  ```

   

* **W&B**

  ```yaml
  wandb:
    project: tcga_survival_baselines
    run_name: multimodal_latest
    mode: online
    group: sean_splits
    id: multimodal_latest
  ```



---

## 1️⃣ Prepare patient-level splits

Generates patient-stratified 70/15/15 splits and saves a tuple `(train_paths, val_paths, test_paths, feature_dim)` to `--output`.  

```bash
python prepare_data_splits.py \
  --config configs/default.yaml \
  --output data/clean_splits.pt
```

---

## 2️⃣ Train

**DDP (recommended):**

```bash
torchrun --standalone --nproc_per_node=7 \
  train.py --config configs/default.yaml \
  --run_name my_experiment --seed 42
```

* CLI flags: `--config`, `--seed`, `--run_name`. 
* Auto-detects feature dim if `feature_dim: null`. 
* Saves config + feature dim in checkpoints. 
* Logs C-index and t-AUC (overall + per-organ).  

---

## 3️⃣ Evaluate

```bash
python eval.py \
  --checkpoint checkpoints/multimodal_latest/model_best.pth \
  --split test \
  --output_csv outputs/preds_test.csv
```

* Loads config from checkpoint and reports C-index + mean t-AUC.  
* Prevents train/test patient overlap. 
* Optionally writes per-slide predictions with metadata. 

---

## 4️⃣ (Optional) Dump patch attentions

For AttnMIL-style models, export patch-level attention maps:

```bash
python dump_attentions.py \
  --checkpoint checkpoints/multimodal_latest/model_best.pth \
  --split val \
  --coords-dir /path/to/coords_h5 \
  --out-dir outputs/attentions \
  --output-csv outputs/attentions/val_all.csv
```

 

---

## 🔁 Reproducibility

```python
from src.seed import set_seed
set_seed(42)
```



---

## 🧩 Notes & tips

* **Multimodal RNA**: `MultimodalCox` expects a 640-D latent per patient; update the CSV path in `src/dataset.py`. 
* **Feature dim**: auto-detected unless specified.  
* **Organs**: use `single_organ_study` / `train_on_organs` to subset (e.g., `"Lung"`). Defaults to `"all"`. 
* **W&B**: disable logging via `wandb.mode: disabled`. 

---

## ⚡ Quick start (end-to-end)

```bash
# 1) Prepare splits
python prepare_data_splits.py --config configs/default.yaml --output data/clean_splits.pt

# 2) Train
torchrun --standalone --nproc_per_node=7 \
  train.py --config configs/default.yaml --run_name demo --seed 123

# 3) Evaluate
python eval.py --checkpoint checkpoints/multimodal_latest/model_best.pth \
  --split test --output_csv outputs/preds_test.csv
```

---

## Citation

If you use this code, please cite the relevant MIL and survival analysis works, and acknowledge this repository.

---
