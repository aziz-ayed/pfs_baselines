import argparse
import os
import pathlib
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PatchBagDataset
from src.collate import pad_collate
from src import models
from src.metrics import harrell, td_auc
import yaml

def collect_predictions(model, loader, device):
    """Runs inference and collects predictions and labels."""
    model.eval()
    
    all_risks, all_times, all_events = [], [], []
    with torch.no_grad():
        # Unpack 5 items from the loader, including rna_feats
        for feats, t, e, _, rna_feats in tqdm(loader, desc="Evaluating"):
            feats = feats.to(device)
            rna_feats = rna_feats.to(device) # Move RNA data to device
            
            # Pass both tensors to the model
            risks = model(feats, rna_feats).cpu()
            
            all_risks.append(risks)
            all_times.append(t)
            all_events.append(e)
            
    return (
        np.concatenate(all_risks),
        np.concatenate(all_times),
        np.concatenate(all_events),
    )

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained survival model.")
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path, help="Path to the model checkpoint (.pth file).")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Which data split to evaluate on.")
    parser.add_argument("--output_csv", type=pathlib.Path, help="Optional path to save prediction scores as a CSV.")
    opts = parser.parse_args()

    print(f"Loading checkpoint: {opts.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(opts.checkpoint, map_location=device, weights_only=False)

    # <<< CHANGE: Use the config saved in the checkpoint for robustness
    if 'config' not in ckpt:
        raise ValueError("Checkpoint must contain a 'config' dictionary.")
    cfg = ckpt['config']
    print("✅ Loaded configuration from checkpoint.")
    
    dim = ckpt.get("dim")
    if dim is None:
        raise ValueError("Checkpoint must contain the feature dimension 'dim'.")

    print(f"Loading data from: {cfg['split_file']}")
    all_splits = torch.load(cfg["split_file"], weights_only=False)
    train_paths, val_paths, test_paths = all_splits[0], all_splits[1], all_splits[2]

    feature_dir = pathlib.Path(cfg["feature_dir"])
    train_paths = [feature_dir / pathlib.Path(p).name for p in train_paths]
    val_paths = [feature_dir / pathlib.Path(p).name for p in val_paths]
    test_paths = [feature_dir / pathlib.Path(p).name for p in test_paths]

    path_map = {"train": train_paths, "val": val_paths, "test": test_paths}
    eval_paths = path_map[opts.split]
    
    print(f"Evaluating on '{opts.split}' split with {len(eval_paths)} samples.")

    # <<< NEW: Added check for data leakage between train and eval sets
    if opts.split != "train":
        # Extract patient IDs from file paths
        train_pids = {"-".join(p.name.split("-")[:3]) for p in train_paths}
        eval_pids = {"-".join(p.name.split("-")[:3]) for p in eval_paths}
        
        # Find the intersection (overlap) between the two sets
        overlap = train_pids.intersection(eval_pids)
        
        if len(overlap) > 0:
            warnings.warn(
                f"\n🚨 Potential data leakage detected! "
                f"{len(overlap)} patient(s) are present in both the training and '{opts.split}' sets.\n"
                f"Overlapping PIDs: {sorted(list(overlap))}"
            )
        else:
            print(f"✅ No overlap found between train and '{opts.split}' patient IDs.")
    
    model_name = cfg.get("model", "AttnMILNewCox") # Use 'model' key from config
    dropout_p = cfg.get("dropout_p", 0.25)
    
    if model_name == "MultimodalCox":
        print(f"✅ Initializing MultimodalCox model.")
        model = models.MultimodalCox(d_path=dim, dropout=dropout_p).to(device)
    else:
        # Fallback to original unimodal logic
        print(f"✅ Initializing Unimodal model: {model_name}")
        Net = getattr(models, model_name)
        try:
            model = Net(dim, dropout=dropout_p).to(device)
        except TypeError:
            model = Net(dim).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])

    dataset = PatchBagDataset(paths=eval_paths, clinical_csv=cfg["clinical_csv"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=pad_collate
    )

    risks, times, events = collect_predictions(model, loader, device)
    
    print("Loading training data for td-AUC reference...")
    train_dataset = PatchBagDataset(paths=train_paths, clinical_csv=cfg["clinical_csv"])
    
    # This logic assumes the dataset after filtering has the same patients.
    # It might be more robust to load all data and then filter.
    train_pids_list = ["-".join(p.name.split("-")[:3]) for p in train_dataset.paths]
    train_T = np.array([train_dataset.clin[pid]["time"] for pid in train_pids_list])
    train_E = np.array([train_dataset.clin[pid]["event"] for pid in train_pids_list])

    ci = harrell(times, events, risks)
    if np.any(events == 1):
        eval_times = np.percentile(times[events == 1], [25, 50, 75])
        auc_scores, auc_mean = td_auc(train_T, train_E, times, events, risks, eval_times)
    else:
        auc_mean = np.nan
        warnings.warn("No events in the evaluation set; AUC cannot be computed.")

    print("\n--- Evaluation Results ---")
    print(f"Checkpoint: {opts.checkpoint.name}")
    print(f"Split:      {opts.split}")
    print(f"C-Index:    {ci:.4f}")
    print(f"Mean t-AUC: {auc_mean:.4f}")
    print("--------------------------")

    if opts.output_csv:
        # <<< FIX: Use the filtered paths from the dataset object itself
        # This ensures all columns have the same length as the prediction arrays.
        final_paths = dataset.paths 
        pids = ["-".join(p.name.split("-")[:3]) for p in final_paths]

        clin_df = pd.read_csv(cfg["clinical_csv"], dtype={"patient_id": str}).set_index('patient_id')
        
        # Use reindex to handle cases where a PID might not be in clin_df
        datasets = clin_df.reindex(pids)['project_id'].tolist()

        opts.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "path": [p.name for p in final_paths],
            "patient_id": pids,
            "dataset": datasets,
            "time": times,
            "event": events,
            "risk_score": risks,
        })
        df.to_csv(opts.output_csv, index=False)
        print(f"Saved prediction scores to: {opts.output_csv}")


if __name__ == "__main__":
    main()