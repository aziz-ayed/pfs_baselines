import torch
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import h5py

def main():
    parser = argparse.ArgumentParser(description="Prepare and save data splits for training using a predefined file.")
    parser.add_argument("--config", required=True, help="Path to the training configuration file.")
    parser.add_argument("--output", default="splits.pt", help="Path to save the output splits file.")
    # New argument for the splits file
    parser.add_argument("--splits_csv", default="data/sean/new_splits.csv", help="Path to the CSV file defining splits.")
    opts = parser.parse_args()

    cfg = yaml.safe_load(open(opts.config))

    # Step 1: Load and clean clinical data (unchanged)
    print("Loading and cleaning clinical data...")
    clin = pd.read_csv(cfg["clinical_csv"])
    clin["patient_id"] = clin["patient_id"].astype(str).str.strip()

    def _choose_time(row):
        if row["progression_recurrence_event"] == 1:
            return row["days_to_progression_recurrence"]
        return row["max_follow_up_days"]
    clin["time"] = clin.apply(_choose_time, axis=1)
    clin["event"] = clin["progression_recurrence_event"]
    clin.dropna(subset=['time', 'event'], inplace=True)
    clin = clin[clin['time'] > 0]
    
    # Step 2: Get a final, clean list of patient IDs and their slide paths (unchanged)
    valid_pids = set(clin["patient_id"])
    all_paths = sorted(Path(cfg["feature_dir"]).glob("*.h5"))
    
    patient_to_paths = {}
    for p in all_paths:
        pid = "-".join(p.name.split("-")[:3])
        if pid in valid_pids:
            patient_to_paths.setdefault(pid, []).append(p)
    
    # --- MODIFIED SECTION ---
    # Step 3: Load patient splits from the predefined CSV file
    print(f"Loading predefined patient splits from {opts.splits_csv}...")
    splits_df = pd.read_csv(opts.splits_csv)
    splits_df["patient_id"] = splits_df["patient_id"].astype(str).str.strip()

    # Ensure we only use patients that have existing feature files
    available_pids = set(patient_to_paths.keys())
    splits_df = splits_df[splits_df['patient_id'].isin(available_pids)]

    # Create lists of patient IDs for each split
    train_p = splits_df[splits_df['split'] == 'train']['patient_id'].tolist()
    val_p = splits_df[splits_df['split'] == 'val']['patient_id'].tolist()
    test_p = splits_df[splits_df['split'] == 'test']['patient_id'].tolist()
    
    # --- END MODIFIED SECTION ---

    # Step 4: Create final lists of file paths (unchanged)
    train_paths = [path.name for p_id in train_p for path in patient_to_paths.get(p_id, [])]
    val_paths = [path.name for p_id in val_p for path in patient_to_paths.get(p_id, [])]
    test_paths = [path.name for p_id in test_p for path in patient_to_paths.get(p_id, [])]
    
    # Step 5: Save the data as a TUPLE (unchanged)
    if not train_paths:
        raise ValueError("No training files found. Check your splits CSV and feature directory.")
        
    full_path_to_sample = Path(cfg["feature_dir"]) / train_paths[0]
    with h5py.File(full_path_to_sample, "r") as f:
        dim = f["features"].shape[1]

    split_data_tuple = (train_paths, val_paths, test_paths, dim)
    
    torch.save(split_data_tuple, opts.output)
    
    # --- NEW: Print final counts ---
    print("\n" + "="*40)
    print("      Final Split Information")
    print("="*40)
    print(f"✅ Successfully saved data splits to {opts.output}")
    print("\n--- Patient Counts ---")
    print(f"Training patients:   {len(train_p)}")
    print(f"Validation patients: {len(val_p)}")
    print(f"Test patients:       {len(test_p)}")
    print("\n--- Slide Counts (from paths) ---")
    print(f"Training slides:     {len(train_paths)}")
    print(f"Validation slides:   {len(val_paths)}")
    print(f"Test slides:         {len(test_paths)}")
    print("="*40)


if __name__ == "__main__":
    main()