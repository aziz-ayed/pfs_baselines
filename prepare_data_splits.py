import torch
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Prepare and save data splits for training.")
    parser.add_argument("--config", required=True, help="Path to the training configuration file.")
    parser.add_argument("--output", default="splits.pt", help="Path to save the output splits file.")
    opts = parser.parse_args()

    cfg = yaml.safe_load(open(opts.config))

    # Step 1: Load and clean clinical data
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
    
    # Step 2: Get a final, clean list of patient IDs and their slide paths
    valid_pids = set(clin["patient_id"])
    all_paths = sorted(Path(cfg["feature_dir"]).glob("*.h5"))
    
    patient_to_paths = {}
    for p in all_paths:
        pid = "-".join(p.name.split("-")[:3])
        if pid in valid_pids:
            patient_to_paths.setdefault(pid, []).append(p)

    # Step 3: Perform patient-level stratified split
    patients = np.array(list(patient_to_paths.keys()))
    y_patient = np.array([clin.set_index("patient_id").loc[p]["event"] for p in patients])
    
    train_p, val_p = train_test_split(
        patients, test_size=cfg.get("val_frac", 0.2), random_state=42, stratify=y_patient
    )
    
    # Step 4: Create final lists of file paths
    train_paths = [path for p_id in train_p for path in patient_to_paths[p_id]]
    val_paths = [path for p_id in val_p for path in patient_to_paths[p_id]]
    
    # Step 5: Save the data as a TUPLE
    with h5py.File(train_paths[0], "r") as f:
        dim = f["features"].shape[1]

    # This is now a tuple, not a dictionary
    split_data_tuple = (train_paths, val_paths, dim)
    
    torch.save(split_data_tuple, opts.output)
    print(f"✅ Successfully saved data splits to {opts.output}")
    print(f"Training slides: {len(train_paths)}, Validation slides: {len(val_paths)}")

if __name__ == "__main__":
    import h5py
    main()