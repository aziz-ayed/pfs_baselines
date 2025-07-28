import torch
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import h5py

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

    # Step 3: Perform patient-level stratified 70/15/15 split
    print("Performing 70/15/15 patient-level stratified split...")
    patients = np.array(list(patient_to_paths.keys()))
    y_patient = np.array([clin.set_index("patient_id").loc[p]["event"] for p in patients])
    
    # First split: 70% for training, 30% for temp (val + test)
    train_p, temp_p, y_train, y_temp = train_test_split(
        patients, y_patient, test_size=0.3, random_state=42, stratify=y_patient
    )
    
    # Second split: Split the 30% temp data into 50/50 for validation and test
    # This results in 15% of the original data for each.
    val_p, test_p = train_test_split(
        temp_p, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Step 4: Create final lists of file paths
    train_paths = [path.name for p_id in train_p for path in patient_to_paths[p_id]]
    val_paths = [path.name for p_id in val_p for path in patient_to_paths[p_id]]
    test_paths = [path.name for p_id in test_p for path in patient_to_paths[p_id]]
    
    # Step 5: Save the data as a TUPLE
    full_path_to_sample = Path(cfg["feature_dir"]) / train_paths[0]
    with h5py.File(full_path_to_sample, "r") as f:
        dim = f["features"].shape[1]

    # This tuple now includes the test set
    split_data_tuple = (train_paths, val_paths, test_paths, dim)
    
    torch.save(split_data_tuple, opts.output)
    print(f"✅ Successfully saved data splits to {opts.output}")
    print(f"Training slides: {len(train_paths)}, Validation slides: {len(val_paths)}, Test slides: {len(test_paths)}")

if __name__ == "__main__":
    main()