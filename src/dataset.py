from pathlib import Path
from typing import Dict, List, Tuple
import h5py
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor

class PatchBagDataset(Dataset):
    """
    A simple dataset initialized with a specific list of file paths.
    It still needs the clinical_csv to look up labels for those paths.
    """
    def __init__(self, paths: List[Path], clinical_csv: str):
        self.paths = paths

        self.clin = None
        # Load clinical data into a dictionary for fast lookups
        if clinical_csv:
            clin = pd.read_csv(clinical_csv)
            if "patient_id" in clin.columns:
                clin["patient_id"] = clin["patient_id"].astype(str).str.strip()

            def _choose_time(row):
                if row["progression_recurrence_event"] == 1:
                    return row["days_to_progression_recurrence"]
                return row["max_follow_up_days"]
            clin["time"] = clin.apply(_choose_time, axis=1)
            clin["event"] = clin["progression_recurrence_event"]

            self.clin = clin.set_index("patient_id")[["time", "event"]].to_dict("index")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        p = self.paths[idx]
        pid = "-".join(p.name.split("-")[:3])
        
        with h5py.File(p, "r") as f:
            feats = torch.tensor(f["features"][:])

        rec = {"time": -1, "event": -1}
        if self.clin and pid in self.clin:
            rec = self.clin[pid]

        return (
            feats,
            torch.tensor(rec["time"], dtype=torch.float32),
            torch.tensor(rec["event"], dtype=torch.float32),
        )