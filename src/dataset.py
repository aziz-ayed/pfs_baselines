from pathlib import Path
from typing import Dict, List, Tuple, Optional
import h5py
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor

# Define maps as module-level constants for efficiency and clarity
ORGAN_MAP = {
    'TCGA-LUAD': 'Lung', 'TCGA-LUSC': 'Lung',
    'TCGA-READ': 'Colon', 'TCGA-COAD': 'Colon',
    'TCGA-BRCA': 'Breast'
}
ORGAN_TO_IDX = {name: i for i, name in enumerate(['Lung', 'Colon', 'Breast'])}


class PatchBagDataset(Dataset):
    """
    A dataset that can be filtered to specific organs upon initialization.
    """

    def __init__(self, paths: List[Path], clinical_csv: str, organs_to_use: Optional[List[str]] = None):
        
        # 1. Load clinical and RNA data
        clin_df = pd.read_csv(clinical_csv)
        clin_df["patient_id"] = clin_df["patient_id"].astype(str).str.strip()
        clin_df['organ'] = clin_df['project_id'].map(ORGAN_MAP)
    
        rna_latent_path = 'data/sean/bulkformer_latents_20250711_024656.csv'
        rna_df = pd.read_csv(rna_latent_path)
        rna_df = rna_df.rename(columns={'cases.submitter_id': 'patient_id'})
        # Ensure RNA data is unique per patient
        rna_df = rna_df.drop_duplicates(subset=['patient_id'], keep='first')
    
        self.latent_cols = [f'latent_{i}' for i in range(640)]
        self.rna_df = rna_df.set_index('patient_id')[self.latent_cols]
        
        self.organ_to_idx = ORGAN_TO_IDX
    
        # Find the intersection of patients across all data sources
        pids_from_paths = {"-".join(p.name.split("-")[:3]) for p in paths}
        pids_from_clin = set(clin_df['patient_id'])
        pids_from_rna = set(self.rna_df.index)
    
        # Get the set of patients that exist everywhere
        common_pids = pids_from_paths.intersection(pids_from_clin).intersection(pids_from_rna)
    
        # 3. Filter everything to only use the common patients
        self.paths = [p for p in paths if "-".join(p.name.split("-")[:3]) in common_pids]
        clin_df = clin_df[clin_df['patient_id'].isin(common_pids)]
        
        # If a filter is specified, apply it now
        if organs_to_use and organs_to_use != "all":
            clin_df = clin_df[clin_df['organ'].isin(organs_to_use)]
    
        # 4. Process time/event and create the final lookup dictionary from the filtered data
        def _choose_time(row):
            if row["progression_recurrence_event"] == 1:
                return row["days_to_progression_recurrence"]
            return row["max_follow_up_days"]
        clin_df["time"] = clin_df.apply(_choose_time, axis=1)
        clin_df["event"] = clin_df["progression_recurrence_event"]
        
        self.clin = clin_df.set_index("patient_id")[["time", "event", "organ"]].to_dict("index")

    def __len__(self) -> int:
        return len(self.paths)

    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, int, Tensor]: # Note the new Tensor in the return type
        p = self.paths[idx]
        pid = "-".join(p.name.split("-")[:3])
    
        with h5py.File(p, "r") as f:
            feats = torch.tensor(f["features"][:])
    
        # Get RNA latent vector for the patient
        rna_latent_vec = torch.tensor(self.rna_df.loc[pid].values, dtype=torch.float32)
    
        rec = self.clin[pid]
        organ_name = rec["organ"]
        organ_idx = ORGAN_TO_IDX.get(organ_name, -1)
    
        return (
            feats,
            torch.tensor(rec["time"], dtype=torch.float32),
            torch.tensor(rec["event"], dtype=torch.float32),
            torch.tensor(organ_idx, dtype=torch.long),
            rna_latent_vec # Add the new tensor here
        )