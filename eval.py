import argparse
import os
import pathlib
from pathlib import Path
import sys
import warnings
from typing import Union, Optional, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import tqdm

from src.dataset import PatchBagDataset
from src.metrics import harrell, td_auc

def calculate_auc_ci(event_times: np.ndarray, event_indicators: np.ndarray, risk_scores: np.ndarray):
    """
    Calculate the Harrell's concordance index and the time-dependent AUC.

    Parameters:
        event_times: Array of event times.
        event_indicators: Array of event indicators (1 if event occurred, 0 if censored).
        risk_scores: Array of predicted risks or scores.
    """
    ci = harrell(event_times, event_indicators, risk_scores)

    # Handle case with no events for percentile calculation
    if np.any(event_indicators == 1):
        eval_times = np.percentile(event_times[event_indicators == 1], [25, 50, 75])
        auc = td_auc(event_times, event_indicators, event_times, event_indicators, risk_scores, eval_times)
    else:
        auc = np.nan
        warnings.warn("No events in validation set; cannot compute AUC.")

    return auc, ci

def collect_scores_from_loader(net: torch.nn.Module, loader: torch.utils.data.DataLoader, device=None):
    net.eval()
    if device is not None:
        net = net.to(device)

    risk_scores, times, indicators = [], [], []
    with torch.no_grad():
        for feats, t, e in tqdm.tqdm(loader, desc="Calculating risk scores", ncols=100):
            if device is not None:
                feats = feats.to(device)
            r = net(feats).cpu()

            times.append(t)
            indicators.append(e)
            risk_scores.append(r)

    times = np.concatenate(times)
    indicators = np.concatenate(indicators)
    risk_scores = np.concatenate(risk_scores)

    return times, indicators, risk_scores

def eval_model(model_path: str, embed_paths: List[Path], clinical_csv: str,
               output_path: str="evaluation_results.csv", device: Optional[torch.device] = None):

    dataset = PatchBagDataset(paths=embed_paths, clinical_csv=clinical_csv)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model = model.to(device)

    loader = torch.utils.data.DataLoader(dataset, shuffle=False)
    times, indicators, risk_scores = collect_scores_from_loader(model, loader, device=device)
    auc, ci = calculate_auc_ci(times, indicators, risk_scores)
    print(f"Model {model_path} evaluation results:\n- AUC: {auc:.3f}\n- CI: {ci:.3f}")

    # Make a table with different columns: embed_path, times, indicators, risk_scores
    output_table = pd.DataFrame({
        "embed_path": [str(p) for p in embed_paths],
        "time": times,
        "event": indicators,
        "risk_score": risk_scores
    })
    output_table.to_csv(output_path, index=False)


def debug():
    feature_dir = "/data/rbg/users/azizayed/chemistry/trident/outputs_tcga/20x_256px_0px_overlap/features_uni_v2"
    split_file = "data/splits.pt"
    clinical_csv = "data/clinical_data.csv"
    train_paths, val_paths, saved_dim = torch.load(split_file, weights_only=False)
    if feature_dir is not None:
        # If feature_dir is specified, change the directory of the files.
        feature_dir: pathlib.Path = pathlib.Path(feature_dir)
        train_paths = [feature_dir / p.name for p in train_paths]
        val_paths = [feature_dir / p.name for p in val_paths]

    embed_paths = train_paths
    embed_paths = [Path(p) for p in embed_paths]
    dataset = PatchBagDataset(paths=embed_paths, clinical_csv=clinical_csv)
    for el in dataset:
        print(el[0].shape, el[1], el[2])
        print(el)
        break


    # eval_model(None, train_paths, clinical_csv=clinical_csv)

def _get_parser():
    parser = argparse.ArgumentParser(description="Evaluate a survival model.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file.")
    parser.add_argument("split_file", type=str, help="Path torch dataset which saved paths to embedding files.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],)
    parser.add_argument("clinical_csv", type=str, help="Path to the clinical CSV file.")
    parser.add_argument("output_path", type=str, help="Path to save the evaluation results.")
    parser.add_argument("--feature_dir", type=str, help="Directory in which features are stored (if different from the split file).")
    parser.add_argument("--limit", default=None, type=int, help="Limit the number of files processed. For debugging.")

    return parser

def main():
    parser = _get_parser()
    args = parser.parse_args()

    train_paths, val_paths, saved_dim = torch.load(args.split_file, weights_only=False)
    if args.feature_dir is not None:
        # If feature_dir is specified, change the directory of the files.
        feature_dir: pathlib.Path = pathlib.Path(args.feature_dir)
        train_paths = [feature_dir / p.name for p in train_paths]
        val_paths = [feature_dir / p.name for p in val_paths]

    if args.limit:
        train_paths = train_paths[:args.limit]
        val_paths = val_paths[:args.limit]
    paths_to_eval = train_paths if args.split == "train" else val_paths

    eval_model(args.model_path, paths_to_eval, args.clinical_csv, args.output_path)

if __name__ == "__main__":
    main()
