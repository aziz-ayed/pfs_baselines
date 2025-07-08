import argparse
import os
import pathlib
from pathlib import Path
import sys
import warnings
from typing import Union, Optional, List
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import tqdm

from src.dataset import PatchBagDataset
from src.metrics import harrell, td_auc
import src.models

def calculate_auc_ci(event_times: np.ndarray, event_indicators: np.ndarray, risk_scores: np.ndarray):
    """
    Calculate the Harrell's concordance index and the time-dependent AUC.

    Parameters:
        event_times: Array of event times.
        event_indicators: Array of event indicators (1 if event occurred, 0 if censored).
        risk_scores: Array of predicted risks or scores.
    """

    # Handle case with no events for percentile calculation
    if np.any(event_indicators == 1):
        eval_times = np.percentile(event_times[event_indicators == 1], [25, 50, 75])
        auc = td_auc(event_times, event_indicators, event_times, event_indicators, risk_scores, eval_times)
        ci = harrell(event_times, event_indicators, risk_scores)
    else:
        auc = np.nan
        ci = np.nan
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

def eval_model(checkpoint_path: str, embed_paths: List[Path], clinical_csv: str,
               score_path: str= "evaluation_scores.csv", device: Optional[torch.device] = None):

    dataset = PatchBagDataset(paths=embed_paths, clinical_csv=clinical_csv)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    dim = checkpoint["dim"]
    model = eval(f"src.models.{model_name}({dim})")
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = torch.utils.data.DataLoader(dataset, shuffle=False)
    times, indicators, risk_scores = collect_scores_from_loader(model, loader, device=device)
    auc, ci = calculate_auc_ci(times, indicators, risk_scores)
    print(f"Model {checkpoint_path} evaluation results:\n- AUC: {auc:.3f}\n- CI: {ci:.3f}")

    # Make a table with different columns: embed_path, times, indicators, risk_scores
    output_table = pd.DataFrame({
        "embed_path": [str(p) for p in embed_paths],
        "time": times,
        "event": indicators,
        "risk_score": risk_scores
    })
    # Create directory if it does not exist
    score_dir = pathlib.Path(score_path).parent
    score_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving evaluation scores to {score_path}")
    output_table.to_csv(score_path, index=False)

def _get_parser():
    parser = argparse.ArgumentParser(description="Evaluate a survival model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    return parser

def main():
    parser = _get_parser()
    opts, _ = parser.parse_known_args()

    cfg = yaml.safe_load(open(opts.config))
    checkpoint_path = opts.checkpoint_path if hasattr(opts, "checkpoint_path") else cfg.get("checkpoint_path", None)
    assert checkpoint_path is not None, "Checkpoint path must be specified either in command line or config file."

    train_paths, val_paths, saved_dim = torch.load(cfg["split_file"], weights_only=False)
    feature_dir = cfg.get("feature_dir", None)
    if feature_dir is not None:
        # If feature_dir is specified, change the directory of the files.
        feature_dir: pathlib.Path = pathlib.Path(feature_dir)
        train_paths = [feature_dir / p.name for p in train_paths]
        val_paths = [feature_dir / p.name for p in val_paths]

    eval_cfg = cfg.get("eval", {})
    limit = eval_cfg.get("limit", None)
    split = eval_cfg.get("split", "")
    if limit is not None:
        train_paths = train_paths[:limit]
        val_paths = val_paths[:limit]
    paths_to_eval = train_paths if split == "train" else val_paths

    score_path = eval_cfg.get("score_path", "evaluation_scores.csv")
    eval_model(checkpoint_path, paths_to_eval, cfg["clinical_csv"], score_path)

if __name__ == "__main__":
    main()
