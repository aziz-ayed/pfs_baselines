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

def _path_to_pid(path: Path) -> str:
    """
    Extract patient ID from the file path.
    Assumes the patient ID is the first three parts of the filename separated by dashes.
    """
    return "-".join(path.name.split("-")[:3])

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

def collect_scores(checkpoint_path: str, embed_paths: List[Path], clinical_csv: str,
                   score_path: str= "evaluation_scores.csv", device: Optional[torch.device] = None,
                   cfg: Optional[dict] = None):

    cfg = cfg or {}
    eval_cfg = cfg.get("eval", {})
    overwrite = eval_cfg.get("overwrite", False)
    if not os.path.exists(score_path) or overwrite:
        dataset = PatchBagDataset(paths=embed_paths, clinical_csv=clinical_csv)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_name = checkpoint.get("model_name", cfg["aggregator"] + "Cox")
        if model_name == "model":
            model_name = cfg["aggregator"] + "Cox"
        print(f"Model name: {model_name}")
        dim = checkpoint["dim"]
        model = eval(f"src.models.{model_name}({dim})")
        model = model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        loader = torch.utils.data.DataLoader(dataset, shuffle=False)
        times, indicators, risk_scores = collect_scores_from_loader(model, loader, device=device)

        # Make a table with different columns: embed_path, times, indicators, risk_scores
        scores_table = pd.DataFrame({
            "embed_path": [str(p) for p in embed_paths],
            "time": times,
            "event": indicators,
            "risk_score": risk_scores
        })

        clinical_table = pd.read_csv(cfg["clinical_csv"])
        scores_table["patient_id"] = scores_table["embed_path"].apply(lambda x: _path_to_pid(Path(x)))
        scores_table = scores_table.merge(clinical_table, on="patient_id", how="left")

        # Create directory if it does not exist
        score_dir = pathlib.Path(score_path).parent
        score_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving evaluation scores to {score_path}")
        scores_table.to_csv(score_path, index=False)

    else:
        print(f"Loading existing evaluation scores from {score_path}")
        scores_table = pd.read_csv(score_path)
        # times = scores_table["time"].values
        # indicators = scores_table["event"].values
        # risk_scores = scores_table["risk_score"].values

    return scores_table

def risks_to_probabilities(times, events, risks, eval_times=None):
    """
    Convert Cox model risks to event probabilities at each unique event time.
    Args:
        times: numpy array of observed times
        events: numpy array of event indicators (1=event, 0=censored)
        risks: numpy array of risk scores (log hazard ratios)
        eval_times: optional, times at which to evaluate probabilities
    Returns:
        eval_times: times at which probabilities are computed
        probs: array of shape (n_samples, n_times), event probabilities
    """
    order = np.argsort(times)
    inv_order = np.argsort(order)
    times, events, risks = times[order], events[order], risks[order]
    exp_risks = np.exp(risks)
    if eval_times is None:
        eval_times = np.unique(times[events == 1])
    baseline_hazard = []
    for t in eval_times:
        at_risk = exp_risks[times >= t]
        events_at_t = (times >= t) & (events == 1)
        baseline_hazard.append(events_at_t.sum() / at_risk.sum())
    baseline_cumhaz = np.cumsum(baseline_hazard)
    # Compute survival for each sample at each eval_time
    surv = np.exp(-np.outer(np.exp(risks), baseline_cumhaz))
    probs = 1 - surv  # Probability of event by time t
    probs = probs[inv_order, :]  # Restore original order
    return eval_times, probs

def calculate_brier_scores(score_table: pd.DataFrame):
    days_to_event_col = "time"
    follow_up_col = "max_follow_up_days"
    event_col = "event"
    score_col = "risk_score"
    required_cols = [days_to_event_col, follow_up_col, event_col, score_col]

    for rc in required_cols:
        if rc not in score_table.columns:
            raise ValueError(f"Required column '{rc}' not found in the score table.")

    time_horizons = [
        {"label": "3m", "days": 90},
        {"label": "6m", "days": 180},
        {"label": "12m", "days": 365},
        {"label": "24m", "days": 730}
    ]
    # eval_times = [th["days"] for th in time_horizons]
    eval_times, probs = risks_to_probabilities(
        score_table[days_to_event_col].values,
        score_table[event_col].values,
        score_table[score_col].values,
    )

    for th in time_horizons:
        th_label = th["label"]
        th_days = th["days"]

        # If we don't have the exact time, use the closest one
        diffs = np.abs(eval_times - th_days)
        idx = np.argmin(diffs)
        if diffs[idx] > 10: # If the closest time is too far, warn
            print(f"Warning: No evaluation time close to {th_days} days found. Using closest time {eval_times[idx]} days.")

        keep_rows = (score_table[days_to_event_col] <= th_days) | (score_table[follow_up_col] >= th_days)

        probs_at_th = probs[keep_rows, idx]
        indicator_value = score_table.loc[keep_rows, event_col].values

        # Calculate Brier score
        brier_score = (probs_at_th - indicator_value) ** 2
        mean_brier_score = brier_score.mean()
        print(f"Mean Brier score at {th_label} ({th_days} days): {mean_brier_score:.4f}")

        score_table.loc[keep_rows, f"brier_{th_label}"] = brier_score

    return score_table



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

    # Skip paths that do not exist
    paths_to_eval = [p for p in paths_to_eval if p.exists()]
    print(f"Evaluating {len(paths_to_eval)} paths from split '{split}'.")

    score_path = eval_cfg.get("score_path", "evaluation_scores.csv")
    scores_table = collect_scores(checkpoint_path, paths_to_eval, cfg["clinical_csv"], score_path, cfg=cfg)

    # Overall performance metrics
    auc, ci = calculate_auc_ci(scores_table["time"].values, scores_table["event"], scores_table["risk_score"].values)
    print(f"Model {checkpoint_path} overall evaluation results:\nAUC: {auc:.3f}, CI: {ci:.3f} (N={len(scores_table)})")
    print(f"Events: {scores_table['event'].sum().astype(int)} / {len(scores_table)} ({scores_table['event'].mean() * 100:.1f}%)")

    stratify_by = ["project_id"]
    for col in stratify_by:
        if col not in scores_table.columns:
            raise ValueError(f"Column '{col}' not found in the scores table.")
        print(f"Stratified by {col}:")
        for value, group in scores_table.groupby(col):
            N = len(group)
            auc, ci = calculate_auc_ci(group["time"].values, group["event"], group["risk_score"].values)
            print(f"  {col}={value}: AUC={auc:.3f}, CI={ci:.3f} (N={N}). Events: {group['event'].sum().astype(int)} / {N} ({group['event'].mean() * 100:.1f}%)")

    return
    scores_table = calculate_brier_scores(scores_table)

    plots_path = eval_cfg.get("plots_path", "evaluation_plots.pdf")
    if plots_path:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        sns.set_theme("notebook", style="whitegrid")
        with PdfPages(plots_path) as pdf:
            # Boxplot of risk scores stratified by event and cancer type (column "project_id")
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(x="project_id", y="risk_score", hue="event", data=scores_table)
            plt.title("Risk Scores by Cancer Type and Event")
            plt.xlabel("Cancer Type")
            plt.ylabel("Risk Score")
            plt.legend(title="Event", loc="upper right")
            pdf.savefig(fig)

            # Histogram of Brier scores, stratified by cancer type
            fig = plt.figure(figsize=(10, 6))
            brier_cols = [col for col in scores_table.columns if col.startswith("brier_")]
            brier_cols = brier_cols[0:1]
            for brier_col in brier_cols:
                sns.histplot(scores_table, x=brier_col, hue="project_id", kde=True, stat="density", common_norm=False)
            plt.title("Brier Scores Distribution by Cancer Type")
            plt.xlabel("Brier Score")
            plt.ylabel("Density")
            pdf.savefig(fig)

            # Boxplot of Brier scores by cancer type
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(x="project_id", y=brier_cols[0], data=scores_table)
            plt.title("Brier Scores by Cancer Type")
            plt.xlabel("Cancer Type")
            plt.ylabel("Brier Score")
            pdf.savefig(fig)


if __name__ == "__main__":
    main()
