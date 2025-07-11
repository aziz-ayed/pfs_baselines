import argparse
import json
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
import sklearn.metrics
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
    if "TCGA" in path.name:
        return "-".join(path.name.split("-")[:3])
    else:
        return os.path.splitext(path.name)[0]  # For non-TCGA paths, use the filename without extension

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
        for feats, targets in tqdm.tqdm(loader, desc="Calculating risk scores", ncols=100):
            if device is not None:
                feats = feats.to(device)
            t, e = targets[:, 0].cpu().numpy(), targets[:, 1].cpu().numpy()
            r = net(feats).cpu()

            times.append(t)
            indicators.append(e)
            risk_scores.append(r)

    times = np.concatenate(times)
    indicators = np.concatenate(indicators)
    risk_scores = np.concatenate(risk_scores).squeeze(-1)

    return times, indicators, risk_scores

def df_to_dataset(df, feature_cols, target_cols) -> torch.utils.data.Dataset:
    features = torch.from_numpy(df[feature_cols].values).to(torch.float32)
    targets = torch.from_numpy(df[target_cols].values).to(torch.float32)
    dataset = torch.utils.data.TensorDataset(features, targets)
    return dataset

def collect_scores(checkpoint_path: str, clinical_csv: str, split: str="val",
                   score_path: str= "evaluation_scores.csv", device: Optional[torch.device] = None,
                   cfg: Optional[dict] = None):

    cfg = cfg or {}
    eval_cfg = cfg.get("eval", {})
    overwrite = eval_cfg.get("overwrite", False)
    target_cols = ["time", "event"]
    if not os.path.exists(score_path) or overwrite:
        clinical_csv = clinical_csv.replace(".csv", "_mod.csv") if "_mod.csv" not in clinical_csv else clinical_csv
        print(f"Loading clinical data from {clinical_csv} for split '{split}'")
        clinical_df = pd.read_csv(clinical_csv)
        clinical_df = clinical_df[clinical_df["split"] == split]

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

        feature_cols = checkpoint["feature_cols"]
        dataset = df_to_dataset(clinical_df, feature_cols, target_cols)

        loader = torch.utils.data.DataLoader(dataset, shuffle=False)
        times, indicators, risk_scores = collect_scores_from_loader(model, loader, device=device)

        # Make a table with different columns: embed_path, times, indicators, risk_scores
        scores_table = pd.DataFrame({
            "submitter_id": clinical_df["submitter_id"].values,
            "time": times,
            "event": indicators,
            "risk_score": risk_scores,
            "max_follow_up_days": clinical_df["max_follow_up_days"].values,
        })

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

def calculate_aucs_timepoints(score_table: pd.DataFrame):
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
        {"label": "18m", "days": 540},
        {"label": "24m", "days": 730},
        {"label": "10y", "days": 3650},
    ]

    for th in time_horizons:
        th_label = th["label"]
        th_days = th["days"]

        positive_within_window = (score_table[days_to_event_col] <= th_days) & (score_table[event_col] == 1)
        keep_rows = positive_within_window | (score_table[follow_up_col] >= th_days)
        positive_within_window = positive_within_window[keep_rows]
        risk_score = score_table.loc[keep_rows, score_col].values

        cur_time_auc = sklearn.metrics.roc_auc_score(positive_within_window, risk_score)
        cur_time_prc = sklearn.metrics.average_precision_score(positive_within_window, risk_score)
        th[f"auc"] = cur_time_auc
        th[f"prc"] = cur_time_prc
        th[f"n"] = len(risk_score)
        th[f"events"] = positive_within_window.sum()

    auc_table = pd.DataFrame(time_horizons)

    return auc_table


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

    if cfg["split_file"].endswith(".pt"):
        train_paths, val_paths, saved_dim = torch.load(cfg["split_file"], weights_only=False)
    elif cfg["split_file"].endswith(".json"):
        with open(cfg["split_file"], "r") as f:
            split_data = json.load(f)
        train_paths = [Path(p) for p in split_data["train"]]
        val_paths = [Path(p) for p in split_data["val"]]
        saved_dim = split_data.get("dim", None)
    else:
        raise ValueError(f"Unsupported split file format: {cfg['split_file']}")

    eval_cfg = cfg.get("eval", {})
    limit = eval_cfg.get("limit", None)
    split = eval_cfg.get("split", "")

    score_path = eval_cfg.get("score_path", "evaluation_scores.csv")
    scores_table = collect_scores(checkpoint_path, cfg["clinical_csv"], split=split, score_path=score_path, cfg=cfg)
    cancer_types = {
        "BRCA": "Breast",
        "LUAD": "Lung",
        "LUSC": "Lung",
        "READ": "Colon",
        "COAD": "Colon",
    }
    if "project_id" in scores_table.columns and "cancer_type" not in scores_table.columns:
        scores_table["cancer_type"] = scores_table["project_id"].apply(
            lambda x: cancer_types.get(x.split("-")[-1], "Other")
        )

    # Overall performance metrics
    auc, ci = calculate_auc_ci(scores_table["time"].values, scores_table["event"], scores_table["risk_score"].values)
    print(f"Model {checkpoint_path} overall evaluation results:\nAUC: {auc:.3f}, CI: {ci:.3f} (N={len(scores_table)})")
    print(f"Events: {scores_table['event'].sum().astype(int)} / {len(scores_table)} ({scores_table['event'].mean() * 100:.1f}%)")

    stratify_by = ["cancer_type"]
    for col in stratify_by:
        if col not in scores_table.columns:
            continue
        print(f"Stratified by {col}:")
        for value, group in scores_table.groupby(col):
            N = len(group)
            auc, ci = calculate_auc_ci(group["time"].values, group["event"], group["risk_score"].values)
            print(f"  {col}={value}: AUC={auc:.3f}, CI={ci:.3f} (N={N}). Events: {group['event'].sum().astype(int)} / {N} ({group['event'].mean() * 100:.1f}%)")

    aucs_timepoints = calculate_aucs_timepoints(scores_table)
    print(f"AUCs at different time points:\n{aucs_timepoints}")

    plots_path = eval_cfg.get("plots_path", "evaluation_plots.pdf")
    if plots_path and "cancer_type" in scores_table.columns:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        sns.set_theme("notebook", style="whitegrid", font_scale=1.2)

        with PdfPages(plots_path) as pdf:
            # Scatter plot of risk scores vs time, using only event-positive samples.
            # Colored by cancer type.
            fig = plt.figure(figsize=(10, 6))
            sns.scatterplot(x="time", y="risk_score", hue="cancer_type", data=scores_table[scores_table["event"] == 1])
            plt.title("Risk Scores vs Time (Event-Positive Samples)")
            plt.xlabel("Time (days)")
            plt.ylabel("Risk Score")
            plt.legend(title="Cancer Type", loc="upper right")
            pdf.savefig(fig)

            # Boxplot of risk scores stratified by event and cancer type
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(x="cancer_type", y="risk_score", hue="event", data=scores_table)
            plt.title("Risk Scores by Cancer Type and Event")
            plt.xlabel("Cancer Type")
            plt.ylabel("Risk Score")
            plt.legend(title="Event", loc="upper right")
            pdf.savefig(fig)

    if False:
            # Histogram of Brier scores, stratified by cancer type
            fig = plt.figure(figsize=(10, 6))
            brier_cols = [col for col in scores_table.columns if col.startswith("brier_")]
            brier_cols = brier_cols[0:1]
            for brier_col in brier_cols:
                sns.histplot(scores_table, x=brier_col, hue="cancer_type", kde=True, stat="density", common_norm=False)
            plt.title("Brier Scores Distribution by Cancer Type")
            plt.xlabel("Brier Score")
            plt.ylabel("Density")
            pdf.savefig(fig)

            # Boxplot of Brier scores by cancer type
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(x="cancer_type", y=brier_cols[0], data=scores_table)
            plt.title("Brier Scores by Cancer Type")
            plt.xlabel("Cancer Type")
            plt.ylabel("Brier Score")
            pdf.savefig(fig)


if __name__ == "__main__":
    main()
