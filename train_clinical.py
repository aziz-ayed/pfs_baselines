# train_clinical.py – Train a model based on clinical features

import argparse, yaml
import json
import os
import h5py
import pathlib
import pprint
import random
from datetime import timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm                      # progress bar
import wandb                              # experiment tracker

from src.dataset import PatchBagDataset
from src.collate import pad_collate
from eval import calculate_auc_ci
from src import models
from src.models import *


# --------------------------------------------------------------------- utils

def cox_loss(risk, t, e, eps=1e-8, clamp_min=-10.0, clamp_max=10.0):
    idx = torch.argsort(t, descending=True)
    risk_clamped = risk[idx].clamp(min=clamp_min, max=clamp_max)
    hr  = torch.exp(risk[idx])
    cumsum_hr = torch.cumsum(hr, 0).clamp(min=eps)
    return torch.nanmean(-(risk_clamped - torch.log(cumsum_hr)) * e[idx])


def get_model(aggregator: str, dim: int, rank: int=None, topk_corr: int=256) -> Tuple[str, nn.Module]:
    model_name = aggregator + "Cox"
    Net = eval(model_name)
    if aggregator == 'TransMIL':
        # Only pass 'topk' to the model that actually uses it
        net = Net(dim, topk=topk_corr)
    else:
        net = Net(dim)

    if rank is not None:
        net = net.to(rank)
    return model_name, net


# ------------------------------------------------------------------ main loop
    
torch.set_num_threads(1)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def _choose_time(row):
    if row["progression_recurrence_event"] == 1:
        return row["days_to_progression_recurrence"]
    return row["max_follow_up_days"]

def prepare_time_event_columns(clinical_df: pd.DataFrame):
    time = clinical_df.apply(_choose_time, axis=1)
    event = clinical_df["progression_recurrence_event"]
    return time, event

def seed_torch(device, seed=1234):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Create torch datasets from the DataFrame
# Can use a simple TensorDataset
def df_to_dataset(df, feature_cols, target_cols) -> torch.utils.data.Dataset:
    features = torch.from_numpy(df[feature_cols].values).to(torch.float32)
    targets = torch.from_numpy(df[target_cols].values).to(torch.float32)
    dataset = torch.utils.data.TensorDataset(features, targets)
    return dataset

def fill_in_yaml_variables_recursive(cfg, source=None):
    if source is None:
        source = cfg
    for key, value in cfg.items():
        if isinstance(value, dict):
            fill_in_yaml_variables_recursive(value, source)
        elif isinstance(value, str):
            try:
                cfg[key] = value.format(**source)
            except KeyError:
                pass
    return cfg

def _get_dts():
    return f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}]"

def run_worker(cfg: dict):
    clinical_csv = cfg["clinical_csv"]
    clinical_df = pd.read_csv(clinical_csv)
    id_col = "submitter_id"
    # device = torch.device(f"cuda:{cfg['gpus'][0]}") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    rank = 0

    # --- Determine feature dimension dynamically ---
    dim = cfg.get("feature_dim")
    assert dim is not None, "Feature dimension must be specified in the config file."
    feature_cols = clinical_df.columns.tolist()[-dim:]

    # feature_categories = ['age_at_index', 'gender', 'race', 'ethnicity', 'ajcc_pathologic_n', 'ajcc_pathologic_m', 'ajcc_pathologic_t', 'laterality', 'morphology']
    # exclude_categories = ["ajcc_pathologic_n", "ajcc_pathologic_m", "ajcc_pathologic_t"]
    exclude_categories = []

    if "seed" in cfg:
        seed = cfg["seed"]
        seed_torch(device, seed)

    split_path: Optional[str] = cfg.get("split_file", None)
    assert os.path.exists(split_path), f"Split file {split_path} does not exist."
    if split_path:
        if split_path.endswith(".json"):
            with open(split_path, "r") as f:
                split_dict = json.load(f)
        elif split_path.endswith(".csv"):
            split_df = pd.read_csv(split_path)
            _id_col = split_df.columns[0]  if split_df.columns[0] in {"patient_id", "submitter_id"} else None
            assert _id_col is not None, "Split file must contain a column for patient IDs."
            split_dict = {
                "train": split_df[split_df["split"] == "train"][_id_col].tolist(),
                "val": split_df[split_df["split"].isin({"val", "dev", "validation"})][_id_col].tolist(),
                "test": split_df[split_df["split"] == "test"][_id_col].tolist(),
            }

    def _value_happens(inval):
        if pd.isna(inval):
            return False
        if isinstance(inval, str):
            return inval.lower() in {"yes", "true", "1"}
        else:
            return inval >= 0

    def _is_progression(row):
        prog_days = _value_happens(row["days_to_progression"])
        rec_days = _value_happens(row["days_to_recurrence"])
        # is_yes = row["progression_or_recurrence"].lower() == "yes"
        is_yes = _value_happens(row["progression_recurrence_event"])
        is_dead = _value_happens(row["days_to_death"])
        bool_prog = prog_days or rec_days or is_yes or is_dead
        return 1 if bool_prog else 0

    if "max_follow_up_days" not in clinical_df.columns:
        clinical_df["max_follow_up_days"] = clinical_df[["days_to_follow_up", "days_to_last_follow_up", "days_to_death"]].max(axis=1)
    if "days_to_progression_recurrence" not in clinical_df.columns:
        event_columns = ["days_to_progression", "days_to_recurrence", "days_to_follow_up", "days_to_last_follow_up", "days_to_death"]
        clinical_df["days_to_progression_recurrence"] = clinical_df[event_columns].min(axis=1).clip(lower=0)

    # Hacks
    if "progression_or_recurrence" in clinical_df.columns:
        # Rename columns to match expected names
        clinical_df.rename(columns={
            "progression_or_recurrence": "progression_recurrence_event",
        }, inplace=True)
        clinical_df["progression_recurrence_event"] = clinical_df.apply(
            _is_progression, axis=1
        )

    target_cols = ["time", "event"]
    if "time" not in clinical_df.columns or "event" not in clinical_df.columns:
        # Prepare time and event columns if not present
        time, event = prepare_time_event_columns(clinical_df)
        clinical_df["time"] = time.values
        clinical_df["event"] = event.values

    # Create column to label split
    if id_col in clinical_df.columns:
        for label in ["train", "val", "test"]:
            cur_pids = split_dict[label]
            clinical_df.loc[clinical_df[id_col].isin(cur_pids), "split"] = label

    keep_rows = pd.notna(clinical_df["days_to_progression_recurrence"]) & pd.notna(clinical_df["progression_recurrence_event"])
    print(f"Keeping {keep_rows.sum()} rows out of {len(clinical_df)} based on 'days_to_progression_recurrence' and 'progression_recurrence_event'.")
    clinical_df = clinical_df[keep_rows].reset_index(drop=True)

    clinical_df.to_csv(clinical_csv.replace(".csv", "_mod.csv"), index=False)

    if exclude_categories:
        keep_feature_cols = [col for col in feature_cols if "_".join(col.split("_")[0:-1]) not in exclude_categories]
        print(f"Keeping {len(keep_feature_cols)} feature columns out of {len(feature_cols)} based on exclusion criteria.")
        feature_cols = keep_feature_cols

    dim = len(feature_cols)

    limit_train_batches: Optional[int] = cfg.get("limit_train_batches", None)
    limit_val_batches: Optional[int] = cfg.get("limit_val_batches", None)
    
    # --- Create datasets ---
    train_df = clinical_df[clinical_df[id_col].isin(split_dict["train"])]
    val_df   = clinical_df[clinical_df[id_col].isin(split_dict["val"])]

    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples.")
    print(f"Events in training set: {train_df['event'].sum()} / {len(train_df)}")
    print(f"Events in validation set: {val_df['event'].sum()} / {len(val_df)}")

    train_set = df_to_dataset(train_df, feature_cols, target_cols)
    val_set   = df_to_dataset(val_df, feature_cols, target_cols)

    train_val_pid_overlap = set(train_df[id_col]).intersection(set(val_df[id_col]))
    print(f"Patient ID overlap between train and val sets: {len(train_val_pid_overlap)}")

    # ---------------- DataLoader ---------------- #
    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"],
    )

    # ---------------- model ---------------- #
    model_name, net = get_model(cfg["aggregator"], dim, rank=rank, topk_corr=cfg.get("topk_corr", 256))

    opt, scaler = (
        torch.optim.Adam(net.parameters(), lr=float(cfg["learning_rate"]),
                         weight_decay=float(cfg["weight_decay"])),
        torch.amp.GradScaler(device),
    )

    # ---------------- W&B (rank‑0 only) ---------------- #
    if rank == 0 and cfg["wandb"]["mode"] != "disabled":
        wandb.init(project=cfg["wandb"]["project"],
                   name   =cfg["wandb"]["run_name"],
                   mode   =cfg["wandb"]["mode"],
                   config =cfg)
        if cfg["wandb"].get("log_grads", False):
            wandb.watch(net, log_freq=100)

    save_dir = pathlib.Path(cfg.get("save_dir", "results"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- epochs ---------------- #
    for epoch in range(cfg["epochs"]):
        net.train()
        net.to(device)
        epoch_loss = 0.0  # Initialize for each epoch

        bar = tqdm(train_loader, disable=rank != 0,
                   desc=f"Epoch {epoch:02d}", ncols=100)
        
        # --- TRAINING LOOP ---
        batch_num = 0
        for feats, targets in bar:
            if limit_train_batches is not None and batch_num >= limit_train_batches:
                break
            batch_num += 1
            t, e = targets[:, 0], targets[:, 1]
            # feats, t, e = feats.cuda(rank), t.cuda(rank), e.cuda(rank)
            feats, t, e = feats.to(device), t.to(device), e.to(device)
            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=False):
                mini_batches = cfg.get("mini_batches", 1)
                if mini_batches > 1:
                    # Split the batch into mini-batches
                    batch_size = feats.size(0)
                    # Use numpy to split the batch into mini-batches
                    index = np.arange(batch_size)
                    mini_batch_indexes = np.array_split(index, mini_batches)
                    losses = []
                    for cur_mini_batch_indexes in mini_batch_indexes:
                        mini_feats = feats[cur_mini_batch_indexes]
                        mini_t = t[cur_mini_batch_indexes]
                        mini_e = e[cur_mini_batch_indexes]
                        preds = net(mini_feats)
                        loss = cox_loss(preds, mini_t, mini_e)
                        losses.append(loss)
                    loss = torch.stack(losses).mean()
                else:
                    preds = net(feats)
                    loss = cox_loss(preds, t, e)

            if torch.isnan(loss):
                print(f"NaN loss encountered at epoch {epoch}, batch {batch_num}. Skipping this batch.")
                continue

            epoch_loss += loss.item()
            # scaler.scale(loss).backward()
            # scaler.unscale_(opt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
            # scaler.step(opt)
            # scaler.update()

            if rank == 0:
                bar.set_postfix(loss=loss.item())

        # --- VALIDATION & LOGGING ---
        if rank == 0:
            net.eval()

            R, T, E = [], [], []
            batch_num = 0
            with torch.no_grad():
                for feats, targets in tqdm(val_loader, desc="Val", ncols=100):
                    if limit_val_batches is not None and batch_num >= limit_val_batches:
                        break
                    batch_num += 1
                    r = net(feats).cpu()
                    t = targets[:, 0].cpu()
                    e = targets[:, 1].cpu()
                    R.append(r); T.append(t); E.append(e)

            if batch_num > 0:
                T = np.concatenate(T); E = np.concatenate(E); R = np.concatenate(R).squeeze(-1)
                count_nans = np.isnan(R).sum()
                if count_nans:
                    print(f"Warning: {count_nans} / {len(R)} nan values found in predictions. Skipping validation.")
                    continue
                auc, ci = calculate_auc_ci(T, E, R)

                avg_epoch_loss = epoch_loss / len(train_loader)
                print(f"{_get_dts()} Epoch {epoch:03d} • CI={ci:.3f} • tAUC={auc:.3f} Loss: {avg_epoch_loss:0.3f} Seed {cfg['seed']}")
                perf_dict = {
                        "train/epoch_loss": avg_epoch_loss,
                        "val/ci": ci,
                        "val/auc": auc,
                        "epoch": epoch
                    }
                # Save performance metrics to file
                perf_file = save_dir / f"{model_name}_epoch_{epoch:03d}_perf.json"
                with open(perf_file, "w") as f:
                    json.dump(perf_dict, f, indent=2)


                if cfg["wandb"]["mode"] != "disabled":
                    wandb.log(perf_dict)
            else:
                print(f"Epoch {epoch:03d} • No validation set provided.")
            
            net.train()

        # --- Save model checkpoint (rank‑0 only) ---
        if rank == 0 and cfg.get("save_checkpoints", False):
            save_path = save_dir / f"{model_name}_epoch_{epoch:03d}.pth"
            torch.save({
                "model_name": model_name,
                "dim": dim,
                "feature_cols": feature_cols,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "config": cfg
            }, save_path)
            print(f"{_get_dts()}  Model saved to {save_path}")

    if rank == 0 and cfg["wandb"]["mode"] != "disabled":
        wandb.finish()


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--local_rank",  type=int, default=0,
                        help="Rank of the process on the node")
    # This is for backward compatibility with some launchers.
    parser.add_argument("--local-rank", type=int, dest="local_rank",
                        help=argparse.SUPPRESS)
    opts, _ = parser.parse_known_args()

    # Fill in variables from yaml config
    cfg = yaml.safe_load(open(opts.config))
    seeds = list(cfg.get("seeds", [1234]))
    if isinstance(seeds, int):
        seeds = [seeds]

    for seed in seeds:
        var_keys = ["seed", "split"]
        cfg["seed"] = seed
        for vk in var_keys:
            if vk not in cfg and vk in cfg["eval"]:
                cfg[vk] = cfg["eval"][vk]
        fill_in_yaml_variables_recursive(cfg)
        fill_in_yaml_variables_recursive(cfg["eval"], cfg)

        pprint.pprint(f"{_get_dts()} Config for seed {seed}:")
        pprint.pprint(cfg)

        run_worker(cfg)
