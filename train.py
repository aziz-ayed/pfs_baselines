# train.py – Corrected Script

import argparse, yaml
import pathlib
from datetime import timedelta
from typing import Optional, Tuple

import numpy as np
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
import os
import h5py

# --------------------------------------------------------------------- utils

def cox_loss(risk, t, e):
    idx = torch.argsort(t, descending=True)
    hr  = torch.exp(risk[idx])
    return (-(risk[idx] - torch.log(torch.cumsum(hr, 0))) * e[idx]).mean()


def ddp_setup(rank: int, world: int):
    """Initialise NCCL backend and bind this rank to one CUDA device."""
    dist.init_process_group("nccl", rank=rank, world_size=world,
                            timeout=timedelta(minutes=30))
    torch.cuda.set_device(rank)

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

def run_worker(rank: int, world: int, cfg: dict):
    # ---------------- distributed init (torchrun) ---------------- #
    if world > 1 and dist.is_available() and not dist.is_initialized():
        ddp_setup(rank, world)

    # --- Load pre-computed PATHS from the file ---
    train_paths, val_paths, saved_dim = torch.load(cfg["split_file"], weights_only=False)
    feature_dir: Optional[str] = cfg.get("feature_dir", None)
    if feature_dir is not None:
        # If feature_dir is specified, change the directory of the files.
        feature_dir: pathlib.Path = pathlib.Path(feature_dir)
        train_paths = [feature_dir / p.name for p in train_paths]
        val_paths = [feature_dir / p.name for p in val_paths]

    # --- Determine feature dimension dynamically ---
    dim = cfg.get("feature_dim")
    if dim is None:
        # If not in config, auto-detect from the first training file
        if rank == 0:
            print("Auto-detecting feature dimension from the first training file...")
        with h5py.File(train_paths[0], "r") as f:
            dim = f["features"].shape[1]

    limit_train_batches: Optional[int] = cfg.get("limit_train_batches", None)
    limit_val_batches: Optional[int] = cfg.get("limit_val_batches", None)
    
    # --- Create datasets using the pre-computed paths ---
    train_set = PatchBagDataset(paths=train_paths, clinical_csv=cfg["clinical_csv"])
    val_set = PatchBagDataset(paths=val_paths, clinical_csv=cfg["clinical_csv"])
    
    if rank == 0:
        print(f"✅ Loaded pre-computed splits from {cfg['split_file']}")
        print(f"✅ Feature dimension set to: {dim}")
        print(f"Training samples: {len(train_set)}, Validation samples: {len(val_set)}")

    # ---------------- DataLoader ---------------- #
    train_samp = DistributedSampler(train_set) if world > 1 else None

    prefetch    = cfg.get("prefetch_factor", 2)
    persistent  = cfg.get("persistent_workers", False)

    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], sampler=train_samp,
        shuffle=train_samp is None, num_workers=cfg["num_workers"],
        collate_fn=pad_collate, pin_memory=True,
        prefetch_factor = prefetch, persistent_workers = persistent
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"], # No sampler here
        collate_fn=pad_collate, pin_memory=True,
        prefetch_factor = prefetch, persistent_workers = persistent
    )

    # ---------------- model ---------------- #
    model_name, net = get_model(cfg["aggregator"], dim, rank=rank, topk_corr=cfg.get("topk_corr", 256))

    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])

    opt, scaler = (
        torch.optim.Adam(net.parameters(), lr=float(cfg["learning_rate"]),
                         weight_decay=float(cfg["weight_decay"])),
        torch.amp.GradScaler("cuda"),
    )

    # ---------------- W&B (rank‑0 only) ---------------- #
    if rank == 0 and cfg["wandb"]["mode"] != "disabled":
        wandb.init(project=cfg["wandb"]["project"],
                   name   =cfg["wandb"]["run_name"],
                   mode   =cfg["wandb"]["mode"],
                   config =cfg)
        if cfg["wandb"].get("log_grads", False):
            wandb.watch(net, log_freq=100)

    # ---------------- epochs ---------------- #
    for epoch in range(cfg["epochs"]):
        if train_samp: train_samp.set_epoch(epoch)
        net.train()
        epoch_loss = 0.0  # Initialize for each epoch

        bar = tqdm(train_loader, disable=rank != 0,
                   desc=f"Epoch {epoch:02d}", ncols=100)
        
        # --- TRAINING LOOP ---
        batch_num = 0
        for feats, t, e in bar:
            if limit_train_batches is not None and batch_num >= limit_train_batches:
                break
            batch_num += 1
            feats, t, e = feats.cuda(rank), t.cuda(rank), e.cuda(rank)
            opt.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
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

            epoch_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            if rank == 0:
                bar.set_postfix(loss=loss.item())

        # --- VALIDATION & LOGGING ---
        if rank == 0:
            net.eval()
            if limit_val_batches is not None and limit_val_batches > 0:
                R, T, E = [], [], []
                batch_num = 0
                with torch.no_grad():
                    for feats, t, e in tqdm(val_loader, desc="Val", ncols=100):
                        if limit_val_batches is not None and batch_num >= limit_val_batches:
                            break
                        batch_num += 1
                        r = net(feats.cuda(rank)).cpu()
                        R.append(r); T.append(t); E.append(e)

                T = np.concatenate(T); E = np.concatenate(E); R = np.concatenate(R)
                auc, ci = calculate_auc_ci(T, E, R)

                print(f"Epoch {epoch:03d} • CI={ci:.3f} • tAUC={auc:.3f}")

                if cfg["wandb"]["mode"] != "disabled":
                    avg_epoch_loss = epoch_loss / len(train_loader)
                    wandb.log({
                        "train/epoch_loss": avg_epoch_loss,
                        "val/ci": ci,
                        "val/auc": auc,
                        "epoch": epoch
                    })
            else:
                print(f"Epoch {epoch:03d} • No validation set provided.")
            
            net.train()

        # --- Save model checkpoint (rank‑0 only) ---
        if rank == 0 and cfg.get("save_checkpoints", False):
            save_dir = pathlib.Path(cfg.get("save_dir", "results"))
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{model_name}_epoch_{epoch:03d}.pth"
            torch.save({
                "model_name": model_name,
                "dim": dim,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "config": cfg
            }, save_path)
            print(f"Model saved to {save_path}")

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

    cfg = yaml.safe_load(open(opts.config))
    world = len(cfg["gpus"])

    if "LOCAL_RANK" in os.environ:            # launched with torchrun
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        run_worker(rank, world, cfg)
    else:                                     # fallback to mp.spawn
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg["gpus"]))
        mp.spawn(run_worker, nprocs=world, args=(world, cfg))