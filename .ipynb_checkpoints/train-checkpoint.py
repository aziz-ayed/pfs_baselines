# train.py – Updated Script

import os, argparse, yaml, warnings, math
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

from src.dataset import PatchBagDataset
from src.collate import pad_collate
from src import models # Now we'll use the models module directly
from src.metrics import harrell, td_auc
from src.seed import set_seed
import os, sys
import h5py
import random
from pathlib import Path

def save_checkpoint(state, is_best, latest_path, best_path):
    """Saves model state, overwriting the latest and optionally the best."""
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, latest_path)
    if is_best:
        torch.save(state, best_path)

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

# ------------------------------------------------------------------ main loop

torch.set_num_threads(1)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def run_worker(rank: int, world: int, cfg: dict):
    # ---------------- distributed init (torchrun) ---------------- #
    if world > 1 and dist.is_available() and not dist.is_initialized():
        ddp_setup(rank, world)

    base_seed = cfg.get("seed", 42)
    set_seed(base_seed + rank)

    # --- Load pre-computed PATHS from the file ---
    train_paths, val_paths, test_paths, saved_dim = torch.load(cfg["split_file"], weights_only=False)
    feature_dir = Path(cfg["feature_dir"])

    train_paths = [feature_dir / p for p in train_paths]
    val_paths = [feature_dir / p for p in val_paths]
    test_paths = [feature_dir / p for p in test_paths]

    if rank == 0:
        print(f"✅ Reconstructed paths to use feature directory: {feature_dir}")

    train_frac = cfg.get("train_frac", 1.0)
    if train_frac < 1.0:
        if rank == 0:
            print(f"Using {train_frac * 100:.0f}% of the training data.")
        random.shuffle(train_paths)
        num_train_samples = int(len(train_paths) * train_frac)
        train_paths = train_paths[:num_train_samples]

    # --- Determine feature dimension dynamically ---
    dim = cfg.get("feature_dim")
    if dim is None:
        if rank == 0:
            print("Auto-detecting feature dimension from the first training file...")
        with h5py.File(train_paths[0], "r") as f:
            dim = f["features"].shape[1]

    # --- Create datasets using the pre-computed paths ---
    train_set = PatchBagDataset(paths=train_paths, clinical_csv=cfg["clinical_csv"], organs_to_use=cfg.get("train_on_organs", "all"))
    val_set = PatchBagDataset(paths=val_paths, clinical_csv=cfg["clinical_csv"], organs_to_use=cfg.get("single_organ_study", "all"))

    if rank == 0:
        print(f"✅ Loaded pre-computed splits from {cfg['split_file']}")
        print(f"✅ Feature dimension set to: {dim}")
        print(f"Training samples: {len(train_set)}, Validation samples: {len(val_set)}")

    # ---------------- DataLoader ---------------- #
    train_samp = DistributedSampler(train_set, shuffle=True, seed=base_seed) if world > 1 else None
    prefetch = cfg.get("prefetch_factor", 2)
    persistent = cfg.get("persistent_workers", False)

    def worker_init_fn(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], sampler=train_samp,
        shuffle=train_samp is None, num_workers=cfg["num_workers"],
        collate_fn=pad_collate, pin_memory=True,
        prefetch_factor=prefetch, persistent_workers=persistent,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"],
        collate_fn=pad_collate, pin_memory=True,
        prefetch_factor=prefetch, persistent_workers=persistent,
        worker_init_fn=worker_init_fn
    )

    # ---------------- model ---------------- #
    # <<< CHANGE: Updated model instantiation logic for multimodal support
    dropout_p = cfg.get("dropout_p", 0.25)
    model_name = cfg.get("model", "AttnMILNewCox") # Default to a unimodal model

    if model_name == "MultimodalCox":
        if rank == 0:
            print("✅ Initializing MultimodalCox model.")
        net = models.MultimodalCox(d_path=dim, dropout=dropout_p).cuda(rank)
    else:
        # Fallback to original unimodal logic
        if rank == 0:
            print(f"✅ Initializing Unimodal model: {model_name}")
        Net = getattr(models, model_name)
        # Handle models that might not have a dropout parameter
        try:
            net = Net(dim, dropout=dropout_p).cuda(rank)
        except TypeError:
            net = Net(dim).cuda(rank)
    # >>> END OF CHANGE

    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])

    opt, scaler = (
        torch.optim.Adam(net.parameters(), lr=float(cfg["learning_rate"]),
                         weight_decay=float(cfg["weight_decay"])),
        torch.amp.GradScaler('cuda'),
    )

    # ---------------- W&B (rank-0 only) ---------------- #
    if rank == 0 and cfg["wandb"]["mode"] != "disabled":
        wandb.init(project=cfg["wandb"]["project"],
                   name=cfg["wandb"]["run_name"],
                   mode=cfg["wandb"]["mode"],
                   group=cfg.get("wandb", {}).get("group"),
                   id=cfg.get("wandb", {}).get("id"),
                   config=cfg,
                   resume="allow")
        if cfg["wandb"].get("log_grads", False):
            wandb.watch(net, log_freq=100)

    # --- CHECKPOINTING & RESUME ---
    start_epoch = 0
    best_val_ci = 0.0
    if rank == 0 and cfg.get("save_checkpoints", False):
        save_dir = Path(cfg.get("ckpt_dir", "checkpoints"))
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_model_name = cfg.get("wandb", {}).get("name", "model")
        latest_path = save_dir / f"{ckpt_model_name}_latest.pth"
        best_path = save_dir / f"{ckpt_model_name}_best.pth"

    resume_path = cfg.get("resume_path")
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=f'cuda:{rank}', weights_only=False)
        model_to_load = net.module if world > 1 else net
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_ci = checkpoint.get('best_val_ci', 0.0)
        if rank == 0:
            print(f"✅ Resumed training from checkpoint: {resume_path}")
            print(f"   Starting at epoch {start_epoch}, Best CI so far: {best_val_ci:.4f}")

    if rank == 0:
        print("Collecting training survival data for validation reference...")
        train_T, train_E = [], []
        with torch.no_grad():
            # <<< CHANGE: Unpack 5 items from the data loader to account for rna_feats
            for _, t, e, _, _ in tqdm(train_loader, desc="Collecting T/E", ncols=100):
                train_T.append(t)
                train_E.append(e)
        train_T = np.concatenate(train_T)
        train_E = np.concatenate(train_E)
        print("✅ Done.")


    # --- MAIN LOOP ---
    for epoch in range(start_epoch, cfg["epochs"]):
        if train_samp: train_samp.set_epoch(epoch)
        net.train()
        epoch_loss = 0.0
        bar = tqdm(train_loader, disable=rank != 0, desc=f"Epoch {epoch:02d}", ncols=100)

        # <<< CHANGE: Unpack 5 items and pass both features to the model
        for feats, t, e, organs, rna_feats in bar:
            feats, t, e = feats.cuda(rank), t.cuda(rank), e.cuda(rank)
            rna_feats = rna_feats.cuda(rank) # Move RNA features to GPU

            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                # Pass both tensors to the model
                risk = net(feats, rna_feats)
                loss = cox_loss(risk, t, e)

            epoch_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            if rank == 0:
                bar.set_postfix(loss=loss.item())
        # >>> END OF CHANGE

        # --- VALIDATION & LOGGING (Rank 0 only) ---
        if rank == 0:
            net.eval()
            
            val_R, val_T, val_E, val_O = [], [], [], []
            with torch.no_grad():
                # <<< CHANGE: Unpack 5 items and pass both features to the model
                for feats, t, e, organs, rna_feats in tqdm(val_loader, desc="Val Metrics", ncols=100):
                    # Move both feature sets to GPU for inference
                    r = net(feats.cuda(rank), rna_feats.cuda(rank)).cpu()
                    val_R.append(r); val_T.append(t); val_E.append(e); val_O.append(organs)
            # >>> END OF CHANGE
            
            val_T = np.concatenate(val_T); val_E = np.concatenate(val_E); val_R = np.concatenate(val_R); val_O = np.concatenate(val_O)

            # --- Calculate All Metrics ---
            log_dict = {}
            landmark_times = [3 * 30.5, 6 * 30.5, 12*30.5, 18*30.5]

            # 2. Validation Set Metrics (Overall)
            val_ci = harrell(val_T, val_E, val_R)
            log_dict['val/ci'] = val_ci
            val_auc_mean = 0.0
            if np.any(val_E == 1):
                val_auc_scores, val_auc_mean = td_auc(train_T, train_E, val_T, val_E, val_R, landmark_times)
                log_dict['val/auc_mean'] = val_auc_mean
                for time, score in zip(landmark_times, val_auc_scores):
                    log_dict[f'val/auc_{int(time/30.5)}mo'] = score

            # 3. Validation Set Metrics (Per-Organ)
            print("\n--- Per-Organ Performance (Validation Set) ---")
            for organ_name, organ_idx in val_set.organ_to_idx.items():
                mask = (val_O == organ_idx)
                if np.sum(mask) > 0:
                    ci_organ = harrell(val_T[mask], val_E[mask], val_R[mask])
                    log_dict[f'val/ci_{organ_name.lower()}'] = ci_organ
                    print(f"{organ_name} C-Index: {ci_organ:.3f} ({np.sum(mask)} samples)")
                    if np.any(val_E[mask] == 1):
                        auc_scores_organ, _ = td_auc(train_T, train_E, val_T[mask], val_E[mask], val_R[mask], landmark_times)
                        for time, score in zip(landmark_times, auc_scores_organ):
                            log_dict[f'val/auc_{int(time/30.5)}mo_{organ_name.lower()}'] = score

            print(f"Epoch {epoch:03d} • CI={val_ci:.3f} • tAUC={val_auc_mean:.3f}")

            # --- Check if this is the best model ---
            is_best = val_ci > best_val_ci
            if is_best:
                best_val_ci = val_ci
                print(f"🏆 New best model found with C-Index: {best_val_ci:.4f}")

            if cfg["wandb"]["mode"] != "disabled":
                log_dict["train/epoch_loss"] = epoch_loss / len(train_loader)
                log_dict["epoch"] = epoch
                wandb.log(log_dict)

            # --- Save Checkpoints ---
            if cfg.get("save_checkpoints", False):
                model_state = net.module.state_dict() if world > 1 else net.state_dict()
                state = {
                    "model_name": model_name, # <<< CHANGE: Save the correct model name
                    "dim": dim,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": opt.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": epoch,
                    "best_val_ci": best_val_ci,
                    "config": cfg
                }
                save_checkpoint(state, is_best, latest_path, best_path)
            
            net.train()

    if rank == 0 and cfg["wandb"]["mode"] != "disabled":
        wandb.finish()


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, help="override random seed")
    parser.add_argument("--run_name", type=str, help="override W&B run name")
    parser.add_argument("--local_rank", type=int, default=0, help="Rank of the process on the node")
    parser.add_argument("--local-rank", type=int, dest="local_rank", help=argparse.SUPPRESS)
    opts, _ = parser.parse_known_args()

    cfg = yaml.safe_load(open(opts.config))

    if opts.seed is not None:
        cfg["seed"] = opts.seed
    if opts.run_name is not None:
        cfg.setdefault("wandb", {})["run_name"] = opts.run_name
        
    world = len(cfg["gpus"])

    if "LOCAL_RANK" in os.environ: # launched with torchrun
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        run_worker(rank, world, cfg)
    else: # fallback to mp.spawn
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg["gpus"]))
        mp.spawn(run_worker, nprocs=world, args=(world, cfg))