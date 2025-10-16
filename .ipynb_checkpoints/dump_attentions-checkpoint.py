#!/usr/bin/env python3
import argparse
import pathlib as pl
import warnings
from typing import List, Tuple, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# your repo modules
from src.dataset import PatchBagDataset
from src.collate import pad_collate
from src import models


# --------------------------- ckpt / model ---------------------------

def _load_checkpoint(ckpt_path: pl.Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        raise ValueError("Checkpoint must contain 'config'.")
    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint missing 'model_state_dict'.")
    if "dim" not in ckpt:
        raise ValueError("Checkpoint missing feature dimension 'dim'.")
    return ckpt["config"], ckpt["model_state_dict"], ckpt["dim"]


def _build_model(cfg: dict, dim: int, device: torch.device) -> torch.nn.Module:
    model_name = cfg.get("model", "AttnMILNewCox")
    dropout_p = cfg.get("dropout_p", 0.25)

    if model_name == "MultimodalCox":
        net = models.MultimodalCox(d_path=dim, dropout=dropout_p).to(device)
        att_mod = net.path_agg  # where the MIL attention lives
    else:
        Net = getattr(models, model_name)
        try:
            net = Net(dim, dropout=dropout_p).to(device)
        except TypeError:
            net = Net(dim).to(device)
        att_mod = net  # attention on the net itself

    net.eval()
    return net, att_mod


# --------------------------- coords utils ---------------------------

def _find_coords_file(coords_dir: pl.Path, slide_stem: str, suffix: str) -> Optional[pl.Path]:
    """
    Try {stem}{suffix}.h5 (e.g., TCGA-XXX_patches.h5). If not found, fall back
    to the first match containing 'patch' with the same stem.
    """
    cand = coords_dir / f"{slide_stem}{suffix}.h5"
    if cand.exists():
        return cand
    alts = sorted(coords_dir.glob(f"{slide_stem}*patch*.h5"))
    return alts[0] if alts else None


def _read_coords(coords_h5: pl.Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(coords_h5.as_posix(), "r") as f:
        # Your inspection showed 'coords' is a dataset (Nx2).
        if "coords" not in f:
            raise RuntimeError(f"{coords_h5} has no 'coords' dataset.")
        c = np.asarray(f["coords"])
    if c.ndim != 2 or c.shape[1] < 2:
        raise RuntimeError(f"'coords' in {coords_h5} must be Nx2 (got {c.shape})")
    return c[:, 0].astype(np.int64), c[:, 1].astype(np.int64)


# --------------------------- attention math ---------------------------

@torch.no_grad()
def _att_per_slide(att_mod: torch.nn.Module, x_b: torch.Tensor) -> torch.Tensor:
    """
    Compute gated-attention for a single slide (no padding in x_b).
    x_b: (N, d) on the correct device.
    Returns: (N,) attention weights that sum to 1.
    """
    x_b = x_b.unsqueeze(0)  # (1, N, d)

    # Optional feature extractor (AttnMILNew*)
    if hasattr(att_mod, "feature_extractor") and att_mod.feature_extractor is not None:
        x_b = att_mod.feature_extractor(x_b)

    # Require the gated-attention parts
    if not (hasattr(att_mod, "V") and hasattr(att_mod, "U") and hasattr(att_mod, "w")):
        raise RuntimeError("This model doesn't expose V/U/w for attention dumping.")

    logits = att_mod.w(torch.tanh(att_mod.V(x_b)) * torch.sigmoid(att_mod.U(x_b)))  # (1,N,1)
    att = torch.softmax(logits, dim=1)[0, :, 0].contiguous()  # (N,)
    return att


def _valid_mask_from_padding(feats_b: torch.Tensor) -> torch.Tensor:
    """
    Detect which rows are real vs padding. Your pad_collate pads with zeros,
    so a row is padding if all features == 0.
    feats_b: (Nmax, d)
    Returns: boolean mask (Nmax,) True for valid rows.
    """
    return feats_b.abs().sum(dim=-1) > 0


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser("Dump patch attentions aligned to TRIDENT coords (no CSVs needed).")
    ap.add_argument("--checkpoint", type=pl.Path, required=True, help="Model checkpoint (.pth).")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True, help="Which split to dump.")
    ap.add_argument("--coords-dir", type=pl.Path, required=True, help="Folder containing {slide}_patches.h5 with a 'coords' dataset.")
    ap.add_argument("--coords-suffix", type=str, default="_patches", help="Suffix for coords files (default: _patches).")
    ap.add_argument("--out-dir", type=pl.Path, required=True, help="Folder to write one CSV per slide.")
    ap.add_argument("--output-csv", type=pl.Path, default=None, help="Optional merged CSV of all slides.")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (order is preserved).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ckpt + model
    cfg, state_dict, dim = _load_checkpoint(args.checkpoint, device)
    net, att_mod = _build_model(cfg, dim, device)
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        warnings.warn(f"Missing keys: {missing}")
    if unexpected:
        warnings.warn(f"Unexpected keys: {unexpected}")

    # Resolve split → feature file list (we use the dataset to load feats)
    split_file = pl.Path(cfg["split_file"])
    feature_dir = pl.Path(cfg["feature_dir"])
    all_splits = torch.load(split_file, weights_only=False)
    split_idx = {"train": 0, "val": 1, "test": 2}[args.split]
    raw_paths: List[str] = all_splits[split_idx]
    eval_paths = [feature_dir / pl.Path(p).name for p in raw_paths]

    # Dataset / Loader (exactly like eval.py)
    dataset = PatchBagDataset(paths=eval_paths, clinical_csv=cfg["clinical_csv"])
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_rows = []

    slide_ptr = 0  # tracks our position in dataset.paths (DataLoader yields in order)

    with torch.no_grad():
        for feats_b, _t, _e, _org, _rna in tqdm(loader, desc=f"attn-{args.split}", unit="batch"):
            B, Nmax, d = feats_b.shape
            feats_b = feats_b.to(device)

            for j in range(B):
                slide_path = dataset.paths[slide_ptr]
                slide_stem = slide_path.stem
                slide_ptr += 1

                # 1) unpad this bag
                row_mask = _valid_mask_from_padding(feats_b[j])  # (Nmax,)
                L = int(row_mask.sum().item())
                if L == 0:
                    warnings.warn(f"[skip] {slide_stem}: empty bag after masking")
                    continue
                x_b = feats_b[j, row_mask, :]  # (L, d)

                # 2) compute attentions for this slide
                att = _att_per_slide(att_mod, x_b).detach().cpu().numpy().astype(np.float32)  # (L,)

                # 3) read coords for this slide
                coords_h5 = _find_coords_file(args.coords_dir, slide_stem, args.coords_suffix)
                if coords_h5 is None or not coords_h5.exists():
                    warnings.warn(f"[skip] {slide_stem}: no coords H5 in {args.coords_dir}")
                    continue
                x0, y0 = _read_coords(coords_h5)  # (N,), (N,)
                Nc = x0.shape[0]

                # 4) length check (coords vs real patches)
                if Nc != L:
                    msg = f"[warn] {slide_stem}: L_att={L} != N_coords={Nc} → truncating to min"
                    warnings.warn(msg)
                    N = min(L, Nc)
                    att = att[:N]
                    x0 = x0[:N]
                    y0 = y0[:N]

                # 5) write CSV row(s)
                df = pd.DataFrame({
                    "slide_id": slide_stem,
                    "idx": np.arange(len(att), dtype=np.int64),
                    "x0": x0,
                    "y0": y0,
                    "attention": att,
                })
                out_csv = args.out_dir / f"{slide_stem}.csv"
                df.to_csv(out_csv, index=False)

                if args.output_csv is not None:
                    merged_rows.append(df)

    if args.output_csv is not None:
        if merged_rows:
            big = pd.concat(merged_rows, ignore_index=True)
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            big.to_csv(args.output_csv, index=False)
            print(f"[ok] merged → {args.output_csv} ({len(big)} rows)")
        else:
            print("[warn] nothing to merge; no rows written.")
    

if __name__ == "__main__":
    main()