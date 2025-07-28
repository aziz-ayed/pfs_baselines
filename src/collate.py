import torch, torch.nn.functional as F

def pad_collate(batch):
    # Unpack the new 5-element tuple
    feats, t, e, organs, rna_feats = zip(*batch)

    max_len = max(f.shape[0] for f in feats)
    feats_pad = torch.stack([F.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in feats])

    # Stack the RNA features
    return feats_pad, torch.tensor(t), torch.tensor(e), torch.tensor(organs), torch.stack(rna_feats)
