import torch, torch.nn.functional as F

def pad_collate(batch):
    feats, t, e = zip(*batch)
    
    max_len = max(f.shape[0] for f in feats)
    feats_pad = torch.stack([F.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in feats])
    
    return feats_pad, torch.tensor(t), torch.tensor(e)