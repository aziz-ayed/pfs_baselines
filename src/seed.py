import os, random, numpy as np, torch

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:          # slows things a bit, but fully repeatable
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:                       # a tiny bit faster, but less repeatable
        torch.backends.cudnn.benchmark = True

    # Some hash-based ops in Python are random by default
    os.environ["PYTHONHASHSEED"] = str(seed)
