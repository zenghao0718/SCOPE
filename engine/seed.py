import os
import random
import numpy as np
import torch


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def epoch_order(n, seed, epoch):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 100000 * epoch)
    return torch.randperm(n, generator=generator).tolist()
