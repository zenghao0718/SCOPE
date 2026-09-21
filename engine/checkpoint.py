from pathlib import Path
import torch


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)
