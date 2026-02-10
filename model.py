import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

def load_model(weights_path: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = ["Ammonia", "Bolivina", "Cibicides", "Elphidium"]

    if weights_path is None:
        env_path = os.getenv("FORAMS_MODEL_PATH")
        if env_path:
            weights_path = env_path
        else:
            base_dir = Path(__file__).resolve().parent
            weights_path = base_dir / "models" / "forams_resnet18.pt"

    weights_path = Path(weights_path)
    if not weights_path.exists():
        fallback_path = weights_path.with_suffix(weights_path.suffix + ".pth.zip")
        if fallback_path.exists():
            weights_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. "
                "Set FORAMS_MODEL_PATH or place the file at models/forams_resnet18.pt"
            )

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "classes" in checkpoint and checkpoint["classes"]:
            classes = list(checkpoint["classes"])

        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    return model, device, classes
