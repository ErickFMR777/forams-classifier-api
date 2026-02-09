import torch
import torch.nn as nn
from torchvision import models

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = ["Ammonia", "Bolivina", "Cibicides", "Elphidium"]

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(
        torch.load("forams_resnet18.pt", map_location=device)
    )

    model.eval()
    model.to(device)

    return model, device, classes
