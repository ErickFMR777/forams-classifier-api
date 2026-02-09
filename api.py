import torch
from torchvision import transforms
from model import load_model

# Cargar modelo una sola vez
model, device, classes = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image):
    """
    image: PIL.Image (RGB)
    return: string (género)
    """
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        pred = torch.argmax(outputs, dim=1).item()

    return classes[pred]
