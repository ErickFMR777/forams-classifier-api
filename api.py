import io
import time
from collections import deque
from datetime import datetime, timezone

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision import transforms

from model import load_model

app = FastAPI(title="Forams Classifier API")

# Cargar modelo una sola vez
model, device, classes = load_model()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MIN_IMAGE_SIZE = (64, 64)

metrics = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "last_predictions": deque(maxlen=10),
}


def predict_image(image: Image.Image) -> tuple[str, float]:
    """
    image: PIL.Image (RGB)
    return: string (genero), float (probabilidad)
    """
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        pred = torch.argmax(probs, dim=0).item()
        prob = float(probs[pred].item())

    return classes[pred], prob


def record_error() -> None:
    metrics["requests_total"] += 1
    metrics["requests_error"] += 1


def record_success(
    genus: str,
    probability: float,
    duration_ms: float,
    width: int,
    height: int,
) -> None:
    metrics["requests_total"] += 1
    metrics["requests_success"] += 1
    metrics["last_predictions"].appendleft(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "genus": genus,
            "probability": round(probability * 100, 2),
            "latency_ms": round(duration_ms, 2),
            "width": width,
            "height": height,
        }
    )


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        record_error()
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        data = await file.read()
        if len(data) > MAX_IMAGE_BYTES:
            record_error()
            raise HTTPException(status_code=413, detail="Image too large")
        image = Image.open(io.BytesIO(data))
    except HTTPException:
        raise
    except Exception as exc:
        record_error()
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    if image.mode not in ("RGB", "RGBA", "L", "LA"):
        record_error()
        raise HTTPException(status_code=400, detail="Unsupported color mode")

    width, height = image.size
    if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
        record_error()
        raise HTTPException(status_code=400, detail="Image too small")

    image = image.convert("RGB")

    start = time.perf_counter()
    genus, probability = predict_image(image)
    duration_ms = (time.perf_counter() - start) * 1000
    record_success(genus, probability, duration_ms, width, height)
    return {"genus": genus, "probability": round(probability * 100, 2)}


@app.get("/metrics", response_class=HTMLResponse)
def metrics_dashboard() -> HTMLResponse:
    rows = []
    for item in metrics["last_predictions"]:
        rows.append(
            "<tr>"
            f"<td>{item['timestamp']}</td>"
            f"<td>{item['genus']}</td>"
            f"<td>{item['probability']}</td>"
            f"<td>{item['latency_ms']}</td>"
            f"<td>{item['width']}x{item['height']}</td>"
            "</tr>"
        )

    table = (
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<thead><tr><th>timestamp</th><th>genus</th><th>probability</th><th>latency_ms</th><th>size</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )

    html = (
        "<html><head><title>Forams Metrics</title></head><body>"
        "<h2>Forams Classifier Metrics</h2>"
        f"<p>total: {metrics['requests_total']} | success: {metrics['requests_success']} | error: {metrics['requests_error']}</p>"
        "<h3>Last 10 predictions</h3>"
        f"{table}"
        "</body></html>"
    )

    return HTMLResponse(content=html)
