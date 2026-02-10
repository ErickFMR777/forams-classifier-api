import csv
import io
from typing import Any

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import load_model

st.set_page_config(
    page_title="Foraminifera Classifier",
    layout="centered"
)

st.title("🦠 Foraminifera Genus Classifier")

st.markdown("""
Clasificación automática de **géneros de foraminíferos**
a partir de imágenes **ópticas o SEM**.
""")


@st.cache_resource
def get_inference() -> tuple[torch.nn.Module, torch.device, list[str]]:
    return load_model()


model, device, classes = get_inference()

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


def predict_image(image: Image.Image) -> tuple[str, float]:
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        pred = torch.argmax(probs, dim=0).item()
        prob = float(probs[pred].item())

    return classes[pred], prob

uploaded_files = st.file_uploader(
    "Sube imagenes (JPG o PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.caption(f"Imagenes seleccionadas: {len(uploaded_files)}")

    if st.button("Clasificar"):
        results: list[dict[str, Any]] = []
        thumbs: list[tuple[Image.Image, str]] = []

        with st.spinner("Analizando imagenes..."):
            for uploaded_file in uploaded_files:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                except Exception:
                    results.append(
                        {
                            "archivo": uploaded_file.name,
                            "genero": "",
                            "confianza": "",
                            "estado": "imagen invalida",
                        }
                    )
                    continue

                genus, probability = predict_image(image)
                results.append(
                    {
                        "archivo": uploaded_file.name,
                        "genero": genus,
                        "confianza": f"{round(probability * 100, 2)}%",
                        "estado": "ok" if genus else "sin resultado",
                    }
                )
                thumbs.append(
                    (image, f"{uploaded_file.name} | {genus} ({round(probability * 100, 2)}%)")
                )

        if results:
            st.subheader("Resultados")
            st.dataframe(results, use_container_width=True)

            if thumbs:
                st.subheader("Vista previa")
                cols = st.columns(3)
                for idx, (img, caption) in enumerate(thumbs):
                    with cols[idx % 3]:
                        st.image(img, caption=caption, width=220)

            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=["archivo", "genero", "confianza", "estado"])
            writer.writeheader()
            writer.writerows(results)

            st.download_button(
                "Descargar CSV",
                data=csv_buffer.getvalue(),
                file_name="resultados_forams.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption("Modelo entrenado con imágenes ópticas y SEM · Deep Learning")
