import streamlit as st
from PIL import Image
from api import predict_image

st.set_page_config(
    page_title="Foraminifera Classifier",
    layout="centered"
)

st.title("🦠 Foraminifera Genus Classifier")

st.markdown("""
Clasificación automática de **géneros de foraminíferos**
a partir de imágenes **ópticas o SEM**.
""")

uploaded_file = st.file_uploader(
    "Sube una imagen (JPG o PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Imagen cargada",
        use_column_width=True
    )

    if st.button("Clasificar"):
        with st.spinner("Analizando imagen..."):
            genus = predict_image(image)

        st.success(f"🧬 Género predicho: **{genus}**")

st.markdown("---")
st.caption("Modelo entrenado con imágenes ópticas y SEM · Deep Learning")
