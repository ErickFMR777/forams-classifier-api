# forams-classifier-api
Clasificador automatizado de 4 géneros de foraminíferos bentónicos (Ammonia, Bolivina, Cibicides, Elphidium)

## Streamlit Cloud (inferencia directa)
La app de Streamlit usa inferencia directa con el modelo (no requiere FastAPI).

### Requisitos
- Python 3.10+
- Git LFS para el archivo del modelo

### Modelo
- Coloca el archivo en `models/forams_resnet18.pt` o `models/forams_resnet18.pt.pth.zip`.
- El modelo se rastrea con Git LFS (ver `.gitattributes`).

### Despliegue
1. Sube el repo a GitHub (publico).
2. En Streamlit Cloud, crea una app nueva apuntando a este repo.
3. Usa `app_streamlit.py` como entrypoint.

### Local
```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```
