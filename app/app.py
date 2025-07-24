import streamlit as st
from PIL import Image
import numpy as np
from src.extract_embeddings import load_model, extract_embedding
from src.recommend import ImageRecommender

st.set_page_config(page_title="Image Recommender", layout="wide")
st.title("🔍 Sistema de Recomendação por Similaridade de Imagens")

# Upload da imagem
uploaded_file = st.file_uploader("Escolha uma imagem para buscar similares", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Imagem de Consulta", use_container_width=False, width=200)

    # Extrair embedding da imagem enviada
    model = load_model()
    query_emb = extract_embedding(model, uploaded_file)

    # Carregar recomendador
    recommender = ImageRecommender("data/embeddings.npy", "data/filenames.pkl")
    results = recommender.recommend(query_emb)

    # Mostrar resultados
    st.subheader("Imagens mais similares:")
    cols = st.columns(5)
    for i, path in enumerate(results[1:]):  # Ignorar a primeira (igual a consulta)
        with cols[i]:
            st.image(Image.open(path), caption=f"Similar {i+1}", use_container_width=False, width=200)
