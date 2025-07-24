
# 🖼️ Sistema de Recomendação por Imagens

Este projeto utiliza Deep Learning para recomendar produtos visualmente semelhantes com base em uma imagem de consulta.

---

## 🔍 Funcionalidades

- Upload de imagem pelo usuário
- Extração de características visuais com EfficientNetB0
- Recomendações de produtos mais similares usando Nearest Neighbors

---

## 🛠️ Tecnologias Utilizadas

- [Python](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/) (EfficientNetB0)
- [Streamlit](https://streamlit.io/) (interface web)
- [Scikit-learn](https://scikit-learn.org/) (busca por similaridade)
- NumPy, Pillow, tqdm

---

## 📦 Notas

- Os arquivos `embeddings.npy` e `filenames.pkl` foram gerados previamente no Colab.
- O dataset original está em `fashion-dataset/images/`, e não deve ser commitado ao Git por conta do tamanho.
- Para rodar localmente, baixe o [Fashion Product Images - Small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) e extraia na raiz do projeto com o nome `fashion-dataset`.

---

## 🗂️ Estrutura do Projeto

```
image-recommender/
├── app/
│   └── app.py                 # Interface Streamlit
├── fashion-dataset/           # Dataset de imagens
├── data/ 
│   ├── embeddings.npy         # Vetores de características extraídos
│   └── filenames.pkl          # Caminhos das imagens
├── src/
│   ├── extract_embeddings.py  # Função de extração de embeddings
│   ├── recommend.py           # Sistema de recomendação
│   └── utils.py               # Funções auxiliares
├── main.py                    # Execução via linha de comando
├── requirements.txt           # Dependências
├── README.md
```

---

## 🚀 Executando localmente

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

🎯 Projeto desenvolvido como parte do desafio do Bootcamp DIO — Sistema de Recomendação por Similaridade de Imagens.
