# Pode ser usado para disparar a extração inicial via terminal

from src.extract_embeddings import generate_embeddings

if __name__ == "__main__":
    generate_embeddings(
        image_folder="fashion-dataset/images",
        output_embedding_path="data/embeddings.npy",
        output_filenames_path="data/filenames.pkl",
        limit=2000
    )
