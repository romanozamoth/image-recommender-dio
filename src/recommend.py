import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

class ImageRecommender:
    def __init__(self, embeddings_path, filenames_path):
        self.embeddings = np.load(embeddings_path)
        with open(filenames_path, 'rb') as f:
            self.filenames = pickle.load(f)
        self.nn = NearestNeighbors(n_neighbors=6, metric='euclidean')
        self.nn.fit(self.embeddings)

    def recommend(self, query_embedding):
        distances, indices = self.nn.kneighbors([query_embedding])
        return [self.filenames[idx] for idx in indices[0]]
