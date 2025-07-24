from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
import os
from tqdm import tqdm
import pickle

def load_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

def extract_embedding(model, image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = model.predict(img_array, verbose=0)
    return embedding.flatten()

def generate_embeddings(image_folder, output_embedding_path, output_filenames_path, limit=None):
    model = load_model()
    embeddings = []
    filenames = []
    image_files = os.listdir(image_folder)[:limit] if limit else os.listdir(image_folder)

    for file in tqdm(image_files):
        path = os.path.join(image_folder, file)
        try:
            emb = extract_embedding(model, path)
            embeddings.append(emb)
            filenames.append(path)
        except:
            continue

    np.save(output_embedding_path, np.array(embeddings))
    with open(output_filenames_path, 'wb') as f:
        pickle.dump(filenames, f)
