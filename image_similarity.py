import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
RAW_IMAGES_PATH = "./artifacts/raw_images"
EMBEDDINGS_PATH = "./artifacts/embeddings"

# Load pre-trained ResNet50 model (excluding top classification layer)
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def preprocess_image(image_path):
    """
    Preprocesses an image for ResNet50.
    """
    image = load_img(image_path, target_size=(224, 224))  # Load and resize image
    img_array = img_to_array(image)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for ResNet50
    return img_array

def get_embedding(image_path):
    """
    Generate embedding for a single image.
    """
    image_array = preprocess_image(image_path)
    embedding = base_model.predict(image_array)
    return embedding

def check_infringement(input_image_path, threshold=0.85):
    """
    Check if the input image matches any stored embeddings and return the image(s) with the highest similarity.
    """
    input_embedding = get_embedding(input_image_path)

    # Initialize variables to store maximum similarity and corresponding file(s)
    max_similarity = 0
    most_similar_files = []

    # Load all stored embeddings and calculate similarity
    for embedding_file in os.listdir(EMBEDDINGS_PATH):
        embedding_path = os.path.join(EMBEDDINGS_PATH, embedding_file)

        # Load stored embedding
        stored_embedding = np.load(embedding_path)

        # Calculate cosine similarity
        similarity = cosine_similarity(input_embedding, stored_embedding)[0][0]

        # Update max similarity and reset the list if a higher similarity is found
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_files = [embedding_file]
        elif similarity == max_similarity:  # Handle ties
            most_similar_files.append(embedding_file)

    # Check if the highest similarity exceeds the threshold
    if max_similarity >= threshold:
        print(f"Potential copyright infringement detected!")
        print(f"Highest similarity: {max_similarity}")
        print(f"Matching files: {most_similar_files}")
        return most_similar_files, max_similarity
    else:
        print("No infringement detected. Image is unique.")
        return None, 0


if __name__ == "__main__":
    input_image_path = r"test_images\atria.png"
    most_similar_files, max_similarity = check_infringement(input_image_path, threshold=0.85)

    if most_similar_files:
        print(f"The most similar files are: {most_similar_files} with similarity {max_similarity}")
    else:
        print("No similar files found.")
