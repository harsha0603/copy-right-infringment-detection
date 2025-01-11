import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from pathlib import Path

# Paths
RAW_IMAGES_PATH = "./artifacts/raw_images"
EMBEDDINGS_PATH = "./artifacts/embeddings"

# Ensure the embeddings folder exists
Path(EMBEDDINGS_PATH).mkdir(parents=True, exist_ok=True)

# Load pre-trained ResNet50 model (excluding top classification layer)
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

def preprocess_image(image_path):
    """
    Preprocesses an image for ResNet50.
    """
    image = load_img(image_path, target_size=(224, 224))  # Load and resize image
    img_array = img_to_array(image)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for ResNet50
    return img_array

def generate_embeddings():
    """
    Generate and save embeddings for all images in RAW_IMAGES_PATH.
    """
    processed_count = 0
    total_images = len(os.listdir(RAW_IMAGES_PATH))

    for file_name in os.listdir(RAW_IMAGES_PATH):
        file_path = os.path.join(RAW_IMAGES_PATH, file_name)

        # Skip non-image files
        if not (file_name.endswith(".jpg") or file_name.endswith(".png")):
            print(f"Skipping non-image file: {file_name}")
            continue

        # Check if embedding already exists
        embedding_file_path = os.path.join(EMBEDDINGS_PATH, f"{Path(file_name).stem}.npy")
        if os.path.exists(embedding_file_path):
            print(f"Embedding already exists for: {file_name}")
            continue

        try:
            # Preprocess image and generate embedding
            image_array = preprocess_image(file_path)
            embedding = model.predict(image_array)

            # Save embedding as .npy file
            np.save(embedding_file_path, embedding)
            print(f"Embedding saved for: {file_name}")
            processed_count += 1
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    print(f"\nProcessed {processed_count}/{total_images} images.")

if __name__ == "__main__":
    generate_embeddings()
