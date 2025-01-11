# image_data_ingestion.py

import os
import gridfs
import numpy as np
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
from copyright_infringement.constants import MONGODB_URL_KEY, DATABASE_NAME
from copyright_infringement.entity.config_entity import ImageIngestionConfig
from copyright_infringement.entity.artifact_entity import ImageIngestionArtifact, ImageMetadata

# Initialize the pre-trained ResNet50 model (without the final classification layer)
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

class ImageDataIngestion:
    def __init__(self, config: ImageIngestionConfig):
        try:
            # Connect to MongoDB
            self.client = MongoClient(config.mongodb_url)
            self.db = self.client[config.database_name]
            self.fs = gridfs.GridFS(self.db)
            print("Connected successfully to MongoDB!")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise

        # Set artifact directory for raw images and embeddings
        self.raw_image_path = config.get_raw_image_path()
        self.embeddings_path = config.get_embeddings_path()
        os.makedirs(self.raw_image_path, exist_ok=True)
        os.makedirs(self.embeddings_path, exist_ok=True)
        print(f"Directories set: Raw images - {self.raw_image_path}, Embeddings - {self.embeddings_path}")

    def preprocess_image(self, image_data):
        """Preprocess the image for the CNN model."""
        try:
            image = Image.open(BytesIO(image_data))  # Convert binary data to image
            image = image.resize((224, 224))  # Resizing image to fit the ResNet50 input
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension
            img_array = preprocess_input(img_array)  # Preprocess input for ResNet50
            return img_array
        except Exception as e:
            print(f"Error during image preprocessing: {e}")
            raise

    def generate_embeddings(self, img_array):
        """Generate embeddings using ResNet50 model."""
        try:
            features = model.predict(img_array)  # Get feature vector from ResNet50
            return features.flatten()  # Flatten the feature vector
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            raise

    def fetch_and_store_images(self):
        image_metadata_list = []

        for file_metadata in self.fs.find():
            try:
                # Access attributes of GridOut object
                filename = file_metadata.filename if hasattr(file_metadata, 'filename') else "unknown"
                content_type = file_metadata.contentType if hasattr(file_metadata, 'contentType') else "unknown"

                file_id = file_metadata._id  # File ID from MongoDB
                print(f"Processing file: {filename} with ID: {file_id}")

                # Fetch the binary image data from GridFS
                image_data = file_metadata.read()  # Read the data directly
                print(f"Fetched image data for {filename}, size: {len(image_data)} bytes")

                # Store the raw image in the artifacts folder
                image_path = os.path.join(self.raw_image_path, filename)
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                print(f"Saved raw image to: {image_path}")

                # Preprocess image
                img_array = self.preprocess_image(image_data)
                print(f"Image preprocessed for {filename}, shape: {img_array.shape}")

                # Generate embeddings from the image
                embeddings = self.generate_embeddings(img_array)
                print(f"Generated embeddings for {filename}, embedding shape: {embeddings.shape}")

                # Store the embeddings in the artifact folder
                embeddings_filename = os.path.join(self.embeddings_path, f"{filename}_embeddings.npy")
                np.save(embeddings_filename, embeddings)
                print(f"Saved embeddings to: {embeddings_filename}")

                # Create image metadata entity
                image_metadata = ImageMetadata(
                    filename=filename,
                    content_type=content_type,
                    embedding_file_path=embeddings_filename,
                    raw_image_path=image_path
                )

                # Add the metadata to the list
                image_metadata_list.append(image_metadata)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
        
        # Create the image ingestion artifact
        ingestion_artifact = ImageIngestionArtifact(image_metadata=image_metadata_list)

        print(f"Image ingestion completed. Processed {len(image_metadata_list)} files.")
        # Return the artifact containing metadata and embeddings info
        return ingestion_artifact



# Usage Example
from urllib.parse import quote_plus

if __name__ == "__main__":
    try:
        # MongoDB credentials
        mongodb_username = "USERNAME"  
        mongodb_password = "PASSWORD"  
        
        # URL-encode the username and password
        encoded_username = quote_plus(mongodb_username)
        encoded_password = quote_plus(mongodb_password)
        
        # Format the MongoDB URL with the encoded credentials
        mongodb_url = f"mongodb+srv://{encoded_username}:{encoded_password}@cluster0.02fs1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

        # Initialize config with the updated MongoDB URL
        image_ingestion_config = ImageIngestionConfig(
            mongodb_url=mongodb_url,  # Use the URL with encoded credentials
            database_name=DATABASE_NAME,  # Ensure you have the correct database name
            raw_image_path="./artifacts/raw_images",  # Optional: Provide your custom path if needed
            embeddings_path="./artifacts/embeddings"  # Optional: Provide your custom path if needed
        )

        # Initialize the ImageDataIngestion class
        image_ingestion = ImageDataIngestion(config=image_ingestion_config)

        # Fetch and store images
        ingestion_artifact = image_ingestion.fetch_and_store_images()
        print(f"Ingestion Artifact: {ingestion_artifact}")
        
    except Exception as e:
        print(f"Fatal error in image ingestion pipeline: {e}")
