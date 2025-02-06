import os
from dotenv import load_dotenv

load_dotenv()
# Use environment variables for sensitive information
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")

# Construct URI with credentials
MONGODB_URL_KEY = f'mongodb+srv://{username}:{password}cluster0.gy0vf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'

# Database and collection names
DATABASE_NAME = "copyright_infringement"
COLLECTION_NAME = "books"

PIPELINE_NAME: str = "copyright_infringement"
ARTIFACT_DIR: str = "artifact"

DATA_PREPROCESSING_DIR: str = os.path.join(ARTIFACT_DIR, "data_preprocessing")
EMBEDDING_DIR: str = os.path.join(ARTIFACT_DIR, "embeddings")  # New embedding artifact directory


RAW_IMAGE_DIR: str = os.path.join(ARTIFACT_DIR, "raw_images")  # Folder for storing raw images
EMBEDDING_DIR: str = os.path.join(ARTIFACT_DIR, "image_embeddings")  # Folder for storing image embeddings

# Ensure these directories exist
os.makedirs(RAW_IMAGE_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)
