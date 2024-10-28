import os
from urllib.parse import quote_plus

# Use environment variables for sensitive information
username = os.getenv("MONGODB_USERNAME")
password = quote_plus(os.getenv("MONGODB_PASSWORD"))

# Construct URI with credentials
MONGODB_URL_KEY = f'mongodb+srv://{username}:{password}cluster0.gy0vf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'

# Database and collection names
DATABASE_NAME = "copyright_infringement"
COLLECTION_NAME = "books"

PIPELINE_NAME: str = "copyright_infringement"
ARTIFACT_DIR: str = "artifact"

DATA_PREPROCESSING_DIR: str = os.path.join(ARTIFACT_DIR, "data_preprocessing")
EMBEDDING_DIR: str = os.path.join(ARTIFACT_DIR, "embeddings")  # New embedding artifact directory