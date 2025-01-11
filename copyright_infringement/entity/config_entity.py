import os
from dataclasses import dataclass
from datetime import datetime
from copyright_infringement.constants import *

# Timestamp for unique artifact directory creation
TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP
    
    def get_component_artifact_dir(self, component_name: str) -> str:
        """Dynamically generates and returns a directory path for a specific component."""
        return os.path.join(self.artifact_dir, component_name)

# Initialize training pipeline configuration
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

# Configuration for Data Ingestion, referencing dynamic component directory
@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = training_pipeline_config.get_component_artifact_dir("data_ingestion_books")
    csv_file_path: str = os.path.join(data_ingestion_dir, "books_data.csv")
    database_name: str = "copyright_infringement"
    collection_name: str = "books"

@dataclass
class DataPreprocessingConfig:
    data_preprocessing_dir: str = os.path.join(ARTIFACT_DIR, "data_preprocessing")
    preprocessed_data_path: str = os.path.join(data_preprocessing_dir, "preprocessed_data.csv")
    
    def get_preprocessing_artifact_dir(self) -> str:
        """Returns the directory for storing preprocessed artifacts."""
        return self.data_preprocessing_dir
    
@dataclass
class PreprocessingParams:
    remove_duplicates: bool = True  # Remove duplicate entries
    lower_case: bool = True          # Convert text to lower case
    remove_stopwords: bool = True    # Remove common stopwords
    stemming: bool = False           # Apply stemming
    lemmatization: bool = False      # Apply lemmatization
    max_length: int = 512            # Max length of sequences for BERT input
    remove_special_chars: bool = True # Remove special characters

# Configuration for Embedding Generation
import os
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    embedding_dir: str = os.path.join(ARTIFACT_DIR, "embeddings")  # Set to a static embeddings directory
    model_name: str = "bert-base-uncased"  # Model name for generating embeddings
    batch_size: int = 16  # Batch size for embedding generation
    embedding_file_path: str = os.path.join(embedding_dir, "text_embeddings.pkl")
    metadata_file_path: str = os.path.join(embedding_dir, "metadata.pkl")  # Metadata (author/title) for reference
    max_length: int = 512


# Create the embeddings directory if it does not exist
# This must be done after defining the EmbeddingConfig class
os.makedirs(os.path.join(ARTIFACT_DIR, "embeddings"), exist_ok=True)

# Initialize the embedding configuration
embedding_config: EmbeddingConfig = EmbeddingConfig()



@dataclass
class SimilarityDetectionConfig:
    similarity_threshold: float = 0.8  


import os
from dataclasses import dataclass
from copyright_infringement.constants import *

import os
from dataclasses import dataclass
from copyright_infringement.constants import *

# image_ingestion_config.py

import os
from dataclasses import dataclass
from copyright_infringement.constants import *

from urllib.parse import quote_plus
import os
from dataclasses import dataclass

from urllib.parse import quote_plus

@dataclass
class ImageIngestionConfig:
    """Configuration for Image Data Ingestion"""
    
    # MongoDB credentials
    mongodb_username: str = "harshamassss"  # Replace with your actual username
    mongodb_password: str = "hs113LWAhjZUBFSq"  # Replace with your actual password

    # URL-encode the username and password
    encoded_username = quote_plus(mongodb_username)
    encoded_password = quote_plus(mongodb_password)
    
    # Format the MongoDB URL with encoded credentials
    mongodb_url: str = f"mongodb+srv://{encoded_username}:{encoded_password}@cluster0.02fs1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    
    database_name: str = "copyright_ingringement"  # Replace this with the actual database name

    def __init__(self, mongodb_url: str = None, database_name: str = None, raw_image_path: str = None, embeddings_path: str = None):
        self.mongodb_url = mongodb_url if mongodb_url else self.mongodb_url
        self.database_name = database_name if database_name else self.database_name
        self.raw_image_dir = raw_image_path if raw_image_path else os.path.join(ARTIFACT_DIR, "raw_images")
        self.embeddings_dir = embeddings_path if embeddings_path else os.path.join(ARTIFACT_DIR, "image_embeddings")

    def get_raw_image_path(self):
        """Returns the path to store raw images."""
        return self.raw_image_dir
    
    def get_embeddings_path(self):
        """Returns the path to store image embeddings."""
        return self.embeddings_dir
