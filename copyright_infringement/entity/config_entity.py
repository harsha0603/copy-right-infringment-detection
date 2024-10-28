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
    similarity_threshold: float = 0.8  # Adjust based on your accuracy needs