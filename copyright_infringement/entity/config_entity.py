import os
from dataclasses import dataclass
from datetime import datetime
from copyright_infringement.constants import PIPELINE_NAME, ARTIFACT_DIR

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
