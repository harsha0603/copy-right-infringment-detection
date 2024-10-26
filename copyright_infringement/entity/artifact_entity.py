from dataclasses import dataclass

# Artifact for Data Ingestion
@dataclass
class DataIngestionArtifact:
    csv_file_path: str  # Path where the CSV file is stored
