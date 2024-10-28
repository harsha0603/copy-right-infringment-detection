from dataclasses import dataclass

# Artifact for Data Ingestion
@dataclass
class DataIngestionArtifact:
    csv_file_path: str  # Path where the CSV file is stored

from dataclasses import dataclass

@dataclass
class DataPreprocessingArtifact:
    preprocessed_data_path: str   # Path where the preprocessed data is stored

# Artifact for Embedding Generation
@dataclass
class EmbeddingArtifact:
    embedding_file_path: str       # Path where text embeddings are saved
    metadata_file_path: str        # Path where metadata (authors/title) is saved


@dataclass
class SimilarityResult:
    is_copyright_infringement: bool
    similarity_score: float
    similar_text: str