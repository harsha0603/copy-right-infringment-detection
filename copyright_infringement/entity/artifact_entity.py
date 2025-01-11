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

from dataclasses import dataclass
from typing import List

from dataclasses import dataclass
from typing import List

@dataclass
class ImageMetadata:
    """Metadata related to each image."""
    filename: str
    content_type: str
    embedding_file_path: str  # Path to the generated embedding
    raw_image_path: str  # Path to the raw image file

@dataclass
class ImageIngestionArtifact:
    """Artifact for the image ingestion process."""
    image_metadata: List[ImageMetadata]  # List of image metadata objects

