import sys
from copyright_infringement.exception import CopyRight
from copyright_infringement.logger import logging

from copyright_infringement.components.data_ingestion import DataIngestion
from copyright_infringement.components.data_preprocessing import DataPreprocessing
from copyright_infringement.components.text_embeddings import DataEmbeddings

from copyright_infringement.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    EmbeddingConfig,
    PreprocessingParams
)
from copyright_infringement.entity.artifact_entity import (
    DataIngestionArtifact,
    DataPreprocessingArtifact,
    EmbeddingArtifact
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.preprocessing_params = PreprocessingParams()
        self.text_embeddings_config = EmbeddingConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion...")
            data_ingestion = DataIngestion(config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed successfully.")
            return data_ingestion_artifact
        except Exception as e:
            raise CopyRight(e, sys) from e

    def start_data_preprocessing(self, csv_file_path: str) -> DataPreprocessingArtifact:
        try:
            logging.info("Starting data preprocessing...")
            data_preprocessing = DataPreprocessing(
                config=self.data_preprocessing_config,
                params=self.preprocessing_params
            )
            artifact = data_preprocessing.initiate_data_preprocessing(csv_file_path)
            logging.info("Data preprocessing completed successfully.")
            return artifact
        except Exception as e:
            raise CopyRight(e, sys)

    def start_text_embeddings(self, preprocessed_data_path: str) -> EmbeddingArtifact:
        try:
            logging.info("Starting text embeddings...")
            data_embeddings = DataEmbeddings(config=self.text_embeddings_config, 
                                             preprocessing_config=self.data_preprocessing_config)
            embedding_artifact = data_embeddings.process_and_save_embeddings(preprocessed_data_path)
            logging.info("Text embeddings generated and saved successfully.")
            return embedding_artifact
        except Exception as e:
            raise CopyRight(e, sys)

    def run_pipeline(self) -> EmbeddingArtifact:
        """
        Runs the complete pipeline by sequentially calling data ingestion, preprocessing,
        and embeddings generation.
        """
        try:
            logging.info("Starting the training pipeline...")

            # Start data ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            csv_file_path = data_ingestion_artifact.csv_file_path

            # Start data preprocessing
            data_preprocessing_artifact = self.start_data_preprocessing(csv_file_path)
            preprocessed_data_path = data_preprocessing_artifact.preprocessed_data_path

            # Start embedding generation
            embedding_artifact = self.start_text_embeddings(preprocessed_data_path)

            logging.info("Training pipeline completed successfully.")
            return embedding_artifact

        except Exception as e:
            raise CopyRight(e, sys)



