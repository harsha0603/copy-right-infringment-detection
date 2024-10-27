import os
import pickle
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from copyright_infringement.entity.config_entity import EmbeddingConfig, DataPreprocessingConfig
from copyright_infringement.entity.artifact_entity import EmbeddingArtifact
from copyright_infringement.constants import EMBEDDING_DIR
from tqdm import tqdm

class DataEmbeddings:
    def __init__(self, config: EmbeddingConfig, preprocessing_config: DataPreprocessingConfig):
        self.config = config
        self.preprocessing_config = preprocessing_config
        self.model = BertModel.from_pretrained(self.config.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.config.model_name)

        # Ensure the embedding directory exists
        os.makedirs(self.config.embedding_dir, exist_ok=True)

    def load_data(self, preprocessed_data_path: str) -> pd.DataFrame:
        """Loads the preprocessed CSV data."""
        return pd.read_csv(preprocessed_data_path)

    def generate_embeddings(self, text: str) -> torch.Tensor:
        """Generates BERT embeddings for the given text."""
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.config.max_length)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def process_and_save_embeddings(self, preprocessed_data_path: str) -> EmbeddingArtifact:
        """Processes text data, generates embeddings, and saves to a pickle file."""
        # Load preprocessed data
        df = self.load_data(preprocessed_data_path)
        
        # Initialize lists to store embeddings and metadata
        embeddings = []
        metadata = {'title': [], 'author': []}
        
        # Specify the column to generate embeddings for (e.g., 'description')
        embedding_column = 'extracted_text'  # Adjust based on your data
        
        # Generate embeddings for each entry in the DataFrame
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
            text = row[embedding_column]  # Use the specified column
            embedding = self.generate_embeddings(text)
            embeddings.append(embedding.cpu().numpy())
            
            # Save metadata information
            metadata['title'].append(row['title'])  # Adjust based on your DataFrame
            metadata['author'].append(row['author'])  # Adjust based on your DataFrame

        # Save embeddings and metadata as pickle files
        with open(self.config.embedding_file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        with open(self.config.metadata_file_path, 'wb') as f:
            pickle.dump(metadata, f)

        print("Embeddings and metadata saved successfully.")

        # Return embedding artifact
        return EmbeddingArtifact(
            embedding_file_path=self.config.embedding_file_path,
            metadata_file_path=self.config.metadata_file_path
        )
