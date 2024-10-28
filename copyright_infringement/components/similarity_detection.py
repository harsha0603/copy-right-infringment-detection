import pickle
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple
import numpy as np
from transformers import BertTokenizer, BertModel
from copyright_infringement.entity.config_entity import SimilarityDetectionConfig, EmbeddingConfig
from copyright_infringement.entity.artifact_entity import SimilarityResult
import logging

class SimilarityDetection:
    def __init__(self, config: SimilarityDetectionConfig, embedding_config: EmbeddingConfig):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(embedding_config.model_name)
        self.model = BertModel.from_pretrained(embedding_config.model_name)
        self.embeddings, self.metadata = self.load_embeddings(embedding_config.embedding_file_path, embedding_config.metadata_file_path)

    def load_embeddings(self, embedding_file_path: str, metadata_file_path: str) -> Tuple[np.ndarray, dict]:
        """Load precomputed embeddings and metadata for similarity comparison."""
        with open(embedding_file_path, 'rb') as f:
            embeddings = pickle.load(f)

        # Convert to NumPy array if it's a list
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        with open(metadata_file_path, 'rb') as f:
            metadata = pickle.load(f)

        # Log the shape of the loaded embeddings
        logging.info(f"Loaded embeddings of shape: {embeddings.shape}")

        return embeddings, metadata

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for the input text using a pre-trained BERT model."""
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    def check_similarity(self, processed_text: str) -> SimilarityResult:
        """Check similarity of the processed text against preloaded embeddings."""
        logging.info(f"Checking similarity for processed text: {processed_text}")
        
        # Generate embedding for the new text
        new_embedding = self.generate_embedding(processed_text)
        logging.info(f"Generated new embedding: {new_embedding}")

        # Calculate cosine similarity with existing embeddings
        similarities = cosine_similarity([new_embedding], self.embeddings)[0]
        logging.info(f"Computed similarities: {similarities}")

        max_similarity_idx = np.argmax(similarities)
        max_similarity_score = similarities[max_similarity_idx]
        logging.info(f"Max similarity score: {max_similarity_score}, Index: {max_similarity_idx}")

        # Check if the max similarity score exceeds the threshold
        if max_similarity_score >= self.config.similarity_threshold:
            similar_text = self.metadata['title'][max_similarity_idx]
            logging.info(f"Copyright infringement detected: {similar_text}")
            return SimilarityResult(
                is_copyright_infringement=True,
                similarity_score=max_similarity_score,
                similar_text=similar_text
            )
        else:
            logging.info("No copyright infringement detected.")
            return SimilarityResult(
                is_copyright_infringement=False,
                similarity_score=max_similarity_score,
                similar_text=""
            )
