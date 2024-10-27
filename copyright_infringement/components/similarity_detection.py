import pickle
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple
import numpy as np
from transformers import BertTokenizer, BertModel
from copyright_infringement.entity.config_entity import SimilarityDetectionConfig,EmbeddingConfig
from copyright_infringement.entity.artifact_entity import SimilarityResult
from copyright_infringement.utils.main_utils import preprocess_text

class SimilarityDetection:
    def __init__(self, config: SimilarityDetectionConfig, embedding_config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(embedding_config.model_name)
        self.model = BertModel.from_pretrained(embedding_config.model_name)

    def load_embeddings(self, embedding_file_path: str, metadata_file_path: str):
        with open(r"artifact\20241028011733\embeddings\text_embeddings.pkl", 'rb') as f:
            self.embeddings = pickle.load(f)


        with open(r"artifact\20241028011733\embeddings\metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)

    def generate_embedding(self, text: str) -> np.ndarray:
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    def check_similarity(self, new_text: str) -> SimilarityResult:
        preprocessed_text = preprocess_text(new_text)
        new_embedding = self.generate_embedding(preprocessed_text)

        # Calculate cosine similarity with existing embeddings
        similarities = cosine_similarity([new_embedding], self.embeddings)[0]
        max_similarity_idx = np.argmax(similarities)
        max_similarity_score = similarities[max_similarity_idx]

        # Check if the max similarity score exceeds the threshold
        if max_similarity_score >= self.config.similarity_threshold:
            similar_text = self.metadata['title'][max_similarity_idx]
            return SimilarityResult(
                is_copyright_infringement=True,
                similarity_score=max_similarity_score,
                similar_text=similar_text
            )
        else:
            return SimilarityResult(
                is_copyright_infringement=False,
                similarity_score=max_similarity_score,
                similar_text=""
            )

def main():
    # Configuration and initialization
    embedding_config = EmbeddingConfig()
    similarity_config = SimilarityDetectionConfig(similarity_threshold=0.8)
    similarity_detector = SimilarityDetection(config=similarity_config, embedding_config=embedding_config)

    # Load precomputed embeddings and metadata
    similarity_detector.load_embeddings(embedding_config.embedding_file_path, embedding_config.metadata_file_path)

    # Accept user input for testing
    new_text = input("Enter text to check for copyright infringement: ")
    result = similarity_detector.check_similarity(new_text)

    # Display result
    if result.is_copyright_infringement:
        print(f"Potential copyright infringement detected!")
        print(f"Similarity Score: {result.similarity_score}")
        print(f"Similar Text: {result.similar_text}")
    else:
        print("No infringement detected.")
        print(f"Similarity Score: {result.similarity_score}")

if __name__ == "__main__":
    main()
