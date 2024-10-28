import pandas as pd
import os
import re
import logging
import string
from nltk.corpus import stopwords
from copyright_infringement.entity.config_entity import DataPreprocessingConfig
from copyright_infringement.entity.artifact_entity import DataPreprocessingArtifact
from copyright_infringement.entity.config_entity import PreprocessingParams

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig, params: PreprocessingParams):
        self.config = config
        self.params = params
        self.stop_words = set(stopwords.words('english'))  # Load English stop words

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from the given CSV file path."""
        try:
            logging.info(f"Loading data from {data_path}")
            data = pd.read_csv(data_path)
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def remove_special_characters(self, text: str) -> str:
        """Remove special characters from text."""
        test = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Keep only alphanumeric characters and spaces
        return text

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data based on the parameters set."""
        if self.params.remove_duplicates:
            data = data.drop_duplicates()
            logging.info("Removed duplicate entries.")

        if self.params.lower_case:
            data['description'] = data['description'].str.lower()
            logging.info("Converted text to lower case.")

        # Remove special characters
        data['description'] = data['description'].apply(self.remove_special_characters)
        logging.info("Removed special characters from text.")

        # Remove stop words
        if self.params.remove_stopwords:
            data['description'] = data['description'].apply(lambda x: ' '.join(
                word for word in x.split() if word not in self.stop_words))
            logging.info("Removed stop words from text.")

        # Additional preprocessing: Stemming or Lemmatization
        if self.params.stemming:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            data['description'] = data['description'].apply(lambda x: ' '.join(stemmer.stem(word) for word in x.split()))
            logging.info("Applied stemming to text.")

        if self.params.lemmatization:
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            data['description'] = data['description'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split()))
            logging.info("Applied lemmatization to text.")

        return data

    def save_preprocessed_data(self, data: pd.DataFrame) -> str:
        """Save the preprocessed data to a CSV file."""
        try:
            os.makedirs(os.path.dirname(self.config.preprocessed_data_path), exist_ok=True)
            data.to_csv(self.config.preprocessed_data_path, index=False)
            logging.info(f"Preprocessed data saved at {self.config.preprocessed_data_path}")
            return self.config.preprocessed_data_path
        except Exception as e:
            logging.error(f"Error saving preprocessed data: {e}")
            raise

    def initiate_data_preprocessing(self, data_path: str) -> DataPreprocessingArtifact:
        """Main method to initiate the data preprocessing."""
        data = self.load_data(data_path)
        preprocessed_data = self.preprocess_data(data)
        preprocessed_data_path = self.save_preprocessed_data(preprocessed_data)
        return DataPreprocessingArtifact(preprocessed_data_path=preprocessed_data_path)

