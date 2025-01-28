import pandas as pd
import os
from pymongo import MongoClient
from copyright_infringement.entity.config_entity import DataIngestionConfig
from copyright_infringement.entity.artifact_entity import DataIngestionArtifact
from urllib.parse import quote_plus
import logging
from dotenv import load_dotenv

load_dotenv()

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.client = MongoClient(
            "mongodb+srv://harsha:harsha123@cluster0.02fs1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        )

    def fetch_data_from_mongodb(self):
        try:
            logging.info(f"Fetching data from MongoDB collection {self.config.collection_name}")
            # Access database and collection
            db = self.client[self.config.database_name]
            collection = db[self.config.collection_name]
            # Fetch data and convert to DataFrame
            data = pd.DataFrame(list(collection.find()))
            logging.info("Data fetched from MongoDB successfully.")
            return data
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise

    def save_data_to_csv(self, data: pd.DataFrame) -> str:
        try:
            os.makedirs(os.path.dirname(self.config.csv_file_path), exist_ok=True)
            data.to_csv(self.config.csv_file_path, index=False)
            logging.info(f"Data saved as CSV at {self.config.csv_file_path}")
            return self.config.csv_file_path
        except Exception as e:
            logging.error(f"Error saving data to CSV: {e}")
            raise

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        # Fetch data from MongoDB
        data = self.fetch_data_from_mongodb()
        
        # Save data to CSV in artifacts
        csv_path = self.save_data_to_csv(data)
        
        # Return DataIngestionArtifact with the CSV path
        return DataIngestionArtifact(csv_file_path=csv_path)
