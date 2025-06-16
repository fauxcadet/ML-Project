import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from  dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'raw.csv')
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion Method")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read dataset as Data frame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test Split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e, sys) from e    
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
# This code is for the data ingestion component of a machine learning project.
# It reads a dataset, splits it into training and testing sets, and saves them to specified paths.
# The code also handles exceptions and logs the process for debugging purposes.
# The DataIngestion class is initialized with a configuration that specifies where to save the data.
# The initiate_data_ingestion method reads the dataset, splits it, and saves the resulting dataframes.
# The code is designed to be run as a standalone script, and it will execute the data ingestion process when run directly.