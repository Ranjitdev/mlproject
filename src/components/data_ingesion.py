import os
import sys
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTraier

@dataclass
class DataIngesionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')

class DataIngesion:
    def __init__(self):
        self.ingesion_config = DataIngesionConfig()

    def initiate_data_ingesion(self):
        logging.info('Entered the data Ingesion')
        try:
            df = pd.read_csv(r'notebook/data/stud.csv')
            logging.info('Loaded the dataset')

            os.makedirs(os.path.dirname(self.ingesion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingesion_config.raw_data_path, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=41)
            train_set.to_csv(self.ingesion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingesion_config.test_data_path, index=False, header=True)
            logging.info('Data ingesion completed')

            return (
                self.ingesion_config.train_data_path,
                self.ingesion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__=='__main__':
    train_dir, test_dir = DataIngesion().initiate_data_ingesion()
    train_arr, test_arr, process_pkl_path = DataTransformation().initiate_data_transformation(train_dir, test_dir)
    ModelTraier().initiate_model_training(train_arr, test_arr)
