import os
import urllib.request as request
import zipfile
import pandas as pd
from src.MobilePriceClassification.logging import logger
from src.MobilePriceClassification.entity import DataIngestionConfig 
from sklearn.model_selection import train_test_split
class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config

    def split_data(self,  test_size=0.2, random_state=42):
        df = pd.read_csv(self.config.input_file)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        train_df.to_csv(self.config.train_csv, index=False)
        test_df.to_csv(self.config.test_csv, index=False)
        return train_df, test_df

    def upload_to_s3(self):
        trainpath = self.config.sess.upload_data(
            path=self.config.train_csv, 
            bucket=self.config.s3_bucket, 
            key_prefix=f'{self.config.s3_prefix}/train'
        )
        logger.info(f"Training data uploaded to: {trainpath}")
        testpath = self.config.sess.upload_data(
            path=self.config.test_csv, 
            bucket=self.config.s3_bucket, 
            key_prefix=f'{self.config.s3_prefix}/test'
        )
        logger.info(f"Test data uploaded to: {testpath}")
        return trainpath, testpath