from src.MobilePriceClassification.config.configuration import ConfigurationManager
from src.MobilePriceClassification.components.data_ingestion import DataIngestion
from src.MobilePriceClassification.logging import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    def initiate_data_ingestion(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion_config()
        data_ingestion=DataIngestion(config=data_ingestion_config)
        train_df,test_df = data_ingestion.split_data()
        data_ingestion.upload_to_s3()