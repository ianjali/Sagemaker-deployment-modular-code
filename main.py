from src.MobilePriceClassification.logging import logger
from src.MobilePriceClassification.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline

# logger.info("Logging is implemented successfully")
STAGE_NAME="Data Ingestion stage"
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_ingestion_pipeline=DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(f"stage {STAGE_NAME} failed : {str(e)}")
    raise e
     