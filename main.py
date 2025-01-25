from src.MobilePriceClassification.logging import logger
from pathlib import Path
from dotenv import load_dotenv
env_path = Path('.env')
load_dotenv(dotenv_path=env_path)
from src.MobilePriceClassification.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.MobilePriceClassification.pipeline.stage_2_model_training import ModelTrainerPipeline  
from src.MobilePriceClassification.pipeline.stage_3_model_deployment_pipeline import ModelDeploymentPipeline  
from src.MobilePriceClassification.pipeline.stage_4_model_inferencing_pipeline import ModelInferencePipeline

# logger.info("Logging is implemented successfully")
delete_endpoint = False
STAGE_NAME="Data Ingestion stage"
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    # data_ingestion_pipeline=DataIngestionTrainingPipeline()
    # data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(f"stage {STAGE_NAME} failed : {str(e)}")
    raise e

STAGE_NAME="Model Training stage"
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    model_training_pipeline=ModelTrainerPipeline()
    artifact = model_training_pipeline.initiate_model_training()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(f"stage {STAGE_NAME} failed : {str(e)}")
    raise e
logger.info(f"Model artifact saved at: {artifact}")
STAGE_NAME="Model Deployment stage"
#deploy model and create an endpoint
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    model_deployment_pipeline=ModelDeploymentPipeline()
    model_deployment_pipeline.initiate_model_deployment(artifact)
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(f"stage {STAGE_NAME} failed : {str(e)}")
    raise e

STAGE_NAME="Model Inferencing stage"
#deploy model and create an endpoint
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    model_inferencing_pipeline=ModelInferencePipeline()
    model_inferencing_pipeline.initiate_model_inferencing()
    if delete_endpoint:
        model_inferencing_pipeline.initiate_model_inferencing(delete_endpoint)
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(f"stage {STAGE_NAME} failed : {str(e)}")
    raise e

