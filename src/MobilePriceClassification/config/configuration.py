from src.MobilePriceClassification.constants import *
from src.MobilePriceClassification.utils.common import read_yaml
from src.MobilePriceClassification.entity import DataIngestionConfig,ModelTrainerConfig,ModelDeployConfig,ModelInferenceConfig
class ConfigurationManager:
    def __init__(self, 
                 config_path=CONFIG_FILE_PATH):
        self.config = read_yaml(config_path)
    
    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        data_ingestion_config=DataIngestionConfig(
            input_file = config.input_file,
            s3_bucket = config.s3_bucket,
            s3_prefix = config.s3_prefix,
            train_csv= config.train_csv,
            test_csv= config.test_csv,
        )
        return data_ingestion_config
    
    def get_model_trainer_config(self)-> ModelTrainerConfig:
        config=self.config.model_trainer
        model_trainer_config=ModelTrainerConfig(
            framework_version = config.framework_version,
            sagemaker_entry_point = config.sagemaker_entry_point,
            instance_type= config.instance_type,
            base_job_name= config.base_job_name,
            n_estimators= config.n_estimators,
            random_state= config.random_state,
            use_spot_instance= config.use_spot_instance    
        )
        return model_trainer_config

    def get_model_deployment_config(self)-> ModelDeployConfig:
        config=self.config.model_deploy
        model_deploy_config=ModelDeployConfig(
            instance_type = config.instance_type, 
            framework_version= config.framework_version,
            sagemaker_entry_point = config.sagemaker_entry_point,
            model_deploy_endpoint=config.model_deploy_endpoint,
            model_deploy_name=config.model_deploy_name
        )
        return model_deploy_config
    
    def get_model_inferencing_config(self)-> ModelInferenceConfig:
        config=self.config.model_inference
        model_inference_config=ModelInferenceConfig(
            endpoint_name= config.endpoint_name,
            test_path= config.test_path
        )
        return model_inference_config
