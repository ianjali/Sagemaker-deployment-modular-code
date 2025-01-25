
import os
from src.MobilePriceClassification.entity import ModelDeployConfig
from sagemaker.sklearn.model import SKLearnModel
from time import gmtime,strftime
from src.MobilePriceClassification.logging import logger
class ModelDeploy:
    def __init__(self, config: ModelDeployConfig):
        self.config = config
        
    def get_model(self,artifact):
        #Packaging trained model for deployment
        #Wraps trained artifacts with deployment configuration
        #Creates deployable model object
        model_name=self.config.model_deploy_name
        logger.info(f"Creating model {model_name} for deployment")
        model=SKLearnModel(
            name=model_name,
            model_data=artifact,
            role=self.config.role,
            entry_point=self.config.sagemaker_entry_point,
            framework_version=self.config.framework_version
        )
        logger.info(f"Creating model {model_name} for deployment successful")
        return model
    def deploy_endpoint(self,model):
        #Creates real-time inference endpoint
        endpoint_name=self.config.model_deploy_endpoint
        logger.info(f"Deploying model to endpoint {endpoint_name}")
        predictor=model.deploy(
            initial_instance_count=1,
            instance_type=self.config.instance_type,
            endpoint_name=endpoint_name
        )
        logger.info(f"Model deployed to endpoint {endpoint_name}")


