from src.MobilePriceClassification.config.configuration import ConfigurationManager
from src.MobilePriceClassification.components.model_deploy import ModelDeploy
from src.MobilePriceClassification.logging import logger

class ModelDeploymentPipeline:
    def __init__(self):
        pass
    def initiate_model_deployment(self,artifact):
        config = ConfigurationManager()
        model_deployment_config = config.get_model_deployment_config()
        model_deploy=ModelDeploy(config=model_deployment_config)
        model = model_deploy.get_model(artifact)
        logger.info("Model deployment initiated")
        # print(model)
        model_deploy.deploy_endpoint(model)
        logger.info("deployed endpoint  successfully")