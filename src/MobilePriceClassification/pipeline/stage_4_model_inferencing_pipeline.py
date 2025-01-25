from src.MobilePriceClassification.config.configuration import ConfigurationManager
from src.MobilePriceClassification.components.model_inference import ModelInference
from src.MobilePriceClassification.logging import logger

class ModelInferencePipeline:
    def __init__(self):
        pass
    def initiate_model_inferencing(self,delete_endpoint=False):
        config = ConfigurationManager()
        model_inference_config = config.get_model_inferencing_config()
        model_inference=ModelInference(config=model_inference_config)
        model_inference.get_predictor()
        logger.info("Got predictor successfully")
        # print(model)
        model_inference.predict()
        if delete_endpoint:
            model_inference.delete_sagemaker_endpoint()
        logger.info("successfully")