from src.MobilePriceClassification.config.configuration import ConfigurationManager
from src.MobilePriceClassification.components.model_trainer import ModelTrainer
from src.MobilePriceClassification.logging import logger

class ModelTrainerPipeline:
    def __init__(self):
        pass
    def initiate_model_training(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer=ModelTrainer(config=model_trainer_config)
        model_trainer.create_sagemaker_entry_point()
        artifact = model_trainer.train()
        return artifact