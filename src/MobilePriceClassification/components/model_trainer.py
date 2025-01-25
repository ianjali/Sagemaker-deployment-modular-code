
import os
from src.MobilePriceClassification.entity import ModelTrainerConfig
from sagemaker.sklearn.estimator import SKLearn
import sagemaker
from src.MobilePriceClassification.logging import logger
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.sklearn_estimator = None
        
    def create_sagemaker_entry_point(self):
        #training configuration
        self.sklearn_estimator=SKLearn(
            entry_point=self.config.sagemaker_entry_point,
            role=self.config.role,
            instance_count=1,
            instance_type=self.config.instance_type,
            framework_version=self.config.framework_version,
            base_job_name=self.config.base_job_name,
            hyperparameters={
                "n_estimators":self.config.n_estimators,
                "random_state":self.config.random_state
            },
            use_spot_instance=self.config.use_spot_instance,
            max_run=3600
        )
    def train(self):
        # launch training job, with asynchronous call
        logger.info("Training job started")
        self.sklearn_estimator.fit({"train": self.config.train_path, "test": self.config.test_path}, wait=True)
        #ensure training job is completed
        self.sklearn_estimator.latest_training_job.wait(logs="None")
        artifact = sagemaker.Session().describe_training_job(
                    TrainingJobName=self.sklearn_estimator.latest_training_job.name
                    )["ModelArtifacts"]["S3ModelArtifacts"]
        logger.info("Training job completed and model artifact saved")
        # print(f"Model artifact saved at: {artifact}")
        return artifact
    


