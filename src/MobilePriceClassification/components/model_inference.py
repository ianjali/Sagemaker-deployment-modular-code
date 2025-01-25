
import os
from src.MobilePriceClassification.entity import ModelInferenceConfig
from time import gmtime,strftime
from sagemaker.predictor import Predictor
import pandas as pd
import json
from src.MobilePriceClassification.logging import logger
class ModelInference:
    def __init__(self, config: ModelInferenceConfig):
        self.config = config
        self.predictor = None
    def get_predictor(self):
        self.predictor = Predictor(endpoint_name=self.config.endpoint_name)
        self.predictor.content_type = "application/json"
    
    def predict(self):
        df=pd.read_csv(self.config.test_path)
        features=list(df.columns)
        label = features.pop(-1)
        x=df[features]
        y=df[label]

        input_data = x[features][:2].values.tolist()
        input_data_json = json.dumps(input_data)
        
        target = y[0:2].values.tolist()
        
        logger.info(f"Input data: {input_data}")
        result = self.predictor.predict(input_data_json)
        logger.info(f"Prediction output: {result}")
        logger.info(f"Target output: {target}")
    def delete_sagemaker_endpoint(self):
        self.config.sm_boto3.delete_endpoint(EndpointName=self.config.endpoint_name)
        logger.info("Endpoint deleted successfully")
        
        
        



