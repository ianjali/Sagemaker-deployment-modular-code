from dataclasses import dataclass
from pathlib import Path
import boto3
import sagemaker
import os
from src.MobilePriceClassification.logging import logger
@dataclass
class DataIngestionConfig:
    input_file: Path
    train_csv: Path
    test_csv: Path
    s3_bucket: str
    s3_prefix: str
    s3_client: boto3.client = None
    sess: sagemaker.Session = None

    def __post_init__(self):
        # Initialize boto3 client and SageMaker session if not provided
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        if self.sess is None:
            self.sess = sagemaker.Session()

@dataclass
class ModelTrainerConfig:
    framework_version: str
    sagemaker_entry_point: Path
    instance_type: str
    base_job_name: str
    n_estimators: int
    random_state: int
    use_spot_instance: bool
    train_path: str = None
    test_path: str = None
    role: str = None
    sm_boto3: boto3.client = None
    def __post_init__(self):
        # Initialize role and s3 bucket path if not provided
        if self.role is None:
            self.role = os.environ.get('AWS_SAGEMAKER_ROLE')
        if self.train_path is None:
            self.train_path = os.environ.get('S3_TRAIN_PATH')
        if self.test_path is None:
            self.test_path = os.environ.get('S3_TEST_PATH')
        if self.sm_boto3 is None:
            self.sm_boto3 = boto3.client('sagemaker')
            
@dataclass          
class ModelDeployConfig:
    instance_type:str
    framework_version: str
    sagemaker_entry_point:str
    model_deploy_endpoint:str
    model_deploy_name:str
    sm_boto3: sagemaker.Session = None
    artifact: str=None
    role: str = None
    def __post_init__(self):
        if self.sm_boto3 is None:
            self.sm_boto3 = sagemaker.Session()
        if self.artifact is None:
            self.artifact = os.environ.get('S3_MODEL')
        if self.role is None:
            self.role = os.environ.get('AWS_SAGEMAKER_ROLE')

@dataclass          
class ModelInferenceConfig:
    test_path:Path
    endpoint_name:str
    sm_boto3: boto3.client = None
    def __post_init__(self):
        if self.sm_boto3 is None:
            self.sm_boto3 = boto3.client('sagemaker')
    
    
    
