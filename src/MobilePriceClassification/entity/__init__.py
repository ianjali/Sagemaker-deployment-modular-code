from dataclasses import dataclass
from pathlib import Path
import boto3
import sagemaker
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
    