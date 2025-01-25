### Complete Machine Learning Model Deployment Pipeline on AWS SageMaker 🌟

This repository provides an end-to-end pipeline for deploying machine learning models on AWS SageMaker, from data ingestion to model inferencing.

For a step-by-step understanding, refer to my Medium article [Deploying Machine Learning Models on Amazon SageMaker](https://medium.com/@mudgal.anjali.am/deploying-machine-learning-models-on-amazon-sagemaker-a-comprehensive-guide-adb72b3b95b0)
What’s Covered in This Pipeline? 🚀
1. **Data Ingestion** 📥
Upload dataset to Amazon S3 for training and testing.

3. **Model Training** 🔧
Train the model using SageMaker with algorithms like RandomForestClassifier. Training is done on cost-effective Spot Instances.

4. **Model Deployment** 🚀
Deploy the trained model for real-time inference via SageMaker endpoints.

6. **Model Inferencing** 🔍
Perform inference on new data by sending requests to the endpoint.

Getting Started 🚀
Clone this repository:
```bash
git clone https://github.com/yourusername/your-repository.git
```


Set up AWS CLI:
👉 [Getting Started with AWS CLI](https://medium.com/@mudgal.anjali.am/getting-started-with-aws-cli-your-complete-setup-guide-9d96a399e950)


Project Structure
your-repository/
│
├── config/                
│   └── config.yaml        # Configuration settings for the project
│
├── artifacts/             # Folder for raw and processed data
│   ├── dataset/
│   │   ├── complete_dataset.csv
│   │   ├── train-V-1.csv
│   │   └── test-V-1.csv
│
├── src/                   # Source code for model training and deployment
│   └── MobilePriceClassification/
│       ├── components/     # Scripts for different model components
│       │   ├── data_ingestion.py   # Data ingestion logic
│       │   ├── model_trainer.py    # Model training logic
│       │   ├── script.py           # Main training script
│       │   ├── model_deploy.py     # Model deployment logic
│       │   └── model_inference.py  # Model inference logic
│       │
│       ├── config/          # Configuration files for the components
│       │   └── configuration.py
│       │
│       ├── constants/       # Constant variables and enums
│       │   └── __init__.py
│       │
│       ├── entity/          # Entity files (such as data schema)
│       │   └── __init__.py
│       │
│       ├── logging/         # Logging utilities
│       │   └── __init__.py
│       │
│       ├── pipeline/        # Pipeline for model stages
│       │   ├── stage_1_data_ingestion_pipeline.py  # Data ingestion pipeline
│       │   ├── stage_2_model_training.py           # Model training pipeline
│       │   ├── stage_3_model_deployment_pipeline.py# Model deployment pipeline
│       │   └── stage_4_model_inferencing_pipeline.py # Model inference pipeline
│       │
│       └── utils/           # Utility functions
│           └── __init__.py
│
├── logs/                  # Logs for tracking model activities
│   └── continuos_logs.log # Logs file
│
├── main.py                # Main script to run the project
├── .env                   # Environment variables for the project
├── requirements.txt       # Dependencies for the project
├── README.md              # Project documentation
└── .gitignore             # Git ignore file





