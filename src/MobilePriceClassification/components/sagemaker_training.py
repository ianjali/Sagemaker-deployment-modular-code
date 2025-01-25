import os
import argparse
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def parse_arguments():
    """
    Parse command-line arguments for the script.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Random Forest Classification Training Script")
    
    # Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in random forest")
    parser.add_argument("--random_state", type=int, default=0, help="Random seed for reproducibility")
    
    # Data and model directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"), help="Directory to save the model")
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"), help="Training data directory")
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"), help="Testing data directory")
    parser.add_argument("--train-file", type=str, default="train-V-1.csv", help="Training data filename")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv", help="Testing data filename")
    
    return parser.parse_known_args()[0]

def load_data(train_path, test_path, train_file, test_file):
    """
    Load training and testing data from CSV files.
    
    Args:
        train_path (str): Path to training data directory
        test_path (str): Path to testing data directory
        train_file (str): Training data filename
        test_file (str): Testing data filename
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, features, label
    """
    # Read CSV files
    train_df = pd.read_csv(os.path.join(train_path, train_file))
    test_df = pd.read_csv(os.path.join(test_path, test_file))
    
    # Extract features and label
    features = list(train_df.columns)
    label = features.pop(-1)
    
    # Separate features and labels
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]
    
    return X_train, X_test, y_train, y_test, features, label

def print_data_info(X_train, X_test, y_train, y_test, features, label):
    """
    Print information about the dataset.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Testing labels
        features (list): Feature column names
        label (str): Label column name
    """
    print("Column order: ", features)
    print("\nLabel column is: ", label)
    
    print("\nData Shape: ")
    print("\n---- SHAPE OF TRAINING DATA (85%) ----")
    print(X_train.shape)
    print(y_train.shape)
    
    print("\n---- SHAPE OF TESTING DATA (15%) ----")
    print(X_test.shape)
    print(y_test.shape)

def train_model(X_train, y_train, n_estimators, random_state):
    """
    Train a Random Forest Classifier.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        n_estimators (int): Number of trees in the forest
        random_state (int): Random seed for reproducibility
    
    Returns:
        RandomForestClassifier: Trained model
    """
    print("\nTraining RandomForest Model ....")
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state,
        verbose=2, 
        n_jobs=1
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, model_dir):
    """
    Save the trained model to a specified directory.
    
    Args:
        model (RandomForestClassifier): Trained model
        model_dir (str): Directory to save the model
    
    Returns:
        str: Full path to the saved model
    """
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model saved at " + model_path)
    return model_path

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and print metrics.
    
    Args:
        model (RandomForestClassifier): Trained model
        X_test (pd.DataFrame): Testing features
        y_test (pd.Series): Testing labels
    """
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)
    
    print("\n---- METRICS RESULTS FOR TESTING DATA ----")
    print("\nTotal Rows are: ", X_test.shape[0])
    print('[TESTING] Model Accuracy is: ', test_acc)
    print('[TESTING] Testing Report: ')
    print(test_rep)

def main():
    """
    Main function to orchestrate the entire machine learning workflow.
    """
    # Print library versions
    print("SKLearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load data
    X_train, X_test, y_train, y_test, features, label = load_data(
        args.train, args.test, args.train_file, args.test_file
    )
    
    # Print data information
    print_data_info(X_train, X_test, y_train, y_test, features, label)
    
    # Train model
    model = train_model(X_train, y_train, args.n_estimators, args.random_state)
    
    # Save model
    save_model(model, args.model_dir)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)

def model_fn(model_dir):
    """
    Load the saved model from a directory.
    
    Args:
        model_dir (str): Directory containing the saved model
    
    Returns:
        RandomForestClassifier: Loaded model
    """
    print(model_dir)
    clf= joblib.load(os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    main()