import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import logging
import argparse
import os
import ast
import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError, ClientError

# Logging setup for better tracking on SageMaker
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_client = boto3.client('s3')
# Helper function to safely parse lists from strings
def safe_eval(param_str):
    try:
        return ast.literal_eval(param_str)
    except (ValueError, SyntaxError):
        raise ValueError(f"Invalid parameter format: {param_str}")
        
# Function to check if a file exists in S3
def check_s3_file_exists(bucket, file_key):
    try:
        s3_client.head_object(Bucket=bucket, Key=file_key)
        logger.info(f"File exists: s3://{bucket}/{file_key}")
        return True
    except ClientError as e:
        logger.error(f"File not found: s3://{bucket}/{file_key} - {e}")
        return False
    
train_file_check = check_s3_file_exists("ec2-utilization-sagemaker-model", f"sagemaker/mobile_price_classification/sklearncontainer/X_train-V-1.csv")
if not train_file_check:
    raise FileNotFoundError(f"Training file X_train-V-1.csv not found in S3 path {args.train}")



# Model loading function for SageMaker
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model
# Model loading redict function for SageMaker    
def predict_fn(input_data, model):
    # Load the scaler and encoder (or you can modify the code to load them only once if needed)
    scaler = joblib.load(os.path.join(os.getenv("SM_MODEL_DIR"), "scaler.joblib"))
    encoder = joblib.load(os.path.join(os.getenv("SM_MODEL_DIR"), "label_encoder.joblib"))
    model_dir = os.getenv("SM_MODEL_DIR", "/opt/ml/model")
    logger.info(f"Checking files in: {model_dir}")
    logger.info(f"Files in model directory: {os.listdir(model_dir)}")
    logger.info(f"Loaded encoder classes: {encoder.classes_}")
    # Preprocess the input data: scale it
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    input_data_scaled = scaler.transform(input_data)

    # Make prediction using the trained model
    prediction = model.predict(input_data_scaled)
    logger.info(f"Raw prediction output: {prediction}")

    

    # Decode the predicted label back to the original label
    #predicted_label = encoder.inverse_transform(prediction)
    '''try:
        predicted_label = encoder.inverse_transform(prediction)
    except ValueError as e:
        logger.error(f"Decoding error — unseen label issue: {e}")
        logger.error(f"Prediction values: {prediction}")
        logger.error(f"Encoder classes: {encoder.classes_}")
        predicted_label = ["unknown"]'''

    return prediction

# Main script execution
if __name__ == "__main__":

    logger.info("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters passed via command-line arguments (for Random Forest)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)

    # Directories for model, train, test, etc.
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR")) 
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")) 
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST")) 
    parser.add_argument("--X-train-file", type=str, default="X_train-V-1.csv")
    parser.add_argument("--X-test-file", type=str, default="X_test-V-1.csv")
    parser.add_argument("--y-train-file", type=str, default="y_train-V-1.csv")
    parser.add_argument("--y-test-file", type=str, default="y_test-V-1.csv")
    

    args = parser.parse_args()

    # Parse parameters with safe_eval
    #n_estimators = safe_eval(args.n_estimators)
    #max_depth = safe_eval(args.max_depth)
    #min_samples_split = safe_eval(args.min_samples_split)
    #min_samples_leaf = safe_eval(args.min_samples_leaf)
    n_estimators = args.n_estimators
    max_depth = args.max_depth
    min_samples_split = args.min_samples_split
    min_samples_leaf = args.min_samples_leaf

    # Check versions for logging
    logger.info("SKLearn Version: %s", sklearn.__version__)
    logger.info("Joblib Version: %s", joblib.__version__)

    logger.info("[INFO] Reading data")
    # Safely load data
    
    try:
        X_train = pd.read_csv(os.path.join(args.train, args.X_train_file))
        y_train = pd.read_csv(os.path.join(args.train, args.y_train_file))
        X_test = pd.read_csv(os.path.join(args.test, args.X_test_file))
        y_test = pd.read_csv(os.path.join(args.test, args.y_test_file))
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Validate shapes of datasets
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Mismatch: X_train and y_train row counts are different!")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("Mismatch: X_test and y_test row counts are different!")

    # Define the param grid for GridSearchCV
    param_grid = {
        'n_estimators': [n_estimators],
        'max_depth': [max_depth],
        'min_samples_split': [min_samples_split],
        'min_samples_leaf': [min_samples_leaf]
    }

    logger.info("Data Shape:")
    logger.info("---- SHAPE OF TRAINING DATA (85%%) ---- %s", str(X_train.shape))
    logger.info("---- SHAPE OF TESTING DATA (15%%) ---- %s", str(X_test.shape))

    logger.info("Training RandomForest Model.....")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Scalar completed.......")
    # Encoding target
    #encoder = LabelEncoder()
    #y_train_encoded = encoder.fit_transform(y_train)
    #y_test_encoded = encoder.transform(y_test)
    y_train = y_train.ravel() if hasattr(y_train, 'ravel') else np.array(y_train).flatten()
    y_test = y_test.ravel() if hasattr(y_test, 'ravel') else np.array(y_test).flatten()
    # Fit and transform the encoder
    encoder = LabelEncoder()
    encoder.fit(np.concatenate([y_train, y_test]))  # Fit on both train and test labels

    # Now transform separately
    y_train_encoded = encoder.transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    logger.info(f"y_train unique labels: {set(y_train)}")
    logger.info(f"y_test unique labels: {set(y_test)}")
    logger.info(f"Encoder classes: {encoder.classes_}")
    logger.info(f"Test labels not in encoder: {set(y_test) - set(encoder.classes_)}")
    # Perform GridSearchCV with RandomForest
    #cv_folds = max(2, min(5, y_train.nunique())) # Handle small datasets with few unique labels
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Log best parameters found by GridSearchCV
    logger.info("Best Parameters: %s", grid_search.best_params_)

    # Get the best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Save the model to the specified directory
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, os.path.join(args.model_dir, "scaler.joblib"))
    joblib.dump(encoder, os.path.join(args.model_dir, "label_encoder.joblib"))
    

    logger.info("Model, scaler, and encoder saved.")
    logger.info("Model persisted at %s", model_path)

    # Predictions and evaluation
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log evaluation metrics
    logger.info(f"Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"F1 Score: {f1:.2f}")
    try:
        roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        logger.info(f"ROC AUC: {roc_auc:.2f}")
    except ValueError:
        logger.warning("ROC AUC unavailable — possibly a multi-class problem")
