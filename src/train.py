"""
Training module for IRIS classification model.
This module loads data, trains a RandomForest classifier, and saves the model.
"""
import os
import json
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import yaml


def load_params(params_path="params.yaml"):
    """Load parameters from params.yaml file."""
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        return params
    return {
        'train': {
            'test_size': 0.2,
            'random_state': 42,
            'n_estimators': 100
        }
    }


def train_model(model_output_path="models/iris_model.joblib", 
                metrics_output_path="models/metrics.json"):
    """
    Loads IRIS data, trains a RandomForestClassifier, and saves the model.
    
    Args:
        model_output_path (str): Path to save the trained model
        metrics_output_path (str): Path to save training metrics
    
    Returns:
        tuple: (model, accuracy)
    """
    # Load parameters
    params = load_params()
    train_params = params.get('train', {})
    
    # Load IRIS dataset
    print("Loading IRIS dataset...")
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Split data
    test_size = train_params.get('test_size', 0.2)
    random_state = train_params.get('random_state', 42)
    n_estimators = train_params.get('n_estimators', 100)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training RandomForest with {n_estimators} estimators...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Calculate train accuracy
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")
    
    # Save metrics
    metrics = {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'n_estimators': n_estimators,
        'test_size': test_size,
        'random_state': random_state
    }
    
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy: {test_accuracy:.4f}")
    print(f"Metrics saved to {metrics_output_path}")
    
    return model, test_accuracy


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train_model()
