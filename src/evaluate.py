"""
Evaluation module for IRIS classification model.
This module loads a trained model and evaluates it on test data.
"""
import os
import sys
import json
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
import joblib


def evaluate_model(model_path="models/iris_model.joblib", 
                   output_path="models/evaluation_report.json"):
    """
    Loads the trained model, evaluates it, and prints a report.
    
    Args:
        model_path (str): Path to the trained model file
        output_path (str): Path to save evaluation report
    
    Returns:
        tuple: (accuracy, report_dict)
    """
    # Load model
    try:
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load data
    print("Loading IRIS dataset for evaluation...")
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Use the same split as training for consistent evaluation
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Print evaluation report
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("="*50 + "\n")
    
    # Save evaluation report
    evaluation_data = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(evaluation_data, f, indent=4)
    
    print(f"Evaluation report saved to {output_path}")
    
    return accuracy, report


if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists("models/iris_model.joblib"):
        print("Model not found. Please train the model first.")
        print("Run: python src/train.py")
        sys.exit(1)
    
    evaluate_model()
