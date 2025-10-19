"""
Unit tests for model evaluation and performance.
"""
import pytest
import numpy as np
import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.train import train_model
from src.evaluate import evaluate_model


@pytest.fixture(scope="module", autouse=True)
def setup_model_for_tests():
    """Ensures a model is trained and available for tests."""
    model_path = "models/iris_model_for_test.joblib"
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print("\nTraining model for tests...")
    train_model(model_output_path=model_path)
    
    yield model_path
    
    # Cleanup (optional)
    # if os.path.exists(model_path):
    #     os.remove(model_path)


def test_model_file_exists(setup_model_for_tests):
    """Test if model file was created successfully."""
    model_path = setup_model_for_tests
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    assert os.path.getsize(model_path) > 0, "Model file is empty"


def test_model_accuracy_threshold(setup_model_for_tests):
    """Test if the model's accuracy meets a minimum threshold."""
    model_path = setup_model_for_tests
    
    # Load the data and split it
    iris = load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=['target'])
    y = df['target']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the trained model
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Define an acceptable accuracy threshold for the IRIS dataset
    ACCURACY_THRESHOLD = 0.90
    
    print(f"\nModel Accuracy in Test: {accuracy:.4f}")
    assert accuracy >= ACCURACY_THRESHOLD, \
        f"Model accuracy ({accuracy:.4f}) is below threshold ({ACCURACY_THRESHOLD})"


def test_model_prediction_integrity(setup_model_for_tests):
    """Test if the model predictions are of expected type and range."""
    model_path = setup_model_for_tests
    
    iris = load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=['target'])
    y = df['target']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    
    # Check prediction count
    assert len(predictions) == len(y_test), "Prediction count mismatch"
    
    # Check prediction data types
    assert all(isinstance(p, (int, np.integer)) for p in predictions), \
        "Predictions are not integers"
    
    # Check prediction values are valid classes
    assert all(p in [0, 1, 2] for p in predictions), \
        "Predictions contain unexpected target values"


def test_model_probability_outputs(setup_model_for_tests):
    """Test if the model outputs valid probability distributions."""
    model_path = setup_model_for_tests
    
    iris = load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=['target'])
    
    _, X_test, _, _ = train_test_split(X, df['target'], test_size=0.2, random_state=42)
    
    model = joblib.load(model_path)
    probabilities = model.predict_proba(X_test)
    
    # Check shape
    assert probabilities.shape == (len(X_test), 3), "Probability shape mismatch"
    
    # Check probabilities sum to 1
    prob_sums = probabilities.sum(axis=1)
    assert np.allclose(prob_sums, 1.0), "Probabilities do not sum to 1"
    
    # Check probabilities are in valid range
    assert (probabilities >= 0).all() and (probabilities <= 1).all(), \
        "Probabilities out of valid range [0, 1]"


def test_evaluate_model_function(setup_model_for_tests):
    """Test the evaluate_model function."""
    model_path = setup_model_for_tests
    output_path = "models/test_evaluation_report.json"
    
    accuracy, report = evaluate_model(model_path=model_path, output_path=output_path)
    
    # Check accuracy is valid
    assert 0 <= accuracy <= 1, "Accuracy out of valid range"
    
    # Check report structure
    assert isinstance(report, dict), "Report is not a dictionary"
    assert 'accuracy' in report, "Report missing accuracy key"
    
    # Cleanup
    if os.path.exists(output_path):
        os.remove(output_path)
