"""
Unit tests for data validation.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from src.data_validation import validate_iris_data


def test_iris_data_shape():
    """Test IRIS dataset has correct shape."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    assert df.shape[0] == 150, f"Expected 150 rows, got {df.shape[0]}"
    assert df.shape[1] == 5, f"Expected 5 columns, got {df.shape[1]}"


def test_iris_data_columns():
    """Test IRIS dataset has correct columns."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    expected_columns = ['sepal length (cm)', 'sepal width (cm)', 
                       'petal length (cm)', 'petal width (cm)', 'target']
    
    assert list(df.columns) == expected_columns, \
        f"Column mismatch. Expected: {expected_columns}, Got: {list(df.columns)}"


def test_iris_no_missing_values():
    """Test IRIS dataset has no missing values."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    missing_count = df.isnull().sum().sum()
    assert missing_count == 0, f"Found {missing_count} missing values"


def test_iris_target_classes():
    """Test IRIS dataset has correct target classes."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    unique_targets = df['target'].unique()
    assert len(unique_targets) == 3, f"Expected 3 classes, got {len(unique_targets)}"
    assert set(unique_targets) == {0, 1, 2}, \
        f"Unexpected target values: {unique_targets}"


def test_iris_data_types():
    """Test IRIS dataset has correct data types."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    for col in df.columns[:-1]:  # All columns except target
        assert df[col].dtype in [np.float64, np.float32, np.int64], \
            f"Column {col} has unexpected dtype: {df[col].dtype}"


def test_iris_value_ranges():
    """Test IRIS dataset values are in reasonable ranges."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    # All measurements should be positive
    assert (df['sepal length (cm)'] > 0).all(), \
        "Sepal length contains non-positive values"
    assert (df['sepal width (cm)'] > 0).all(), \
        "Sepal width contains non-positive values"
    assert (df['petal length (cm)'] >= 0).all(), \
        "Petal length contains negative values"
    assert (df['petal width (cm)'] >= 0).all(), \
        "Petal width contains negative values"


def test_iris_class_distribution():
    """Test IRIS dataset has balanced class distribution."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    class_counts = df['target'].value_counts()
    
    # Each class should have exactly 50 samples
    for class_label in [0, 1, 2]:
        assert class_counts[class_label] == 50, \
            f"Class {class_label} has {class_counts[class_label]} samples, expected 50"


def test_validate_iris_data_function():
    """Test the validate_iris_data function."""
    # Should not raise any exceptions
    result = validate_iris_data()
    assert result is True, "Validation function should return True"


def test_feature_correlation():
    """Test that features have expected correlations."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    # Petal length and petal width should be highly correlated
    correlation = df['petal length (cm)'].corr(df['petal width (cm)'])
    assert correlation > 0.9, \
        f"Petal length and width correlation ({correlation:.2f}) is lower than expected"
