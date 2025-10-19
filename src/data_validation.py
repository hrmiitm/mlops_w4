"""
Data validation module for IRIS dataset.
Ensures data quality and integrity.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


def validate_iris_data():
    """
    Validates the IRIS dataset structure and content.
    
    Returns:
        bool: True if all validations pass
    
    Raises:
        AssertionError: If any validation fails
    """
    # Load data
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    # Check shape
    assert df.shape[0] == 150, f"Expected 150 rows, got {df.shape[0]}"
    assert df.shape[1] == 5, f"Expected 5 columns, got {df.shape[1]}"
    
    # Check columns
    expected_columns = ['sepal length (cm)', 'sepal width (cm)', 
                       'petal length (cm)', 'petal width (cm)', 'target']
    assert list(df.columns) == expected_columns, f"Column mismatch: {list(df.columns)}"
    
    # Check for missing values
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"
    
    # Check target values
    unique_targets = df['target'].unique()
    assert len(unique_targets) == 3, f"Expected 3 classes, got {len(unique_targets)}"
    assert set(unique_targets) == {0, 1, 2}, f"Unexpected target values: {unique_targets}"
    
    # Check data types
    for col in df.columns[:-1]:
        assert df[col].dtype in [np.float64, np.float32], f"Column {col} has wrong dtype"
    
    # Check value ranges (basic sanity checks)
    assert (df['sepal length (cm)'] > 0).all(), "Sepal length contains non-positive values"
    assert (df['sepal width (cm)'] > 0).all(), "Sepal width contains non-positive values"
    assert (df['petal length (cm)'] >= 0).all(), "Petal length contains negative values"
    assert (df['petal width (cm)'] >= 0).all(), "Petal width contains negative values"
    
    print("✓ All data validation checks passed!")
    return True


if __name__ == "__main__":
    validate_iris_data()
