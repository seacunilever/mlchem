import pytest
import pandas as pd
import numpy as np
from mlchem.ml.preprocessing.feature_transformation import polynomial_expansion

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })

def test_polynomial_expansion_degree_2(sample_dataframe):
    degree = 2
    expanded_df = polynomial_expansion(sample_dataframe, degree)
    expected_columns = ['feature1', 'feature2', 'feature1^2', 'feature1 feature2', 'feature2^2']
    assert isinstance(expanded_df, pd.DataFrame)
    assert list(expanded_df.columns) == expected_columns
    assert expanded_df.shape[1] == len(expected_columns)

def test_polynomial_expansion_degree_3(sample_dataframe):
    degree = 3
    expanded_df = polynomial_expansion(sample_dataframe, degree)
    expected_columns = [
        'feature1', 'feature2', 'feature1^2', 'feature1 feature2', 'feature2^2',
        'feature1^3', 'feature1^2 feature2', 'feature1 feature2^2', 'feature2^3'
    ]
    assert isinstance(expanded_df, pd.DataFrame)
    assert list(expanded_df.columns) == expected_columns
    assert expanded_df.shape[1] == len(expected_columns)

def test_polynomial_expansion_empty_dataframe():
    empty_df = pd.DataFrame()
    degree = 2
    expanded_df = polynomial_expansion(empty_df, degree)
    assert expanded_df.empty

if __name__ == "__main__":
    pytest.main()