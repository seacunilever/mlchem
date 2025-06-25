import pytest
import pandas as pd
from mlchem.ml.feature_selection.filters import collinearity_filter, diversity_filter

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'feature3': [5, 4, 3, 2, 1],
        'feature4': [1, 1, 1, 1, 1],
        'target': [1, 0, 1, 0, 1],
        })
    return data

def test_collinearity_filter(sample_data):
    # Test without target variable
    filtered_df = collinearity_filter(sample_data, threshold=0.9)
    assert 'feature1' in filtered_df.columns
    assert 'feature2' not in filtered_df.columns
    assert 'feature3' not in filtered_df.columns
    assert 'feature4' in filtered_df.columns

    # Test with target variable
    filtered_df = collinearity_filter(sample_data, threshold=0.9, target_variable='target')
    assert 'feature1' in filtered_df.columns
    assert 'feature2' not in filtered_df.columns
    assert 'feature3' not in filtered_df.columns
    assert 'feature4' not in filtered_df.columns
    assert 'target' in filtered_df.columns

def test_diversity_filter(sample_data):
    # Test without target variable
    filtered_df = diversity_filter(sample_data, threshold=0.8)
    assert 'feature1' in filtered_df.columns
    assert 'feature2' in filtered_df.columns
    assert 'feature3' in filtered_df.columns
    assert 'feature4' not in filtered_df.columns

    # Test with target variable
    filtered_df = diversity_filter(sample_data, threshold=0.8, target_variable='target')
    assert 'feature1' in filtered_df.columns
    assert 'feature2' in filtered_df.columns
    assert 'feature3' in filtered_df.columns
    assert 'feature4' not in filtered_df.columns
    assert 'target' in filtered_df.columns

if __name__ == "__main__":
    pytest.main()