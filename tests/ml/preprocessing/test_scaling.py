import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from mlchem.ml.preprocessing.scaling import scale_df_standard, scale_df_minmax, scale_df_robust, transform_df

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [7, 8, 9]
    })

def test_scale_df_standard(sample_dataframe):
    scaled_df, scaler = scale_df_standard(sample_dataframe, last_columns_to_preserve=1)
    assert isinstance(scaled_df, pd.DataFrame)
    assert isinstance(scaler, StandardScaler)
    assert scaled_df.shape == (3, 2)
    assert 'feature3' not in scaled_df.columns

def test_scale_df_minmax(sample_dataframe):
    scaled_df, scaler = scale_df_minmax(sample_dataframe, last_columns_to_preserve=1)
    assert isinstance(scaled_df, pd.DataFrame)
    assert isinstance(scaler, MinMaxScaler)
    assert scaled_df.shape == (3, 2)
    assert 'feature3' not in scaled_df.columns

def test_scale_df_robust(sample_dataframe):
    scaled_df, scaler = scale_df_robust(sample_dataframe, last_columns_to_preserve=1)
    assert isinstance(scaled_df, pd.DataFrame)
    assert isinstance(scaler, RobustScaler)
    assert scaled_df.shape == (3, 2)
    assert 'feature3' not in scaled_df.columns

def test_transform_df_standard(sample_dataframe):
    _, scaler = scale_df_standard(sample_dataframe, last_columns_to_preserve=1)
    transformed_df, _ = transform_df(sample_dataframe, scaler, last_columns_to_preserve=1)
    assert isinstance(transformed_df, pd.DataFrame)
    assert transformed_df.shape == (3, 2)
    assert 'feature3' not in transformed_df.columns

def test_transform_df_minmax(sample_dataframe):
    _, scaler = scale_df_minmax(sample_dataframe, last_columns_to_preserve=1)
    transformed_df, _ = transform_df(sample_dataframe, scaler, last_columns_to_preserve=1)
    assert isinstance(transformed_df, pd.DataFrame)
    assert transformed_df.shape == (3, 2)
    assert 'feature3' not in transformed_df.columns

def test_transform_df_robust(sample_dataframe):
    _, scaler = scale_df_robust(sample_dataframe, last_columns_to_preserve=1)
    transformed_df, _ = transform_df(sample_dataframe, scaler, last_columns_to_preserve=1)
    assert isinstance(transformed_df, pd.DataFrame)
    assert transformed_df.shape == (3, 2)
    assert 'feature3' not in transformed_df.columns

if __name__ == "__main__":
    pytest.main()