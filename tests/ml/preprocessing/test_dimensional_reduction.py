import pytest
import pandas as pd
import numpy as np
from mlchem.ml.preprocessing.dimensional_reduction import Compressor

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])

@pytest.fixture
def compressor(sample_dataframe):
    return Compressor(dataframe=sample_dataframe)

def test_compress_PCA_variance(compressor):
    compressor.compress_PCA(n_components_or_variance=0.8)
    assert hasattr(compressor, 'X_compressed')
    assert hasattr(compressor, 'dataframe_compressed')
    assert compressor.dataframe_compressed.shape[1] <= 10

def test_compress_PCA_components(compressor):
    compressor.compress_PCA(n_components_or_variance=3)
    assert hasattr(compressor, 'X_compressed')
    assert hasattr(compressor, 'dataframe_compressed')
    assert compressor.dataframe_compressed.shape[1] == 3

def test_compress_TSNE(compressor):
    compressor.compress_TSNE(n_components=2)
    assert hasattr(compressor, 'X_compressed')
    assert hasattr(compressor, 'dataframe_compressed')
    assert compressor.dataframe_compressed.shape[1] == 2

def test_compress_SE(compressor):
    compressor.compress_SE(n_components=2, neighbours_number_or_fraction=5)
    assert hasattr(compressor, 'X_compressed')
    assert hasattr(compressor, 'dataframe_compressed')
    assert compressor.dataframe_compressed.shape[1] == 2

def test_compress_UMAP(compressor):
    compressor.compress_UMAP(n_components=2, neighbours_number_or_fraction=5)
    assert hasattr(compressor, 'X_compressed')
    assert hasattr(compressor, 'dataframe_compressed')
    assert compressor.dataframe_compressed.shape[1] == 2

def test_compress_MDS(compressor):
    compressor.compress_MDS(n_components=2)
    assert hasattr(compressor, 'X_compressed')
    assert hasattr(compressor, 'dataframe_compressed')
    assert compressor.dataframe_compressed.shape[1] == 2

def test_compress_LLE(compressor):
    compressor.compress_LLE(n_components=2, neighbours_number_or_fraction=5)
    assert hasattr(compressor, 'X_compressed')
    assert hasattr(compressor, 'dataframe_compressed')
    assert compressor.dataframe_compressed.shape[1] == 2

def test_compress_ISOMAP(compressor):
    compressor.compress_ISOMAP(n_components=2, neighbours_number_or_fraction=5)
    assert hasattr(compressor, 'X_compressed')
    assert hasattr(compressor, 'dataframe_compressed')
    assert compressor.dataframe_compressed.shape[1] == 2

if __name__ == "__main__":
    pytest.main()