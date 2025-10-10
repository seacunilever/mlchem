# mlchem - cheminformatics library
# Copyright © 2025 as Unilever Global IP Limited

# Redistribution and use in source and binary forms, with or without modification,
# are permitted under the terms of the BSD-3 License, provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in
#        the documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# You should have received a copy of the BSD-3 License along with mlchem.
# If not, see https://interoperable-europe.ec.europa.eu/licence/bsd-3-clause-new-or-revised-license .
# It is the responsibility of mlchem users to familiarise themselves with all dependencies and their associated licenses.

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