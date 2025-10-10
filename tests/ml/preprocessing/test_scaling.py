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