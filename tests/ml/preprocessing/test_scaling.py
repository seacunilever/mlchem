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
from unittest.mock import MagicMock
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from mlchem.ml.preprocessing.scaling import scale_df_standard, scale_df_minmax, scale_df_robust, transform_df

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [7, 8, 9]
    })


@pytest.fixture
def sample_dataframe_with_binary():
    return pd.DataFrame({
        'continuous': [10.0, 20.0, 30.0],
        'binary': [0, 1, 0],
        'tail': [100, 101, 102],
    })

def test_scale_df_standard(sample_dataframe):
    scaled_df, scaler = scale_df_standard(sample_dataframe, last_columns_to_preserve=1)
    assert isinstance(scaled_df, pd.DataFrame)
    assert isinstance(scaler, StandardScaler)
    assert scaled_df.shape == (3, 3)
    assert 'feature3' in scaled_df.columns
    assert scaled_df['feature3'].equals(sample_dataframe['feature3'])

def test_scale_df_minmax(sample_dataframe):
    scaled_df, scaler = scale_df_minmax(sample_dataframe, last_columns_to_preserve=1)
    assert isinstance(scaled_df, pd.DataFrame)
    assert isinstance(scaler, MinMaxScaler)
    assert scaled_df.shape == (3, 3)
    assert 'feature3' in scaled_df.columns
    assert scaled_df['feature3'].equals(sample_dataframe['feature3'])

def test_scale_df_robust(sample_dataframe):
    scaled_df, scaler = scale_df_robust(sample_dataframe, last_columns_to_preserve=1)
    assert isinstance(scaled_df, pd.DataFrame)
    assert isinstance(scaler, RobustScaler)
    assert scaled_df.shape == (3, 3)
    assert 'feature3' in scaled_df.columns
    assert scaled_df['feature3'].equals(sample_dataframe['feature3'])

def test_transform_df_standard(sample_dataframe):
    _, scaler = scale_df_standard(sample_dataframe, last_columns_to_preserve=1)
    transformed_df, _ = transform_df(sample_dataframe, scaler, last_columns_to_preserve=1)
    assert isinstance(transformed_df, pd.DataFrame)
    assert transformed_df.shape == (3, 3)
    assert 'feature3' in transformed_df.columns
    assert transformed_df['feature3'].equals(sample_dataframe['feature3'])

def test_transform_df_minmax(sample_dataframe):
    _, scaler = scale_df_minmax(sample_dataframe, last_columns_to_preserve=1)
    transformed_df, _ = transform_df(sample_dataframe, scaler, last_columns_to_preserve=1)
    assert isinstance(transformed_df, pd.DataFrame)
    assert transformed_df.shape == (3, 3)
    assert 'feature3' in transformed_df.columns
    assert transformed_df['feature3'].equals(sample_dataframe['feature3'])

def test_transform_df_robust(sample_dataframe):
    _, scaler = scale_df_robust(sample_dataframe, last_columns_to_preserve=1)
    transformed_df, _ = transform_df(sample_dataframe, scaler, last_columns_to_preserve=1)
    assert isinstance(transformed_df, pd.DataFrame)
    assert transformed_df.shape == (3, 3)
    assert 'feature3' in transformed_df.columns
    assert transformed_df['feature3'].equals(sample_dataframe['feature3'])


@pytest.mark.parametrize(
    'scale_func, scaler_cls',
    [
        (scale_df_standard, StandardScaler),
        (scale_df_minmax, MinMaxScaler),
        (scale_df_robust, RobustScaler),
    ],
)
def test_scale_df_skip_binary_columns(scale_func, scaler_cls, sample_dataframe_with_binary):
    scaled_df, scaler = scale_func(
        sample_dataframe_with_binary,
        last_columns_to_preserve=1,
        skip_binary_columns=True,
    )

    assert isinstance(scaled_df, pd.DataFrame)
    assert isinstance(scaler, scaler_cls)
    assert scaled_df.shape == sample_dataframe_with_binary.shape
    assert not np.allclose(
        scaled_df['continuous'].to_numpy(),
        sample_dataframe_with_binary['continuous'].to_numpy(),
    )
    assert scaled_df['binary'].equals(sample_dataframe_with_binary['binary'])
    assert scaled_df['tail'].equals(sample_dataframe_with_binary['tail'])


@pytest.mark.parametrize('scale_func', [scale_df_standard, scale_df_minmax, scale_df_robust])
def test_transform_df_skip_binary_columns(scale_func, sample_dataframe_with_binary):
    _, scaler = scale_func(
        sample_dataframe_with_binary,
        last_columns_to_preserve=1,
        skip_binary_columns=True,
    )
    transformed_df, _ = transform_df(
        sample_dataframe_with_binary,
        scaler,
        last_columns_to_preserve=1,
    )

    assert isinstance(transformed_df, pd.DataFrame)
    assert transformed_df.shape == sample_dataframe_with_binary.shape
    assert transformed_df['binary'].equals(sample_dataframe_with_binary['binary'])
    assert transformed_df['tail'].equals(sample_dataframe_with_binary['tail'])


def test_scale_df_standard_zero_columns_preserved(sample_dataframe):
    scaled_df, _ = scale_df_standard(sample_dataframe, last_columns_to_preserve=0)
    assert scaled_df.shape == sample_dataframe.shape


def test_scale_df_standard_negative_columns_preserved_raises(sample_dataframe):
    with pytest.raises(ValueError, match="must be >= 0"):
        scale_df_standard(sample_dataframe, last_columns_to_preserve=-1)


def test_scale_df_minmax_negative_columns_preserved_raises(sample_dataframe):
    with pytest.raises(ValueError, match="must be >= 0"):
        scale_df_minmax(sample_dataframe, last_columns_to_preserve=-1)


def test_scale_df_robust_negative_columns_preserved_raises(sample_dataframe):
    with pytest.raises(ValueError, match="must be >= 0"):
        scale_df_robust(sample_dataframe, last_columns_to_preserve=-1)


def test_transform_df_negative_columns_preserved_raises(sample_dataframe):
    _, scaler = scale_df_standard(sample_dataframe, last_columns_to_preserve=0)
    with pytest.raises(ValueError, match="must be >= 0"):
        transform_df(sample_dataframe, scaler, last_columns_to_preserve=-1)


def test_scale_df_standard_wraps_scaler_valueerror(monkeypatch, sample_dataframe):
    scaler = MagicMock()
    scaler.fit_transform.side_effect = [IndexError('shape'), ValueError('bad data')]
    monkeypatch.setattr('mlchem.ml.preprocessing.scaling.StandardScaler', lambda: scaler)

    with pytest.raises(ValueError, match='Error in scaling data: bad data'):
        scale_df_standard(sample_dataframe, last_columns_to_preserve=0)


def test_scale_df_minmax_wraps_scaler_valueerror(monkeypatch, sample_dataframe):
    scaler = MagicMock()
    scaler.fit_transform.side_effect = [IndexError('shape'), ValueError('bad data')]
    monkeypatch.setattr('mlchem.ml.preprocessing.scaling.MinMaxScaler', lambda: scaler)

    with pytest.raises(ValueError, match='Error in scaling data: bad data'):
        scale_df_minmax(sample_dataframe, last_columns_to_preserve=0)


def test_scale_df_robust_wraps_scaler_valueerror(monkeypatch, sample_dataframe):
    scaler = MagicMock()
    scaler.fit_transform.side_effect = [IndexError('shape'), ValueError('bad data')]
    monkeypatch.setattr('mlchem.ml.preprocessing.scaling.RobustScaler', lambda: scaler)

    with pytest.raises(ValueError, match='Error in scaling data: bad data'):
        scale_df_robust(sample_dataframe, last_columns_to_preserve=0)


def test_transform_df_wraps_scaler_valueerror(sample_dataframe):
    scaler = MagicMock()
    scaler.transform.side_effect = [IndexError('shape'), ValueError('bad data')]

    with pytest.raises(ValueError, match='Error in scaling data: bad data'):
        transform_df(sample_dataframe, scaler, last_columns_to_preserve=0)


def test_scale_df_standard_handles_duplicate_columns():
    df = pd.DataFrame(
        [[1, 4, 7], [2, 5, 8], [3, 6, 9]],
        columns=['feature', 'feature', 'tail'],
    )
    with pytest.raises(Exception, match='Expected unique column names'):
        scale_df_standard(df, last_columns_to_preserve=1)


def test_transform_df_handles_duplicate_columns():
    df = pd.DataFrame(
        [[1, 4, 7], [2, 5, 8], [3, 6, 9]],
        columns=['feature', 'feature', 'tail'],
    )
    with pytest.raises(Exception, match='Expected unique column names'):
        scale_df_minmax(df, last_columns_to_preserve=1)

if __name__ == "__main__":
    pytest.main()