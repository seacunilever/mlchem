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