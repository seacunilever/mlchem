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
import logging
import pandas as pd
from unittest.mock import patch
from mlchem.ml.preprocessing.undersampling import check_class_balance, undersample

@pytest.fixture
def sample_data():
    train_set = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'class': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    })
    test_set = pd.DataFrame({
        'feature1': [11, 12, 13, 14, 15],
        'class': [0, 1, 0, 1, 0]
    })
    return train_set, test_set

def test_check_class_balance_logs_by_default(sample_data, caplog):
    train_set, test_set = sample_data
    y_train = train_set['class'].values.tolist()
    with caplog.at_level(logging.INFO, logger='mlchem.ml.preprocessing.undersampling'):
        check_class_balance(y_train)
    assert any('CLASS BALANCE' in msg for msg in caplog.messages)
    assert any('[0]: 2 (0.20)' in msg for msg in caplog.messages)
    assert any('[1]: 8 (0.80)' in msg for msg in caplog.messages)


def test_check_class_balance_logs_at_info_level(sample_data, caplog):
    train_set, test_set = sample_data
    y_train = train_set['class'].values.tolist()
    with caplog.at_level(logging.INFO, logger='mlchem.ml.preprocessing.undersampling'):
        check_class_balance(y_train, log_level='INFO')
    assert any('CLASS BALANCE' in msg for msg in caplog.messages)


def test_check_class_balance_returns_correct_summary(sample_data):
    train_set, _ = sample_data
    y_train = train_set['class'].values.tolist()

    summary = check_class_balance(y_train)

    assert summary[0]['count'] == 2
    assert summary[1]['count'] == 8
    assert abs(summary[0]['proportion'] - 0.2) < 0.01
    assert abs(summary[1]['proportion'] - 0.8) < 0.01

def test_undersample(sample_data):
    train_set, test_set = sample_data
    undersampled_train_set, undersampled_test_set = undersample(
        train_set=train_set,
        test_set=test_set,
        class_column='class',
        desired_proportion_majority=0.5,
        add_dropped_to_test=False,
        random_seed=1
    )
    assert len(undersampled_train_set) == 4
    assert len(undersampled_test_set) == 5

def test_undersample_add_dropped_to_test(sample_data):
    train_set, test_set = sample_data
    undersampled_train_set, undersampled_test_set = undersample(
        train_set=train_set,
        test_set=test_set,
        class_column='class',
        desired_proportion_majority=0.5,
        add_dropped_to_test=True,
        random_seed=1
    )
    assert len(undersampled_train_set) == 4
    assert len(undersampled_test_set) == 11


def test_undersample_ratio_equal_1_raises_valueerror(sample_data):
    train_set, test_set = sample_data
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        undersample(
            train_set=train_set,
            test_set=test_set,
            class_column='class',
            desired_proportion_majority=1.0,
            add_dropped_to_test=False,
            random_seed=1,
        )


def test_undersample_ratio_zero_raises_valueerror(sample_data):
    train_set, test_set = sample_data
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        undersample(
            train_set=train_set,
            test_set=test_set,
            class_column='class',
            desired_proportion_majority=0.0,
            add_dropped_to_test=False,
            random_seed=1,
        )


def test_undersample_ratio_negative_raises_valueerror(sample_data):
    train_set, test_set = sample_data
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        undersample(
            train_set=train_set,
            test_set=test_set,
            class_column='class',
            desired_proportion_majority=-0.2,
            add_dropped_to_test=False,
            random_seed=1,
        )


def test_undersample_ratio_above_one_raises_valueerror(sample_data):
    train_set, test_set = sample_data
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        undersample(
            train_set=train_set,
            test_set=test_set,
            class_column='class',
            desired_proportion_majority=1.2,
            add_dropped_to_test=False,
            random_seed=1,
        )


def test_undersample_ratio_above_one_raises_clean_error(sample_data):
    train_set, test_set = sample_data
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        undersample(
            train_set=train_set,
            test_set=test_set,
            class_column='class',
            desired_proportion_majority=1.2,
            add_dropped_to_test=False,
            random_seed=1,
        )


def test_undersample_random_seed_zero_seeds_rng(sample_data):
    train_set, test_set = sample_data
    with patch('random.seed') as seed_mock:
        undersample(
            train_set=train_set,
            test_set=test_set,
            class_column='class',
            desired_proportion_majority=0.5,
            add_dropped_to_test=False,
            random_seed=0,
        )
    seed_mock.assert_called_once_with(0)


def test_undersample_rejects_non_binary_labels():
    train_set = pd.DataFrame(
        {
            'feature1': [1, 2, 3, 4, 5, 6],
            'class': ['a', 'b', 'c', 'a', 'b', 'c'],
        }
    )
    test_set = pd.DataFrame({'feature1': [7, 8], 'class': ['a', 'b']})

    with pytest.raises(ValueError, match='supports exactly two classes'):
        undersample(
            train_set=train_set,
            test_set=test_set,
            class_column='class',
            desired_proportion_majority=0.5,
        )


def test_undersample_rejects_impossible_majority_growth_target(sample_data):
    train_set, test_set = sample_data

    # Current majority proportion is 0.8; target 0.95 would require growth.
    with pytest.raises(ValueError, match='implies majority-class growth'):
        undersample(
            train_set=train_set,
            test_set=test_set,
            class_column='class',
            desired_proportion_majority=0.95,
        )

if __name__ == "__main__":
    pytest.main()