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
import numpy as np
from rdkit.DataStructs import ExplicitBitVect
from mlchem.metrics import (
    get_sensitivity, get_specificity, get_geometric_S, get_mcc, get_rmse,
    calculate_reliability_components,
    rmse_to_std_ratio, get_r2, DiceSimilarity, OnBitSimilarity,
    SokalSimilarity, AllBitSimilarity, CosineSimilarity, RusselSimilarity,
    TverskySimilarity, TanimotoSimilarity, AsymmetricSimilarity,
    KulczynskiSimilarity, OffBitProjSimilarity, McConnaugheySimilarity,
    BraunBlanquetSimilarity, RogotGoldbergSimilarity, FingerprintSimilarity
)

@pytest.fixture
def sample_classification_data():
    y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
    return y_true, y_pred

@pytest.fixture
def sample_regression_data():
    y_true = [2.5, 0.0, 2.1, 1.6]
    y_pred = [3.0, -0.1, 2.0, 1.5]
    return y_true, y_pred

@pytest.fixture
def sample_fingerprints():
    fp1 = ExplicitBitVect(1024)
    fp2 = ExplicitBitVect(1024)
    fp1.SetBit(0)
    fp1.SetBit(1)
    fp2.SetBit(0)
    fp2.SetBit(2)
    return fp1, fp2

def test_get_sensitivity(sample_classification_data):
    y_true, y_pred = sample_classification_data
    sensitivity = get_sensitivity(y_true, y_pred)
    assert sensitivity == 0.8

def test_get_specificity(sample_classification_data):
    y_true, y_pred = sample_classification_data
    specificity = get_specificity(y_true, y_pred)
    assert specificity == 0.8

def test_get_geometric_S(sample_classification_data):
    y_true, y_pred = sample_classification_data
    geometric_S = get_geometric_S(y_true, y_pred)
    assert geometric_S == pytest.approx(0.8, 0.01)

def test_get_geometric_S_degenerate(y_true=[0,0,0], y_pred=[0,0,0]):
    geometric_S = get_geometric_S(y_true, y_pred)
    assert geometric_S == 1

def test_sensitivity_all_negative():
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    assert get_sensitivity(y_true, y_pred) == 0.0

def test_specificity_all_positive():
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 1]
    assert get_specificity(y_true, y_pred) == 0.0

def test_sensitivity_with_missing_class():
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    # Only one class present, should not crash
    assert get_sensitivity(y_true, y_pred) == 0.0

def test_specificity_with_missing_class():
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 1]
    # Only one class present, should not crash
    assert get_specificity(y_true, y_pred) == 0.0

def test_get_mcc(sample_classification_data):
    y_true, y_pred = sample_classification_data
    mcc = get_mcc(y_true, y_pred)
    assert mcc == pytest.approx(0.6, 0.01)

def test_get_rmse(sample_regression_data):
    y_true, y_pred = sample_regression_data
    rmse = get_rmse(y_true, y_pred)
    assert rmse == pytest.approx(0.264, 0.01)


def test_calculate_reliability_components_greater_logic():
    scores = calculate_reliability_components(
        train_score=0.9,
        cv_score=0.8,
        test_score=0.7,
        logic='greater',
    )

    expected_geometric_mean = (0.9 * 0.8 * 0.7) ** (1/3)
    expected_instability = abs(0.9 - 0.8) + abs(0.9 - 0.7) + abs(0.8 - 0.7)
    assert scores['geometric_mean'] == pytest.approx(expected_geometric_mean)
    assert scores['performance_score'] == pytest.approx(expected_geometric_mean)
    assert scores['instability_score'] == pytest.approx(expected_instability)
    assert scores['reliability_score'] == pytest.approx(
        expected_geometric_mean / (1 + expected_instability)
    )


def test_calculate_reliability_components_lower_logic_inverts_performance():
    scores = calculate_reliability_components(
        train_score=0.5,
        cv_score=0.6,
        test_score=0.55,
        logic='lower',
    )

    expected_geometric_mean = (0.5 * 0.6 * 0.55) ** (1/3)
    expected_performance = 1 / expected_geometric_mean
    expected_instability = abs(0.5 - 0.6) + abs(0.5 - 0.55) + abs(0.6 - 0.55)
    assert scores['geometric_mean'] == pytest.approx(expected_geometric_mean)
    assert scores['performance_score'] == pytest.approx(expected_performance)
    assert scores['instability_score'] == pytest.approx(expected_instability)
    assert scores['reliability_score'] == pytest.approx(
        expected_performance / (1 + expected_instability)
    )


def test_calculate_reliability_components_rejects_invalid_logic():
    with pytest.raises(ValueError, match="'logic' must be either 'lower' or 'greater'"):
        calculate_reliability_components(0.9, 0.8, 0.7, logic='best')

def test_rmse_to_std_ratio(sample_regression_data):
    y_true, y_pred = sample_regression_data
    ratio = rmse_to_std_ratio(y_true, y_pred)
    assert ratio == pytest.approx(3.59, 0.01)

def test_get_r2(sample_regression_data):
    y_true, y_pred = sample_regression_data
    r2 = get_r2(y_true, y_pred)
    assert r2 == pytest.approx(0.96, 0.01)


def test_get_r2_constant_target_returns_nan():
    y_true = [1.0, 1.0, 1.0, 1.0]
    y_pred = [0.8, 1.1, 1.2, 0.9]
    r2 = get_r2(y_true, y_pred)
    assert np.isnan(r2)


def test_rmse_to_std_ratio_zero_rmse_returns_infinity():
    y_true = [0.0, 1.0, 2.0, 3.0]
    y_pred = [0.0, 1.0, 2.0, 3.0]
    ratio = rmse_to_std_ratio(y_true, y_pred)
    assert np.isinf(ratio)

def test_DiceSimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = DiceSimilarity(fp1, fp2)
    assert similarity == pytest.approx(0.5, 0.01)

def test_OnBitSimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = OnBitSimilarity(fp1, fp2)
    assert similarity == pytest.approx(0.333, 0.01)

def test_SokalSimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = SokalSimilarity(fp1, fp2)
    assert similarity == pytest.approx(0.2, 0.01)

def test_AllBitSimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = AllBitSimilarity(fp1, fp2)
    assert similarity == pytest.approx(0.99, 0.01)

def test_CosineSimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = CosineSimilarity(fp1, fp2)
    assert similarity == pytest.approx(0.5, 0.01)

def test_RusselSimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = RusselSimilarity(fp1, fp2)
    assert similarity == pytest.approx(0.00097, 0.01)

def test_TverskySimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = TverskySimilarity(fp1, fp2, a=0.5, b=0.5)
    assert similarity == pytest.approx(0.5, 0.01)

def test_TanimotoSimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = TanimotoSimilarity(fp1, fp2)
    assert similarity == pytest.approx(0.333, 0.01)

def test_AsymmetricSimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = AsymmetricSimilarity(fp1, fp2)
    assert similarity == pytest.approx(0.5, 0.01)

def test_KulczynskiSimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = KulczynskiSimilarity(fp1, fp2)
    assert similarity == pytest.approx(0.5, 0.01)

def test_OffBitProjSimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity_vector = OffBitProjSimilarity(fp1, fp2)
    similarity = similarity_vector[0]  # Extract the first element
    assert similarity == pytest.approx(0.999, 0.01)

def test_McConnaugheySimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = McConnaugheySimilarity(fp1, fp2)
    assert similarity == pytest.approx(0.0, 0.01)

def test_BraunBlanquetSimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = BraunBlanquetSimilarity(fp1, fp2)
    assert similarity == pytest.approx(0.5, 0.01)

def test_RogotGoldbergSimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = RogotGoldbergSimilarity(fp1, fp2)
    assert similarity == pytest.approx(0.749, 0.01)

def test_FingerprintSimilarity(sample_fingerprints):
    fp1, fp2 = sample_fingerprints
    similarity = FingerprintSimilarity(fp1, fp2, metric=TanimotoSimilarity)
    assert similarity == pytest.approx(0.333, 0.01)


def test_similarity_identical_fingerprints_is_one():
    fp = ExplicitBitVect(1024)
    fp.SetBit(0)
    fp.SetBit(10)
    fp.SetBit(200)

    assert TanimotoSimilarity(fp, fp) == pytest.approx(1.0, 1e-12)
    assert DiceSimilarity(fp, fp) == pytest.approx(1.0, 1e-12)
    assert CosineSimilarity(fp, fp) == pytest.approx(1.0, 1e-12)
    assert FingerprintSimilarity(fp, fp, metric=TanimotoSimilarity) == pytest.approx(1.0, 1e-12)

if __name__ == "__main__":
    pytest.main()