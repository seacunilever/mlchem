import pytest
from rdkit.DataStructs import ExplicitBitVect
from mlchem.metrics import (
    get_sensitivity, get_specificity, get_geometric_S, get_mcc, get_rmse,
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

def test_rmse_to_std_ratio(sample_regression_data):
    y_true, y_pred = sample_regression_data
    ratio = rmse_to_std_ratio(y_true, y_pred)
    assert ratio == pytest.approx(3.59, 0.01)

def test_get_r2(sample_regression_data):
    y_true, y_pred = sample_regression_data
    r2 = get_r2(y_true, y_pred)
    assert r2 == pytest.approx(0.96, 0.01)

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

if __name__ == "__main__":
    pytest.main()