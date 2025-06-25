import pytest
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlchem.chem.visualise.simmaps import SimMaps
import matplotlib.pyplot as plt

@pytest.fixture
def sample_molecule():
    return Chem.MolFromSmiles('CCO')

@pytest.fixture
def sample_estimator():
    # Create a simple RandomForestClassifier for testing
    estimator = RandomForestClassifier()
    # Fit the estimator with dummy data
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    estimator.fit(X, y)
    return estimator

@pytest.fixture
def d2d():
    return Draw.MolDraw2DCairo(150,100)

def test_get_weights_from_model(sample_molecule, sample_estimator):
    estimator_cols = ['m1', 'm2']
    result = SimMaps.get_weights_from_model(
        mol_input=sample_molecule,
        estimator=sample_estimator,
        estimator_cols=estimator_cols,
        model_type='classification',
        actual_val=0.5,
        fp_type='m',
        normalise=True,
        return_df=True
    )
    assert isinstance(result, pd.DataFrame)
    assert 'Delta' in result.columns

def test_get_weights_from_fingerprint(sample_molecule):
    result = SimMaps.get_weights_from_fingerprint(
        refmol=sample_molecule,
        probemol=sample_molecule,
        fp_type='m',
        similarity_metric='Tanimoto',
        normalise=True,
        return_df=True
    )
    assert isinstance(result, pd.DataFrame)
    assert 'Delta' in result.columns


if __name__ == "__main__":
    pytest.main()