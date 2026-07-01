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
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlchem.chem.visualise.simmaps import SimMaps
from mlchem.chem.manipulation import create_molecule
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


def test_get_weights_from_model_returns_array(sample_molecule, sample_estimator):
    estimator_cols = ['m1', 'm2']
    result = SimMaps.get_weights_from_model(
        mol_input=sample_molecule,
        estimator=sample_estimator,
        estimator_cols=estimator_cols,
        model_type='classification',
        actual_val=0.5,
        fp_type='m',
        normalise=False,
        return_df=False,
    )
    assert isinstance(result, np.ndarray)
    assert len(result) == sample_molecule.GetNumAtoms()


def test_get_weights_from_fingerprint_returns_array(sample_molecule):
    result = SimMaps.get_weights_from_fingerprint(
        refmol=sample_molecule,
        probemol=sample_molecule,
        fp_type='m',
        similarity_metric='Dice',
        normalise=False,
        return_df=False,
    )
    assert isinstance(result, np.ndarray)
    assert len(result) == sample_molecule.GetNumAtoms()


def test_get_similarity_map_from_weights_requires_draw2d(sample_molecule):
    with pytest.raises(ValueError, match='draw2d argument must be provided'):
        SimMaps.get_similarity_map_from_weights(sample_molecule, [0.1, 0.2, 0.3], draw2d=None)


def test_get_similarity_map_from_weights_rejects_too_few_atoms():
    methane = Chem.MolFromSmiles('C')
    with pytest.raises(ValueError, match='too few atoms'):
        SimMaps.get_similarity_map_from_weights(
            methane,
            [0.1],
            draw2d=Draw.MolDraw2DCairo(150, 100),
        )


def test_get_similarity_map_from_weights_with_string_colormap():
    mol = create_molecule('CCO', is_3d=True)
    weights = [0.3, -0.2, 0.1]
    draw2d = Draw.MolDraw2DCairo(200, 160)
    returned = SimMaps.get_similarity_map_from_weights(
        mol=mol,
        weights=weights,
        colorMap='PiYG',
        contour_colour='black',
        draw2d=draw2d,
    )
    assert returned is draw2d


if __name__ == "__main__":
    pytest.main()