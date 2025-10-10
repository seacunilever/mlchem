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
from mlchem.chem.calculator.descriptors import (get_rdkitDesc,
                                                get_mordredDesc,
                                                get_allDesc,
                                                get_atomicDesc,
                                                get_chemotypes,
                                                get_fingerprint,
                                                get_fingerprint_df,
                                                get_EHT_descriptors
                                                )
from rdkit import DataStructs


def test_get_rdkitDesc_2D():
    # Test with 2D descriptors only
    smiles_list = ['CCO', 'CCN', 'CCC']
    result = get_rdkitDesc(smiles_list, include_3D=False)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'MolWt' in result.columns  # Example check for a specific descriptor


def test_get_rdkitDesc_3D():
    # Test with 3D descriptors included
    smiles_list = ['CCO', 'CCN', 'CCC']
    result = get_rdkitDesc(smiles_list, include_3D=True)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'Asphericity' in result.columns  # Example check for a specific 3D descriptor


def test_get_rdkitDesc_invalid_input():
    # Test with invalid input
    with pytest.raises(ValueError):
        get_rdkitDesc(['invalid_smiles'], include_3D=True)


def test_get_mordredDesc_2D():
    # Test with 2D descriptors only
    smiles_list = ['CCO', 'CCN', 'CCC']
    result = get_mordredDesc(smiles_list, include_3D=False)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'nAtom' in result.columns  # Example check for a specific 2D descriptor


def test_get_mordredDesc_3D():
    # Test with 3D descriptors included
    smiles_list = ['CCO', 'CCN', 'CCC']
    result = get_mordredDesc(smiles_list, include_3D=True)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'PBF' in result.columns  # Example check for a specific 3D descriptor


def test_get_mordredDesc_invalid_input(capfd):
    # Test with invalid input
    result = get_mordredDesc(['abc^'], include_3D=True)
    out, err = capfd.readouterr()
    assert "Problem encountered with: abc^." in out


def test_get_allDesc_2D():
    # Test with 2D descriptors only
    smiles_list = ['CCO', 'CCN', 'CCC']
    result = get_allDesc(smiles_list, include_3D=False)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'MolWt' in result.columns  # Example check for a specific RDKit descriptor
    assert 'nAtom' in result.columns  # Example check for a specific Mordred descriptor


def test_get_allDesc_3D():
    # Test with 3D descriptors included
    smiles_list = ['CCO', 'CCN', 'CCC']
    result = get_allDesc(smiles_list, include_3D=True)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'PBF' in result.columns  # Example check for a specific Mordred 3D descriptor
    assert 'Asphericity' in result.columns  # Example check for a specific rdkit 3D descriptor
    assert 'MinEStateIndex' in result.columns  # Example check for a specific rdkit descriptor
    assert 'nAtom' in result.columns  # Example check for a specific Mordred descriptor


def test_get_allDesc_invalid_input():
    # Test with invalid input
    with pytest.raises(ValueError):
        get_allDesc(['invalid_smiles'], include_3D=True)


def test_get_atomicDesc_valid_input():
    # Test with valid input
    smiles = 'CCO'
    atom_index = 1  # Index of the carbon atom in ethanol
    result = get_atomicDesc(smiles, atom_index)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'SMILES' in result.columns  # Example check for a specific descriptor
    assert result.at[smiles, 'SYMBOL'] == 'C'  # Check if the symbol is correct


def test_get_atomicDesc_invalid_smiles():
    # Test with invalid SMILES input
    with pytest.raises(RuntimeError):
        get_atomicDesc('invalid_smiles', 0)


def test_get_atomicDesc_invalid_atom_index():
    # Test with invalid atom index
    smiles = 'CCO'
    invalid_atom_index = 10  # Index out of range
    with pytest.raises(IndexError):
        get_atomicDesc(smiles, invalid_atom_index)


def test_get_chemotypes_default_dict():
    # Test with default chemotype dictionary

    smiles_list = ['CCO', 'CCN', 'COCC','CCOCCNCO']
    result = get_chemotypes(smiles_list)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert list(result['Alcohol']) == [1,0,0,1]
    assert list(result['Amine']) == [0,1,0,1]
    assert list(result['Ether']) == [0,0,1,1]


def test_get_fingerprint_morgan():
    # Test with Morgan fingerprint
    smiles = 'CCO'
    result = get_fingerprint(smiles, fp_type='m')

    assert isinstance(result, DataStructs.cDataStructs.ExplicitBitVect)
    assert result.GetNumBits() == 2048  # Check the size of the fingerprint


def test_get_fingerprint_atom_pair():
    # Test with Atom Pair fingerprint
    smiles = 'CCO'
    result = get_fingerprint(smiles, fp_type='ap')

    assert isinstance(result, DataStructs.cDataStructs.ExplicitBitVect)
    assert result.GetNumBits() == 2048  # Check the size of the fingerprint


def test_get_fingerprint_rdkit():
    # Test with RDKit fingerprint
    smiles = 'CCO'
    result = get_fingerprint(smiles, fp_type='rk')

    assert isinstance(result, DataStructs.cDataStructs.ExplicitBitVect)
    assert result.GetNumBits() == 2048  # Check the size of the fingerprint


def test_get_fingerprint_topological_torsion():
    # Test with Topological Torsion fingerprint
    smiles = 'CCO'
    result = get_fingerprint(smiles, fp_type='tt')

    assert isinstance(result, DataStructs.cDataStructs.ExplicitBitVect)
    assert result.GetNumBits() == 2048  # Check the size of the fingerprint


def test_get_fingerprint_maccs():
    # Test with MACCS keys fingerprint
    smiles = 'CCO'
    result = get_fingerprint(smiles, fp_type='mac')

    assert isinstance(result, DataStructs.cDataStructs.ExplicitBitVect)
    assert result.GetNumBits() == 167  # MACCS keys have a fixed size of 167 bits


def test_get_fingerprint_with_bit_info():
    # Test with bit information included
    smiles = 'CCO'
    result, bit_info = get_fingerprint(smiles, fp_type='m', include_bit_info=True)

    assert isinstance(result, DataStructs.cDataStructs.ExplicitBitVect)
    assert isinstance(bit_info, dict)
    assert result.GetNumBits() == 2048  # Check the size of the fingerprint


def test_get_chemotypes_invalid_smiles():
    # Test with invalid SMILES input
    with pytest.raises(ValueError):
        get_chemotypes(['invalid_smiles'])


def test_get_fingerprint_df_morgan():
    # Test with Morgan fingerprint
    smiles_list = ['CCO', 'CCN', 'CCC']
    result = get_fingerprint_df(smiles_list, fp_type='m')

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.shape[1] == 2048  # Check the number of bits in the fingerprint


def test_get_fingerprint_df_atom_pair():
    # Test with Atom Pair fingerprint
    smiles_list = ['CCO', 'CCN', 'CCC']
    result = get_fingerprint_df(smiles_list, fp_type='ap')

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.shape[1] == 2048  # Check the number of bits in the fingerprint


def test_get_fingerprint_df_rdkit():
    # Test with RDKit fingerprint
    smiles_list = ['CCO', 'CCN', 'CCC']
    result = get_fingerprint_df(smiles_list, fp_type='rk')

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.shape[1] == 2048  # Check the number of bits in the fingerprint


def test_get_fingerprint_df_topological_torsion():
    # Test with Topological Torsion fingerprint
    smiles_list = ['CCO', 'CCN', 'CCC']
    result = get_fingerprint_df(smiles_list, fp_type='tt')

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.shape[1] == 2048  # Check the number of bits in the fingerprint


def test_get_fingerprint_df_maccs():
    # Test with MACCS keys fingerprint
    smiles_list = ['CCO', 'CCN', 'CCC']
    result = get_fingerprint_df(smiles_list, fp_type='mac')

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.shape[1] == 167  # MACCS keys have a fixed size of 167 bits


def test_get_fingerprint_df_with_bit_info():
    # Test with bit information included
    smiles_list = ['CCO', 'CCN', 'CCC']
    result, bit_info = get_fingerprint_df(smiles_list, fp_type='m', include_bit_info=True)

    assert isinstance(result, pd.DataFrame)
    assert isinstance(bit_info, dict)
    assert not result.empty
    assert result.shape[1] == 2048  # Check the number of bits in the fingerprint


def test_get_EHT_descriptors_valid_input():
    from mlchem.chem.manipulation import create_molecule
    # Test with valid input
    smiles = 'CCO'
    mol = create_molecule(smiles,is_3d=True)
    result = get_EHT_descriptors(mol)

    assert isinstance(result, dict)
    assert 'AtomicCharges' in result
    assert 'TotalEnergy' in result


def test_get_EHT_descriptors_invalid_input():
    from mlchem.chem.manipulation import create_molecule
    # Test with invalid input (non-3D molecule)
    smiles = 'CCO'
    mol = create_molecule(smiles,is_3d=False)
    with pytest.raises(ValueError):
        get_EHT_descriptors(mol)


def test_get_EHT_descriptors_with_conf_id():
    from mlchem.chem.manipulation import PropManager as pm
    from mlchem.chem.manipulation import create_molecule

    # Test with a specific conformer ID
    smiles = 'CCO'
    mol = create_molecule(smiles)
    pm.Conformation.generate_conformers(mol,10)
    result = get_EHT_descriptors(mol, conf_id=6)

    assert isinstance(result, dict)
    assert 'AtomicCharges' in result
    assert 'TotalEnergy' in result


if __name__ == "__main__":
    pytest.main()
