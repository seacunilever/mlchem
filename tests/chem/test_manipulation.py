import pytest
from rdkit import Chem
import numpy as np
from mlchem.chem.manipulation import (mol_from_string,
                                      smiles_to_inchi,
                                      create_molecule,
                                      kekulise_smiles,
                                      unkekulise_smiles,
                                      smarts_from_string,
                                      mol_to_binary,
                                      generate_resonance,
                                      neutralise_mol,
                                      remove_smarts_pattern,
                                      PropManager,
                                      PatternRecognition)

def test_mol_from_string():
    # Test with a valid SMILES string
    smiles = "CCO"
    mol = mol_from_string(smiles)
    assert mol is not None, "Failed to create molecule from valid SMILES string"
    assert Chem.MolToSmiles(mol) == smiles, "SMILES string does not match"
    assert mol.GetNumAtoms() == 3

    # Test with a valid InChI string
    inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
    mol = mol_from_string(inchi)
    assert mol is not None, "Failed to create molecule from valid InChI string"
    assert Chem.MolToInchi(mol) == inchi, "InChI string does not match"
    assert mol.GetNumAtoms() == 3

    # Test with an invalid string
    invalid = "invalid_string"
    with pytest.raises(ValueError):
        mol_from_string(invalid)

def test_smiles_to_inchi():
    # Test with a valid SMILES string
    smiles = "CCO"
    inchi = smiles_to_inchi(smiles)
    assert inchi is not None, "Failed to convert valid SMILES string to InChI"
    assert inchi == 'InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3', "InChI string does not match"

    # Test with another valid SMILES string
    smiles = "C1=CC=CC=C1"
    inchi = smiles_to_inchi(smiles)
    assert inchi is not None, "Failed to convert valid SMILES string to InChI"
    assert inchi == 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H', "InChI string does not match"

    # Test with an invalid SMILES string
    invalid_smiles = "invalid_smiles"
    with pytest.raises(ValueError):
        smiles_to_inchi(invalid_smiles)

def test_create_molecule():
    # Test with SMILES string
    smiles = "CCO"
    mol = create_molecule(smiles)
    assert mol is not None, "Failed to create molecule from SMILES"

    # Test with RDKit Mol object
    mol_input = Chem.MolFromSmiles(smiles)
    mol = create_molecule(mol_input)
    assert mol is not None, "Failed to create molecule from RDKit Mol object"

    # Test with adding hydrogens
    mol = create_molecule(smiles, add_hydrogens=True)
    assert mol.GetNumAtoms() > Chem.MolFromSmiles(smiles).GetNumAtoms(), "Failed to add hydrogens"

    # Test with 3D coordinates
    mol = create_molecule(smiles, is_3d=True)
    assert mol.GetConformer(), "Mol is 3D but without conformers"

    # Test with optimization
    mol = create_molecule(smiles, optimise=True)
    assert mol is not None, "Failed to optimise molecule"

def test_kekulise_smiles():
    smiles = "c1ccccc1"
    expected_kekule = "C1=CC=CC=C1"
    assert kekulise_smiles(smiles) == expected_kekule

def test_unkekulise_smiles():
    kekule_smiles = "C1=CC=CC=C1"
    expected_smiles = "c1ccccc1"
    assert unkekulise_smiles(kekule_smiles) == expected_smiles

def test_smarts_from_string():
    smiles = "CCO"
    expected_smarts = "[#6]-[#6]-[#8]"
    assert smarts_from_string(smiles) == expected_smarts

def test_mol_to_binary():
    smiles = "CCO"
    mol = mol_from_string(smiles)
    binary = mol_to_binary(mol)
    assert isinstance(binary, bytes)
    assert len(binary) > 0

def test_generate_resonance():
    smiles = "C1=CC=CC=C1"
    resonance_structures = generate_resonance(smiles)
    assert isinstance(resonance_structures, list)
    assert len(resonance_structures) > 0
    for img in resonance_structures:
        assert isinstance(img, bytes)
        assert len(img) > 0

def test_neutralise_mol():
    smiles = "CC(O)[O-]"
    mol = mol_from_string(smiles)
    neutral_mol = neutralise_mol(mol)
    neutral_smiles = Chem.MolToSmiles(neutral_mol)
    assert neutral_smiles == "CC(O)O"

def test_remove_smarts_pattern():
    smiles = "CCOCO"
    smarts = "[OD1]"     # terminal oxygen
    mol = mol_from_string(smiles)
    modified_mol = remove_smarts_pattern(mol, smarts)
    modified_smiles = Chem.MolToSmiles(modified_mol)
    assert modified_smiles == "CCOC"

def test_assign_atom_mapnumbers():
    smiles = "CCO"
    mol = mol_from_string(smiles)
    PropManager.Base.assign_atom_mapnumbers(mol)
    for atom in mol.GetAtoms():
        assert atom.HasProp('molAtomMapNumber')

def test_assign_atom_labels():
    smiles = "CCO"
    mol = mol_from_string(smiles)
    labels = ["C1", "C2", "O1"]
    PropManager.Base.assign_atom_labels(mol, labels)
    for atom, label in zip(mol.GetAtoms(), labels):
        assert atom.GetProp('atomLabel') == label

def test_assign_atom_notes():
    smiles = "CCO"
    mol = mol_from_string(smiles)
    notes = ["note1", "note2", "note3"]
    PropManager.Base.assign_atom_notes(mol, notes)
    for atom, note in zip(mol.GetAtoms(), notes):
        assert atom.GetProp('atomNote') == note

def test_assign_bond_notes():
    smiles = "CCO"
    mol = mol_from_string(smiles)
    notes = ["note1", "note2"]
    PropManager.Base.assign_bond_notes(mol, notes)
    for bond, note in zip(mol.GetBonds(), notes):
        assert bond.GetProp('bondNote') == note

def test_clear_all_atomprops():
    smiles = "CCO"
    mol = mol_from_string(smiles)
    PropManager.Base.assign_atom_mapnumbers(mol)
    PropManager.Base.clear_all_atomprops(mol)
    for atom in mol.GetAtoms():
        assert not atom.HasProp('molAtomMapNumber')

def test_clear_prop():
    smiles = "CCO"
    mol = mol_from_string(smiles)
    PropManager.Base.set_prop(mol, 'testProp', 'testValue')
    PropManager.Base.clear_prop(mol, 'testProp')
    assert not mol.HasProp('testProp')

def test_get_props_dict():
    smiles = "CCO"
    mol = mol_from_string(smiles)
    PropManager.Base.set_prop(mol, 'testProp', 'testValue')
    props_dict = PropManager.Base.get_props_dict(mol)
    assert 'testProp' in props_dict
    assert props_dict['testProp'] == 'testValue'

def test_get_prop_names():
    smiles = "CCO"
    mol = mol_from_string(smiles)
    PropManager.Base.set_prop(mol, 'testProp', 'testValue')
    prop_names = PropManager.Base.get_prop_names(mol)
    assert 'testProp' in prop_names

def test_get_prop():
    smiles = "CCO"
    mol = mol_from_string(smiles)
    PropManager.Base.set_prop(mol, 'testProp', 'testValue')
    prop_value = PropManager.Base.get_prop(mol, 'testProp')
    assert prop_value == 'testValue'

def test_set_prop():
    smiles = "CCO"
    mol = mol_from_string(smiles)
    PropManager.Base.set_prop(mol, 'testProp', 'testValue')
    assert mol.GetProp('testProp') == 'testValue'

def test_get_owning_mol():
    smiles = "CCO"
    mol = create_molecule(smiles)
    atom = mol.GetAtomWithIdx(0)
    owning_mol = PropManager.Base.get_owning_mol(atom)
    assert smiles == Chem.MolToSmiles(owning_mol)

class TestPropManagerMol:

    @pytest.fixture
    def mol(self):
        smiles = "CCO"
        return create_molecule(smiles,is_3d=True)

    def test_get_atoms(self, mol):
        atoms = PropManager.Mol.get_atoms(mol)
        assert len(atoms) == 3

    def test_get_atoms_from_idx(self, mol):
        atom = PropManager.Mol.get_atoms_from_idx(mol, 1)
        assert atom.GetSymbol() == "C"

    def test_get_bonds(self, mol):
        bonds = PropManager.Mol.get_bonds(mol)
        assert len(bonds) == 2

    def test_get_bonds_from_idx(self, mol):
        bond = PropManager.Mol.get_bonds_from_idx(mol, 0)[0]
        assert bond.GetBondType() == Chem.rdchem.BondType.SINGLE

    def test_get_bond_between_atoms(self, mol):
        bond = PropManager.Mol.get_bond_between_atoms(mol, 0, 1)
        assert bond.GetBondType() == Chem.rdchem.BondType.SINGLE

    def test_get_coordinates(self):
        mol2d = create_molecule('CCO',is_3d=False)
        mol3d = create_molecule('CCO',is_3d=True)
        coords_2d = PropManager.Mol.get_coordinates(mol2d, is_3d=False)
        coords_3d = PropManager.Mol.get_coordinates(mol3d, is_3d=True)
        assert coords_2d.shape == (3, 3)
        assert coords_3d.shape == (3, 3)
        assert coords_2d[:,-1].sum() == 0     # no 3d position for any atom
        assert (coords_3d != 0).sum() > 0     # at least 1 nonzero 3d position

    def test_get_conformer(self, mol):
        conformer = PropManager.Mol.get_conformer(mol)
        assert isinstance(conformer, Chem.rdchem.Conformer)

    def test_get_conformers(self, mol):
        conformers = PropManager.Mol.get_conformers(mol)
        assert len(conformers) == 1

    def test_get_conf_ids(self, mol):
        conf_ids = PropManager.Mol.get_conf_ids(mol)
        assert conf_ids == [0]

    def test_get_distance_matrix(self, mol):
        dist_matrix = PropManager.Mol.get_distance_matrix(mol, is_3d=False)
        assert dist_matrix.shape == (3, 3)

    def test_get_gasteiger_charges(self, mol):
        charges = PropManager.Mol.get_gasteiger_charges(mol)
        assert len(charges) == 3

    def test_get_stereogroups(self, mol):
        stereogroups = PropManager.Mol.get_stereogroups(mol)
        assert [sg for sg in stereogroups] == []

    def test_remove_conformer(self, mol):
        PropManager.Mol.remove_conformer(mol, 0)
        assert len(mol.GetConformers()) == 0

    def test_remove_all_conformers(self, mol):
        PropManager.Mol.remove_all_conformers(mol)
        assert len(mol.GetConformers()) == 0


class TestPatternRecognition:

    # Base

    @staticmethod
    def test_check_smarts_pattern():
        # Test with a valid SMILES string and SMARTS pattern
        smiles = "CCO"
        smarts_pattern = "[OX2H]"
        result = PatternRecognition.Base.check_smarts_pattern(smiles, smarts_pattern)
        assert result[0] is True, "Pattern not found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == smarts_pattern, "SMARTS pattern does not match with input"

        # Test with a molecule object and SMARTS pattern
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.Base.check_smarts_pattern(mol, smarts_pattern)
        assert result[0] is True, "Pattern not found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == smarts_pattern, "SMARTS pattern does not match with input"
    
    @staticmethod
    def test_pattern_abs_fraction_greater_than():
        # Test with single pattern function: Carbon > 50% tot atoms
        smiles = "CCO"
        pattern_function = PatternRecognition.MolPatterns.check_carbon
        threshold = 0.5
        result = PatternRecognition.Base.pattern_abs_fraction_greater_than(smiles, pattern_function, threshold)
        assert result is True, f"Pattern fraction does not exceed the threshold. {result}"

        # Test with nested pattern functions: Aromatic Carbon > 50% tot atoms
        smiles = 'c1ccccc1'
        pattern_function = PatternRecognition.MolPatterns.check_pattern_aromatic
        hidden_pattern_function = PatternRecognition.MolPatterns.check_carbon
        result = PatternRecognition.Base.pattern_abs_fraction_greater_than(smiles, pattern_function, threshold, hidden_pattern_function)
        assert result is True, f"Pattern fraction does not exceed the threshold. {result}"

    @staticmethod
    def test_pattern_rel_fraction_greater_than():
        # Test with single pattern function: Carbon > 50% tot C
        smiles = "CCO"
        pattern_function_1 = PatternRecognition.MolPatterns.check_alkyl_carbon
        pattern_function_2 = PatternRecognition.MolPatterns.check_carbon
        threshold = 0.5
        result = PatternRecognition.Base.pattern_rel_fraction_greater_than(smiles,pattern_function_1,pattern_function_2,threshold)
        assert result is True, "Pattern fraction does not exceed the threshold"

    @staticmethod
    def test_get_atoms():
        # Test with a valid SMILES string
        smiles = "CCO"
        atoms = PatternRecognition.Base.get_atoms(smiles)
        assert len(atoms) == 3, "Number of atoms does not match"

    @staticmethod
    def test_count_atoms():
        # Test with a valid SMILES string
        smiles = "CCO"
        atom_count = PatternRecognition.Base.count_atoms(smiles)
        assert atom_count == 3, "Number of atoms does not match"

    @staticmethod
    def test_get_bonds():

        # Test with a valid SMILES string
        smiles = "CCO"
        bonds = PatternRecognition.Base.get_bonds(smiles)
        assert len(bonds) == 2, "Number of bonds does not match"

        smiles = "C"
        bonds = PatternRecognition.Base.get_bonds(smiles)
        assert len(bonds) == 0, "Expected 0 bonds"

    @staticmethod
    def test_count_bonds():
        # Test with a valid SMILES string
        smiles = "CCO"
        bond_count = PatternRecognition.Base.count_bonds(smiles)
        assert bond_count == 2, "Number of bonds does not match"

        # Test with a molecule object
        mol = Chem.MolFromSmiles("CCO")
        bond_count = PatternRecognition.Base.count_bonds(mol)
        assert bond_count == 2, "Number of bonds does not match"

    @staticmethod
    def test_get_tautomers():
        # Test with a valid SMILES string
        smiles = "CCC"
        tautomers = PatternRecognition.Base.get_tautomers(smiles)
        assert len(tautomers) == 1

        smiles = "c1cnccc1O"
        tautomers = PatternRecognition.Base.get_tautomers(smiles)
        assert len(tautomers) > 1

    @staticmethod
    def test_get_stereoisomers():
        # Test with a valid SMILES string
        smiles = "CC"
        stereoisomers, images = PatternRecognition.Base.get_stereoisomers(smiles)
        assert len(stereoisomers) == 1
        assert len(images) == len(stereoisomers)

        smiles = "CCCC(CO)CC(O)"
        stereoisomers, images = PatternRecognition.Base.get_stereoisomers(smiles)
        assert len(stereoisomers) > 1
        assert len(images) == len(stereoisomers)

    @staticmethod
    def test_is_organic():
        # Test with an organic molecule
        smiles = "CCO"
        result = PatternRecognition.Base.is_organic(smiles)
        assert result is True, "Molecule should be organic"

        # Test with an inorganic molecule
        smiles = "[Na+].[Cl-]"
        result = PatternRecognition.Base.is_organic(smiles)
        assert result is False, "Molecule should be inorganic"

    @staticmethod
    def test_has_carbon_ion():
        # Test with a molecule containing a carbocation
        smiles = "CCC[C+]CC"
        result = PatternRecognition.Base.has_carbon_ion(smiles)
        assert result is True, "Molecule should contain a carbocation"

        # Test with a molecule containing a carbanion
        smiles = "CCC[C-]CC"
        result = PatternRecognition.Base.has_carbon_ion(smiles)
        assert result is True, "Molecule should contain a carbanion"

        # Test with a molecule without carbon ions
        smiles = "CCO"
        result = PatternRecognition.Base.has_carbon_ion(smiles)
        assert result is False, "Molecule should not contain carbon ions"

    @staticmethod
    def test_has_metal_salt():
        # Test with a molecule containing a metal salt
        smiles = "[Na+].[Cl-]"
        result = PatternRecognition.Base.has_metal_salt(smiles)
        assert result is True, "Molecule should contain a metal salt"

        # Test with a molecule without metal salts
        smiles = "CCO"
        result = PatternRecognition.Base.has_metal_salt(smiles)
        assert result is False, "Molecule should not contain metal salts"

    @staticmethod
    def test_get_MCS():
        # Test with two smile
        smiles1 = "CCCCCCO"
        smiles2 = "CCCCCN"
        result = PatternRecognition.Base.get_MCS(smiles1, smiles2)
        
        assert result[0] is True, "MCS not found"
        assert len(result[2]) > 0, "No atom indices returned for molecule 1"
        assert len(result[3]) > 0, "No atom indices returned for molecule 2"
        assert result[4] > 0, "Similarity score should be greater than 0"

    # Atoms

    @staticmethod
    def test_is_SP():
        # Test with an SP-hybridised atom
        mol = Chem.MolFromSmiles("C#C")
        atom = mol.GetAtomWithIdx(0)
        result = PatternRecognition.Atoms.is_SP(atom)
        assert result == 1, "Atom should be SP-hybridised"

        # Test with a non-SP-hybridised atom
        mol = Chem.MolFromSmiles("CCO")
        atom = mol.GetAtomWithIdx(0)
        result = PatternRecognition.Atoms.is_SP(atom)
        assert result == 0, "Atom should not be SP-hybridised"

    @staticmethod
    def test_is_SP2():
        # Test with an SP2-hybridised atom
        mol = Chem.MolFromSmiles("C=C")
        atom = mol.GetAtomWithIdx(0)
        result = PatternRecognition.Atoms.is_SP2(atom)
        assert result == 1, "Atom should be SP2-hybridised"

        # Test with a non-SP2-hybridised atom
        mol = Chem.MolFromSmiles("CCO")
        atom = mol.GetAtomWithIdx(0)
        result = PatternRecognition.Atoms.is_SP2(atom)
        assert result == 0, "Atom should not be SP2-hybridised"

    @staticmethod
    def test_is_SP3():
        # Test with an SP3-hybridised atom
        mol = Chem.MolFromSmiles("CCO")
        atom = mol.GetAtomWithIdx(0)
        result = PatternRecognition.Atoms.is_SP3(atom)
        assert result == 1, "Atom should be SP3-hybridised"

        # Test with a non-SP3-hybridised atom
        mol = Chem.MolFromSmiles("C=C")
        atom = mol.GetAtomWithIdx(0)
        result = PatternRecognition.Atoms.is_SP3(atom)
        assert result == 0, "Atom should not be SP3-hybridised"

    @staticmethod
    def test_get_ring_size():
        # Test with an atom in a 6-membered ring (benzene)
        mol = Chem.MolFromSmiles("c1ccccc1")
        atom = mol.GetAtomWithIdx(0)
        ring_size = PatternRecognition.Atoms.get_ring_size(atom)
        assert ring_size == 6, f"Expected ring size 6, got {ring_size}"

        # Test with an atom not in a ring (ethanol)
        mol = Chem.MolFromSmiles("CCO")
        atom = mol.GetAtomWithIdx(0)
        ring_size = PatternRecognition.Atoms.get_ring_size(atom)
        assert ring_size == 0, f"Expected ring size 0, got {ring_size}"

        # Test with an atom in a 3-membered ring (cyclopropane)
        mol = Chem.MolFromSmiles("C1CC1")
        atom = mol.GetAtomWithIdx(0)
        ring_size = PatternRecognition.Atoms.get_ring_size(atom)
        assert ring_size == 3, f"Expected ring size 3, got {ring_size}"

    # Bonds

    @staticmethod
    def test_is_single_bond():
        # Test with a single bond
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondWithIdx(0)
        result = PatternRecognition.Bonds.is_single_bond(bond)
        assert result == 1, "Bond should be a single bond"

        # Test with a non-single bond
        mol = Chem.MolFromSmiles("C=C")
        bond = mol.GetBondWithIdx(0)
        result = PatternRecognition.Bonds.is_single_bond(bond)
        assert result == 0, "Bond should not be a single bond"

    @staticmethod
    def test_is_double_bond():
        # Test with a double bond
        mol = Chem.MolFromSmiles("C=C")
        bond = mol.GetBondWithIdx(0)
        result = PatternRecognition.Bonds.is_double_bond(bond)
        assert result == 1, "Bond should be a double bond"

        # Test with a non-double bond
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondWithIdx(0)
        result = PatternRecognition.Bonds.is_double_bond(bond)
        assert result == 0, "Bond should not be a double bond"

    @staticmethod
    def test_is_triple_bond():
        # Test with a triple bond
        mol = Chem.MolFromSmiles("C#C")
        bond = mol.GetBondWithIdx(0)
        result = PatternRecognition.Bonds.is_triple_bond(bond)
        assert result == 1, "Bond should be a triple bond"

        # Test with a non-triple bond
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondWithIdx(0)
        result = PatternRecognition.Bonds.is_triple_bond(bond)
        assert result == 0, "Bond should not be a triple bond"

    @staticmethod
    def test_is_dative_bond():
        # Test with a dative bond
        mol = Chem.MolFromSmiles("[Fe]->[O]")
        bond = mol.GetBondWithIdx(0)
        result = PatternRecognition.Bonds.is_dative_bond(bond)
        assert result == 1, "Bond should be a dative bond"

        # Test with a non-dative bond
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondWithIdx(0)
        result = PatternRecognition.Bonds.is_dative_bond(bond)
        assert result == 0, "Bond should not be a dative bond"

    @staticmethod
    def test_check_bonds():
        # Test with a molecule containing bonds
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.Bonds.check_bonds(mol)
        assert result[0] is True, "Bonds should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "*~*", "SMARTS pattern does not match"

    @staticmethod
    def test_check_rotatable_bonds():
        # Test with a molecule containing rotatable bonds
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.Bonds.check_rotatable_bonds(mol)
        assert result[0] is True, "Rotatable bonds should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]", "SMARTS pattern does not match"

        mol = Chem.MolFromSmiles("C")
        result = PatternRecognition.Bonds.check_rotatable_bonds(mol)
        assert result[0] is False, "Rotatable bonds should not be found"
        assert len(result[1]) == 0, "Atom indices returned"
        assert result[2] == "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_single_bonds():
        # Test with a molecule containing single bonds
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.Bonds.check_single_bonds(mol)
        assert result[0] is True, "Single bonds should be found in the molecule"
        assert len(result[1]) == 3, "Number of atoms connected via single bonds was expected to be 3"
        assert result[2] == "*-*", "SMARTS pattern does not match"

        mol = Chem.MolFromSmiles("C")
        result = PatternRecognition.Bonds.check_single_bonds(mol)
        assert result[0] is False, "Single bonds should no be found"
        assert len(result[1]) == 0, "Number of atoms connected via single bonds was expected to be 0"
        assert result[2] == "*-*", "SMARTS pattern does not match"

    @staticmethod
    def test_check_double_bonds():
        # Test with a molecule containing double bonds
        mol = Chem.MolFromSmiles("C=C")
        result = PatternRecognition.Bonds.check_double_bonds(mol)
        assert result[0] is True, "Double bonds should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "*=*", "SMARTS pattern does not match"

        mol = Chem.MolFromSmiles("C-C")
        result = PatternRecognition.Bonds.check_double_bonds(mol)
        assert result[0] is False, "Double bonds should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices returned"
        assert result[2] == "*=*", "SMARTS pattern does not match"

    @staticmethod
    def test_check_triple_bonds():
        # Test with a molecule containing triple bonds
        mol = Chem.MolFromSmiles("C#C")
        result = PatternRecognition.Bonds.check_triple_bonds(mol)
        assert result[0] is True, "Triple bonds should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "*#*", "SMARTS pattern does not match"

        mol = Chem.MolFromSmiles("C=C")
        result = PatternRecognition.Bonds.check_triple_bonds(mol)
        assert result[0] is False, "Triple bonds should be not found in the molecule"
        assert len(result[1]) == 0, "Atom indices returned"
        assert result[2] == "*#*", "SMARTS pattern does not match"

    @staticmethod
    def test_check_aromatic_bonds():
        # Test with a molecule containing aromatic bonds
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = PatternRecognition.Bonds.check_aromatic_bonds(mol)
        assert result[0] is True, "Aromatic bonds should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "*:*", "SMARTS pattern does not match"

        mol = Chem.MolFromSmiles("C1CCCCC1")
        result = PatternRecognition.Bonds.check_aromatic_bonds(mol)
        assert result[0] is False, "Aromatic bonds should not be found in the molecule"
        assert len(result[1]) == 0, "No atom indices returned"
        assert result[2] == "*:*", "SMARTS pattern does not match"

    @staticmethod
    def test_check_cyclic_bonds():
        # Test with a molecule containing cyclic bonds
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = PatternRecognition.Bonds.check_cyclic_bonds(mol)
        assert result[0] is True, "Cyclic bonds should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "*@*", "SMARTS pattern does not match"

        mol = Chem.MolFromSmiles("CCCN")
        result = PatternRecognition.Bonds.check_cyclic_bonds(mol)
        assert result[0] is False, "Cyclic bonds should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices returned"
        assert result[2] == "*@*", "SMARTS pattern does not match"

    # MolPatterns

    @staticmethod
    def test_check_pattern_aromatic():
        # Test with a valid SMILES string and pattern function
        smiles = "c1ccccc1"
        pattern_function = PatternRecognition.MolPatterns.check_carbon
        result = PatternRecognition.MolPatterns.check_pattern_aromatic(smiles, pattern_function)
        assert result[0] is True, "Pattern not found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert len(result[2]) == 2

        smiles = "C1CCCCC1"
        pattern_function = PatternRecognition.MolPatterns.check_carbon
        result = PatternRecognition.MolPatterns.check_pattern_aromatic(smiles, pattern_function)
        assert result[0] is False, "Pattern found in the molecule"
        assert len(result[1]) == 0, "Atom indices returned"
        assert len(result[2]) == 2

    @staticmethod
    def test_check_pattern_aromatic_substituent():
        # Test with a valid SMILES string and pattern function
        smiles = "c1ccccc1O"
        pattern_function = PatternRecognition.MolPatterns.check_oxygen
        result = PatternRecognition.MolPatterns.check_pattern_aromatic_substituent(smiles, pattern_function)
        assert result[0] is True, "Pattern not found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert len(result[2]) == 2

        smiles = "c1ccccc1C"
        pattern_function = PatternRecognition.MolPatterns.check_oxygen
        result = PatternRecognition.MolPatterns.check_pattern_aromatic_substituent(smiles, pattern_function)
        assert result[0] is False, "Pattern found in the molecule"
        assert len(result[1]) == 0, "Atom indices returned"
        assert len(result[2]) == 2

    @staticmethod
    def test_check_pattern_aliphatic():
        # Test with a valid SMILES string and pattern function
        smiles = "CCO"
        pattern_function = PatternRecognition.MolPatterns.check_carbon
        result = PatternRecognition.MolPatterns.check_pattern_aliphatic(smiles, pattern_function)
        assert result[0] is True, "Pattern not found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert len(result[2]) == 2

        smiles = "c1ccccc1"
        pattern_function = PatternRecognition.MolPatterns.check_carbon
        result = PatternRecognition.MolPatterns.check_pattern_aliphatic(smiles, pattern_function)
        assert result[0] is False, "Pattern  found in the molecule"
        assert len(result[1]) == 0, "Atom indices returned"
        assert len(result[2]) == 2

    @staticmethod
    def test_check_carbon():
        # Test with a valid SMILES string
        smiles = "CCO"
        result = PatternRecognition.MolPatterns.check_carbon(smiles)
        assert result[0] is True, "Pattern not found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6]", "SMARTS pattern does not match"

        smiles = "NO"
        result = PatternRecognition.MolPatterns.check_carbon(smiles)
        assert result[0] is False, "Pattern found in the molecule"
        assert len(result[1]) == 0, "Atom indices returned"
        assert result[2] == "[#6]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_carbanion():
        # Test with a valid SMILES string
        smiles = "[CH2-]C"
        result = PatternRecognition.MolPatterns.check_carbanion(smiles)
        assert result[0] is True, "Pattern not found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6-]", "SMARTS pattern does not match"

        smiles = "CCC"
        result = PatternRecognition.MolPatterns.check_carbanion(smiles)
        assert result[0] is False, "Pattern found in the molecule"
        assert len(result[1]) == 0, "Atom indices returned"
        assert result[2] == "[#6-]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_carbocation():
        # Test with a valid SMILES string
        smiles = "[CH3+]"
        result = PatternRecognition.MolPatterns.check_carbocation(smiles)
        assert result[0] is True, "Pattern not found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6+]", "SMARTS pattern does not match"

        smiles = "CC"
        result = PatternRecognition.MolPatterns.check_carbocation(smiles)
        assert result[0] is False, "Pattern found in the molecule"
        assert len(result[1]) == 0, "Atom indices returned"
        assert result[2] == "[#6+]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_alkyl_carbon():
        # Test with a valid SMILES string
        smiles = "CC"
        result = PatternRecognition.MolPatterns.check_alkyl_carbon(smiles)
        assert result[0] is True, "Pattern not found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX4]", "SMARTS pattern does not match"

        smiles = "N"
        result = PatternRecognition.MolPatterns.check_alkyl_carbon(smiles)
        assert result[0] is False, "Pattern found in the molecule"
        assert len(result[1]) == 0, "Atom indices returned"
        assert result[2] == "[CX4]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_allenic_carbon():
        # Test with a valid SMILES string
        smiles = "C=C=C"
        result = PatternRecognition.MolPatterns.check_allenic_carbon(smiles)
        assert result[0] is True, "Pattern not found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([CX2](=C)=C)]", "SMARTS pattern does not match"

        smiles = "C=C"
        result = PatternRecognition.MolPatterns.check_allenic_carbon(smiles)
        assert result[0] is False, "Pattern found in the molecule"
        assert len(result[1]) == 0, "Atom indices returned"
        assert result[2] == "[$([CX2](=C)=C)]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_vinylic_carbon():
        # Test with a valid SMILES string
        smiles = "C=C"
        result = PatternRecognition.MolPatterns.check_vinylic_carbon(smiles)
        assert result[0] is True, "Pattern not found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([CX3]=[CX3])]", "SMARTS pattern does not match"

        smiles = "CC"
        result = PatternRecognition.MolPatterns.check_vinylic_carbon(smiles)
        assert result[0] is False, "Pattern found in the molecule"
        assert len(result[1]) == 0, "Atom indices returned"
        assert result[2] == "[$([CX3]=[CX3])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_acetylenic_carbon():
        # Test with a molecule containing acetylenic carbon atoms
        mol = Chem.MolFromSmiles("C#C")
        result = PatternRecognition.MolPatterns.check_acetylenic_carbon(mol)
        assert result[0] is True, "Acetylenic carbon atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([CX2]#C)]", "SMARTS pattern does not match"

        # Test with a molecule not containing acetylenic carbon atoms
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_acetylenic_carbon(mol)
        assert result[0] is False, "Acetylenic carbon atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([CX2]#C)]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_carbonyl():
        # Test with a molecule containing carbonyl groups
        mol = Chem.MolFromSmiles("CC(=O)C")
        result = PatternRecognition.MolPatterns.check_carbonyl(mol)
        assert result[0] is True, "Carbonyl groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]", "SMARTS pattern does not match"

        # Test with a molecule not containing carbonyl groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_carbonyl(mol)
        assert result[0] is False, "Carbonyl groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_acyl_halide():
        # Test with a molecule containing acyl halides
        mol = Chem.MolFromSmiles("CC(=O)Cl")
        result = PatternRecognition.MolPatterns.check_acyl_halide(mol)
        assert result[0] is True, "Acyl halides should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX3](=[OX1])[F,Cl,Br,I]", "SMARTS pattern does not match"

        # Test with a molecule not containing acyl halides
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_acyl_halide(mol)
        assert result[0] is False, "Acyl halides should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX3](=[OX1])[F,Cl,Br,I]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_aldehyde():
        # Test with a molecule containing aldehyde groups
        mol = Chem.MolFromSmiles("CC=O")
        result = PatternRecognition.MolPatterns.check_aldehyde(mol)
        assert result[0] is True, "Aldehyde groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX3H1](=O)[#6]", "SMARTS pattern does not match"

        # Test with a molecule not containing aldehyde groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_aldehyde(mol)
        assert result[0] is False, "Aldehyde groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX3H1](=O)[#6]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_anhydride():
        # Test with a molecule containing anhydride groups
        mol = Chem.MolFromSmiles("CC(=O)OC(=O)C")
        result = PatternRecognition.MolPatterns.check_anhydride(mol)
        assert result[0] is True, "Anhydride groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX3](=[OX1])[OX2][CX3](=[OX1])", "SMARTS pattern does not match"

        # Test with a molecule not containing anhydride groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_anhydride(mol)
        assert result[0] is False, "Anhydride groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX3](=[OX1])[OX2][CX3](=[OX1])", "SMARTS pattern does not match"

    @staticmethod
    def test_check_carboxyl():
        # Test with a molecule containing carboxyl groups
        mol = Chem.MolFromSmiles("CC(=O)O")
        result = PatternRecognition.MolPatterns.check_carboxyl(mol)
        assert result[0] is True, "Carboxyl groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX3](=O)[OX1H0-,OX2H1]", "SMARTS pattern does not match"

        # Test with a molecule not containing carboxyl groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_carboxyl(mol)
        assert result[0] is False, "Carboxyl groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX3](=O)[OX1H0-,OX2H1]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_carbonic_acid():
        # Test with a molecule containing carbonic acid groups
        mol = Chem.MolFromSmiles("O=C(O)O")
        result = PatternRecognition.MolPatterns.check_carbonic_acid(mol)
        assert result[0] is True, "Carbonic acid groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX3](=[OX1])([OX2])[OX2H,OX1H0-1]", "SMARTS pattern does not match"

        # Test with a molecule not containing carbonic acid groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_carbonic_acid(mol)
        assert result[0] is False, "Carbonic acid groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX3](=[OX1])([OX2])[OX2H,OX1H0-1]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_carbonate_ester():
        # Test with a molecule containing carbonate esters
        mol = Chem.MolFromSmiles("O=C(OCC)OC")
        result = PatternRecognition.MolPatterns.check_carbonate_ester(mol)
        assert result[0] is True, "Carbonate esters should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == ("[CX3](=[OX1])([OX2H0])[OX2H,OX1H0-1]", "[CX3](=[OX1])([OX2H0])[OX2H0]"), "SMARTS patterns do not match"

        # Test with a molecule not containing carbonate esters
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_carbonate_ester(mol)
        assert result[0] is False, "Carbonate esters should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == ("[CX3](=[OX1])([OX2H0])[OX2H,OX1H0-1]", "[CX3](=[OX1])([OX2H0])[OX2H0]"), "SMARTS patterns do not match"

    @staticmethod
    def test_check_ester():
        # Test with a molecule containing ester groups
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        result = PatternRecognition.MolPatterns.check_ester(mol)
        assert result[0] is True, "Ester groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6][CX3](=O)[OX2H0][#6]", "SMARTS pattern does not match"

        # Test with a molecule not containing ester groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_ester(mol)
        assert result[0] is False, "Ester groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#6][CX3](=O)[OX2H0][#6]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_ketone():
        # Test with a molecule containing ketone groups
        mol = Chem.MolFromSmiles("CC(=O)C")
        result = PatternRecognition.MolPatterns.check_ketone(mol)
        assert result[0] is True, "Ketone groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6][CX3](=O)[#6]", "SMARTS pattern does not match"

        # Test with a molecule not containing ketone groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_ketone(mol)
        assert result[0] is False, "Ketone groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#6][CX3](=O)[#6]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_ether():
        # Test with a molecule containing ether groups
        mol = Chem.MolFromSmiles("CCOCC")
        result = PatternRecognition.MolPatterns.check_ether(mol)
        assert result[0] is True, "Ether groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[OD2]([#6])[#6]", "SMARTS pattern does not match"

        # Test with a molecule not containing ether groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_ether(mol)
        assert result[0] is False, "Ether groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[OD2]([#6])[#6]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_alpha_diketone():
        # Test with a molecule containing alpha-diketone groups
        mol = Chem.MolFromSmiles("O=C(C)C(C)=O")
        result = PatternRecognition.MolPatterns.check_alpha_diketone(mol)
        assert result[0] is True, "Alpha-diketone groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "O=[#6D3]([#6])[#6D3]([#6])=O", "SMARTS pattern does not match"

        # Test with a molecule not containing alpha-diketone groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_alpha_diketone(mol)
        assert result[0] is False, "Alpha-diketone groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "O=[#6D3]([#6])[#6D3]([#6])=O", "SMARTS pattern does not match"

    @staticmethod
    def test_check_beta_diketone():
        # Test with a molecule containing beta-diketone groups
        mol = Chem.MolFromSmiles("O=C(C)CC(C)=O")
        result = PatternRecognition.MolPatterns.check_beta_diketone(mol)
        assert result[0] is True, "Beta-diketone groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "O=[#6D3]([#6])[#6][#6D3]([#6])=O", "SMARTS pattern does not match"

        # Test with a molecule not containing beta-diketone groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_beta_diketone(mol)
        assert result[0] is False, "Beta-diketone groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "O=[#6D3]([#6])[#6][#6D3]([#6])=O", "SMARTS pattern does not match"

    @staticmethod
    def test_check_gamma_diketone():
        # Test with a molecule containing gamma-diketone groups
        mol = Chem.MolFromSmiles("O=C(C)CCC(C)=O")
        result = PatternRecognition.MolPatterns.check_gamma_diketone(mol)
        assert result[0] is True, "Gamma-diketone groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "O=[#6D3]([#6])[#6][#6][#6D3]([#6])=O", "SMARTS pattern does not match"

        # Test with a molecule not containing gamma-diketone groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_gamma_diketone(mol)
        assert result[0] is False, "Gamma-diketone groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "O=[#6D3]([#6])[#6][#6][#6D3]([#6])=O", "SMARTS pattern does not match"

    @staticmethod
    def test_check_alpha_dicarbonyl():
        # Test with a molecule containing alpha-dicarbonyl groups
        mol = Chem.MolFromSmiles("O=CC=O")
        result = PatternRecognition.MolPatterns.check_alpha_dicarbonyl(mol)
        assert result[0] is True, "Alpha-dicarbonyl groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "O=[#6][#6]=O", "SMARTS pattern does not match"

        # Test with a molecule not containing alpha-dicarbonyl groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_alpha_dicarbonyl(mol)
        assert result[0] is False, "Alpha-dicarbonyl groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "O=[#6][#6]=O", "SMARTS pattern does not match"

    @staticmethod
    def test_check_beta_dicarbonyl():
        # Test with a molecule containing beta-dicarbonyl groups
        mol = Chem.MolFromSmiles("O=CCC=O")
        result = PatternRecognition.MolPatterns.check_beta_dicarbonyl(mol)
        assert result[0] is True, "Beta-dicarbonyl groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "O=[#6][#6][#6]=O", "SMARTS pattern does not match"

        # Test with a molecule not containing beta-dicarbonyl groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_beta_dicarbonyl(mol)
        assert result[0] is False, "Beta-dicarbonyl groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "O=[#6][#6][#6]=O", "SMARTS pattern does not match"

    @staticmethod
    def test_check_gamma_dicarbonyl():
        # Test with a molecule containing gamma-dicarbonyl groups
        mol = Chem.MolFromSmiles("O=CCCC=O")
        result = PatternRecognition.MolPatterns.check_gamma_dicarbonyl(mol)
        assert result[0] is True, "Gamma-dicarbonyl groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "O=[#6][#6][#6][#6]=O", "SMARTS pattern does not match"

        # Test with a molecule not containing gamma-dicarbonyl groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_gamma_dicarbonyl(mol)
        assert result[0] is False, "Gamma-dicarbonyl groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "O=[#6][#6][#6][#6]=O", "SMARTS pattern does not match"

    @staticmethod
    def test_check_delta_dicarbonyl():
        # Test with a molecule containing delta-dicarbonyl groups
        mol = Chem.MolFromSmiles("O=CCCCC=O")
        result = PatternRecognition.MolPatterns.check_delta_dicarbonyl(mol)
        assert result[0] is True, "Delta-dicarbonyl groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "O=[#6][#6][#6][#6][#6]=O", "SMARTS pattern does not match"

        # Test with a molecule not containing delta-dicarbonyl groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_delta_dicarbonyl(mol)
        assert result[0] is False, "Delta-dicarbonyl groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "O=[#6][#6][#6][#6][#6]=O", "SMARTS pattern does not match"

    # N

    @staticmethod
    def test_check_nitrogen():
        # Test with a molecule containing nitrogen atoms
        mol = Chem.MolFromSmiles("CCN")
        result = PatternRecognition.MolPatterns.check_nitrogen(mol)
        assert result[0] is True, "Nitrogen atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#7]", "SMARTS pattern does not match"

        # Test with a molecule not containing nitrogen atoms
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_nitrogen(mol)
        assert result[0] is False, "Nitrogen atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#7]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_amine():
        # Test with a molecule containing amine groups
        mol = Chem.MolFromSmiles("CCN")
        result = PatternRecognition.MolPatterns.check_amine(mol)
        assert result[0] is True, "Amine groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == (('[#7]', '[#6][#7D1&!$(NC=O)]'),('[#7]', '[#6][#7D2&!$(NC=O)][#6]'),('[#7]', '[#6][#7D3&!$(NC=O)]([#6])[#6]'),('[#7]', '[#6][#7D4+&!$(NC=O)]([#6])([#6])([#6])')), "SMARTS patterns do not match"

        # Test with a molecule not containing amine groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_amine(mol)
        assert result[0] is False, "Amine groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == (('[#7]', '[#6][#7D1&!$(NC=O)]'),('[#7]', '[#6][#7D2&!$(NC=O)][#6]'),('[#7]', '[#6][#7D3&!$(NC=O)]([#6])[#6]'),('[#7]', '[#6][#7D4+&!$(NC=O)]([#6])([#6])([#6])')), "SMARTS patterns do not match"

    @staticmethod
    def test_check_amine_primary():
        # Test with a molecule containing primary amine groups
        mol = Chem.MolFromSmiles("CCN")
        result = PatternRecognition.MolPatterns.check_amine_primary(mol)
        assert result[0] is True, "Primary amine groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == ('[#7]', '[#6][#7D1&!$(NC=O)]'), "SMARTS patterns do not match"

        # Test with a molecule not containing primary amine groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_amine_primary(mol)
        assert result[0] is False, "Primary amine groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == ("[#7]", "[#6][#7D1&!$(NC=O)]"), "SMARTS patterns do not match"

    @staticmethod
    def test_check_amine_secondary():
        # Test with a molecule containing secondary amine groups
        mol = Chem.MolFromSmiles("CCCCCCNCC")
        result = PatternRecognition.MolPatterns.check_amine_secondary(mol)
        assert result[0] is True, "Secondary amine groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == ("[#7]", "[#6][#7D2&!$(NC=O)][#6]"), "SMARTS patterns do not match"

        # Test with a molecule not containing secondary amine groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_amine_secondary(mol)
        assert result[0] is False, "Secondary amine groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == ("[#7]", "[#6][#7D2&!$(NC=O)][#6]"), "SMARTS patterns do not match"

    @staticmethod
    def test_check_amine_tertiary():
        # Test with a molecule containing tertiary amine groups
        mol = Chem.MolFromSmiles("CC1=NC=C(N1CCO)[N+](=O)[O-]")
        result = PatternRecognition.MolPatterns.check_amine_tertiary(mol)
        assert result[0] is True, "Tertiary amine groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == ('[#7]', '[#6][#7D3&!$(NC=O)]([#6])[#6]'), "SMARTS patterns do not match"

        # Test with a molecule not containing tertiary amine groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_amine_tertiary(mol)
        assert result[0] is False, "Tertiary amine groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == ('[#7]', '[#6][#7D3&!$(NC=O)]([#6])[#6]'), "SMARTS patterns do not match"

    @staticmethod
    def test_check_amine_quaternary():
        # Test with a molecule containing quaternary amine groups
        mol = Chem.MolFromSmiles("C[N+](C)(C)CC(=O)[O-]")
        result = PatternRecognition.MolPatterns.check_amine_quaternary(mol)
        assert result[0] is True, "Quaternary amine groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == ('[#7]', '[#6][#7D4+&!$(NC=O)]([#6])([#6])([#6])'), "SMARTS patterns do not match"

        # Test with a molecule not containing quaternary amine groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_amine_quaternary(mol)
        assert result[0] is False, "Quaternary amine groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == ('[#7]', '[#6][#7D4+&!$(NC=O)]([#6])([#6])([#6])'), "SMARTS patterns do not match"

    @staticmethod
    def test_check_enamine():
        # Test with a molecule containing enamine groups
        mol = Chem.MolFromSmiles("CN(C=C)N=O")
        result = PatternRecognition.MolPatterns.check_enamine(mol)
        assert result[0] is True, "Enamine groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[NX3][CX3]=[CX3]", "SMARTS pattern does not match"

        # Test with a molecule not containing enamine groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_enamine(mol)
        assert result[0] is False, "Enamine groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[NX3][CX3]=[CX3]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_amide():
        # Test with a molecule containing amide groups
        mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        result = PatternRecognition.MolPatterns.check_amide(mol)
        assert result[0] is True, "Amide groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#7X3][#6X3](=[OX1])[#6]", "SMARTS pattern does not match"

        # Test with a molecule not containing amide groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_amide(mol)
        assert result[0] is False, "Amide groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#7X3][#6X3](=[OX1])[#6]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_carbamate():
        # Test with a molecule containing carbamate groups
        mol = Chem.MolFromSmiles("CC(C)OC1=CC=CC=C1OC(=O)NC")
        result = PatternRecognition.MolPatterns.check_carbamate(mol)
        assert result[0] is True, "Carbamate groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]", "SMARTS pattern does not match"

        # Test with a molecule not containing carbamate groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_carbamate(mol)
        assert result[0] is False, "Carbamate groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]", "SMARTS pattern does not match"

    @staticmethod
    def test_alpha_nitroalkane():
        # Test with a molecule containing alpha-nitroalkane groups
        mol = Chem.MolFromSmiles("CCC[N+](=O)[O-]")
        result = PatternRecognition.MolPatterns.alpha_nitroalkane(mol)
        assert result[0] is True, "Alpha-nitroalkane groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX4H1,H2,H3][#7D3](=[#8])[#8]", "SMARTS pattern does not match"

        # Test with a molecule not containing alpha-nitroalkane groups
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.alpha_nitroalkane(mol)
        assert result[0] is False, "Alpha-nitroalkane groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX4H1,H2,H3][#7D3](=[#8])[#8]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_alanine():
        # Test with a molecule containing alanine residues
        mol = Chem.MolFromSmiles("C[C@@H](C(=O)O)N")
        result = PatternRecognition.MolPatterns.check_alanine(mol)
        assert result[0] is True, "Alanine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with a molecule not containing alanine residues
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_alanine(mol)
        assert result[0] is False, "Alanine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_arginine():
        # Test with a molecule containing arginine residues
        mol = Chem.MolFromSmiles("C(C[C@@H](C(=O)O)N)CN=C(N)N")
        result = PatternRecognition.MolPatterns.check_arginine(mol)
        assert result[0] is True, "Arginine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX3](=[OX1])([OX2])[CH1X4]([NX3])[CH2X4][CH2X4][CH2X4][ND2]=[CD3]([NX3])[NX3]", "SMARTS pattern does not match"

        # Test with a molecule not containing arginine residues
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_arginine(mol)
        assert result[0] is False, "Arginine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX3](=[OX1])([OX2])[CH1X4]([NX3])[CH2X4][CH2X4][CH2X4][ND2]=[CD3]([NX3])[NX3]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_asparagine():
        # Test with a molecule containing asparagine residues
        mol = Chem.MolFromSmiles("C([C@@H](C(=O)O)N)C(=O)N")
        result = PatternRecognition.MolPatterns.check_asparagine(mol)
        assert result[0] is True, "Asparagine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CX3](=[OX1])[NX3H2])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with a molecule not containing asparagine residues
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_asparagine(mol)
        assert result[0] is False, "Asparagine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CX3](=[OX1])[NX3H2])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_aspartate():
        # Test with a molecule containing aspartate residues
        mol = Chem.MolFromSmiles("C([C@@H](C(=O)O)N)C(=O)O")
        result = PatternRecognition.MolPatterns.check_aspartate(mol)
        assert result[0] is True, "Aspartate residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CX3](=[OX1])[OH0-,OH])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with a molecule not containing aspartate residues
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_aspartate(mol)
        assert result[0] is False, "Aspartate residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CX3](=[OX1])[OH0-,OH])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_cysteine():
        # Test with a molecule containing cysteine residues
        mol = Chem.MolFromSmiles("C([C@@H](C(=O)O)N)S")
        result = PatternRecognition.MolPatterns.check_cysteine(mol)
        assert result[0] is True, "Cysteine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][SX2H,SX1H0-])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with a molecule not containing cysteine residues
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_cysteine(mol)
        assert result[0] is False, "Cysteine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][SX2H,SX1H0-])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_glutamate():
        # Test with a molecule containing glutamate residues
        mol = Chem.MolFromSmiles("C(CC(=O)O)[C@@H](C(=O)O)N")
        result = PatternRecognition.MolPatterns.check_glutamate(mol)
        assert result[0] is True, "Glutamate residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with a molecule not containing glutamate residues
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_glutamate(mol)
        assert result[0] is False, "Glutamate residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"
    
    @staticmethod
    def test_check_glycine():
        # Test with a molecule containing glycine residues
        mol = Chem.MolFromSmiles("CC(C)C(=O)NCC(=O)O")
        result = PatternRecognition.MolPatterns.check_glycine(mol)
        assert result[0] is True, "Glycine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N])]", "SMARTS pattern does not match"

        # Test with a molecule not containing glycine residues
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_glycine(mol)
        assert result[0] is False, "Glycine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_histidine():
        # Test with a molecule containing histidine residues
        mol = Chem.MolFromSmiles("C1=C(NC=N1)C[C@@H](C(=O)O)N")
        result = PatternRecognition.MolPatterns.check_histidine(mol)
        assert result[0] is True, "Histidine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1)[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with a molecule not containing histidine residues
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_histidine(mol)
        assert result[0] is False, "Histidine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1)[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_isoleucine():
        # Test with a molecule containing isoleucine residues
        mol = Chem.MolFromSmiles("CC[C@H](C)[C@@H](C(=O)O)N")
        result = PatternRecognition.MolPatterns.check_isoleucine(mol)
        assert result[0] is True, "Isoleucine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CHX4]([CH3X4])[CH2X4][CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with a molecule not containing isoleucine residues
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_isoleucine(mol)
        assert result[0] is False, "Isoleucine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CHX4]([CH3X4])[CH2X4][CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_leucine():
        # Test with a molecule containing leucine residues
        mol = Chem.MolFromSmiles("CC(C)CC(C(=O)O)N")
        result = PatternRecognition.MolPatterns.check_leucine(mol)
        assert result[0] is True, "Leucine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CHX4]([CH3X4])[CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with a molecule not containing leucine residues
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_leucine(mol)
        assert result[0] is False, "Leucine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CHX4]([CH3X4])[CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_lysine():
        # Test with a molecule containing lysine residues
        mol = Chem.MolFromSmiles("CC(C)CCCC(C(=O)O)N")
        result = PatternRecognition.MolPatterns.check_lysine(mol)
        assert result[0] is True, "Lysine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX3](=[OX1])([OX2])[CH1X4]([NX3])[CH2X4][CH2X4][CH2X4][CD3]([CD1])[CD1]", "SMARTS pattern does not match"

        # Test with a molecule not containing lysine residues
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_lysine(mol)
        assert result[0] is False, "Lysine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX3](=[OX1])([OX2])[CH1X4]([NX3])[CH2X4][CH2X4][CH2X4][CD3]([CD1])[CD1]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_methionine():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CSCCC(C(=O)O)N")
        result = PatternRecognition.MolPatterns.check_methionine(mol)
        assert result[0] is True, "Methionine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CH2X4][SX2][CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCC(C(=O)O)N ")
        result = PatternRecognition.MolPatterns.check_methionine(mol)
        assert result[0] is False, "Methionine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][CH2X4][SX2][CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_phenylalanine():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC=C(C=C1)C[C@@H](C(=O)O)N")
        result = PatternRecognition.MolPatterns.check_phenylalanine(mol)
        assert result[0] is True, "Phenylalanine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1)[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1=CC=C(C=C1)C[C@@H](C(=O)O)C")
        result = PatternRecognition.MolPatterns.check_phenylalanine(mol)
        assert result[0] is False, "Phenylalanine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1)[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_proline():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1C[C@H](NC1)C(=O)O")
        result = PatternRecognition.MolPatterns.check_proline(mol)
        assert result[0] is True, "Proline residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1C[C@H](NC1)C(=O)C")
        result = PatternRecognition.MolPatterns.check_proline(mol)
        assert result[0] is False, "Proline residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_serine():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C([C@@H](C(=O)O)N)O")
        result = PatternRecognition.MolPatterns.check_serine(mol)
        assert result[0] is True, "Serine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][OX2H])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C([C@@H](C(=O)O)N)C")
        result = PatternRecognition.MolPatterns.check_serine(mol)
        assert result[0] is False, "Serine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][OX2H])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_threonine():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C[C@H]([C@@H](C(=O)O)N)O")
        result = PatternRecognition.MolPatterns.check_threonine(mol)
        assert result[0] is True, "Threonine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CHX4]([CH3X4])[OX2H])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C[C@H]([C@@H](CC(=O)O)N)O")
        result = PatternRecognition.MolPatterns.check_threonine(mol)
        assert result[0] is False, "Threonine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CHX4]([CH3X4])[OX2H])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_tryptophan():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N")
        result = PatternRecognition.MolPatterns.check_tryptophan(mol)
        assert result[0] is True, "Tryptophan residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12)[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1=CC=C2C(=C1)CC(=CN2)C[C@@H](C(=O)O)N")
        result = PatternRecognition.MolPatterns.check_tryptophan(mol)
        assert result[0] is False, "Tryptophan residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12)[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_tyrosine():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC(=CC=C1C[C@@H](C(=O)O)N)O")
        result = PatternRecognition.MolPatterns.check_tyrosine(mol)
        assert result[0] is True, "Tyrosine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1)[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCN")
        result = PatternRecognition.MolPatterns.check_tyrosine(mol)
        assert result[0] is False, "Tyrosine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1)[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"
    
    @staticmethod
    def test_check_valine():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CC(C)[C@@H](C(=O)O)N")
        result = PatternRecognition.MolPatterns.check_valine(mol)
        assert result[0] is True, "Valine residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CHX4]([CH3X4])[CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CC(C)[C@@H](C(=O)C)N")
        result = PatternRecognition.MolPatterns.check_valine(mol)
        assert result[0] is False, "Valine residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CHX4]([CH3X4])[CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_aminoacid():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CC(C)[C@@H](C(=O)O)N")
        result = PatternRecognition.MolPatterns.check_aminoacid(mol)
        assert result[0] is True, "Aminoacid residues should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert len(result[2]) == 72, "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("OCCCCCN")
        result = PatternRecognition.MolPatterns.check_aminoacid(mol)
        assert result[0] is False, "Aminoacid residues should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert len(result[2]) == 0, "SMARTS pattern does not match"

    @staticmethod
    def test_check_azide():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC=C(C=C1)CN=[N+]=[N-]")
        result = PatternRecognition.MolPatterns.check_azide(mol)
        assert result[0] is True, "Azide groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCNCN")
        result = PatternRecognition.MolPatterns.check_azide(mol)
        assert result[0] is False, "Azide groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_azo():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CN=NC")
        result = PatternRecognition.MolPatterns.check_azo(mol)
        assert result[0] is True, "Azo groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6][#7D2]=[#7D2][#6]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CNNN")
        result = PatternRecognition.MolPatterns.check_azo(mol)
        assert result[0] is False, "Azo groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#6][#7D2]=[#7D2][#6]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_azoxy():
        # Test with a valid input
        mol = Chem.MolFromSmiles(r"CC(=O)OC/N=[N+](/C)\[O-] ")
        result = PatternRecognition.MolPatterns.check_azoxy(mol)
        assert result[0] is True, "Azoxy groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCNO")
        result = PatternRecognition.MolPatterns.check_azoxy(mol)
        assert result[0] is False, "Azoxy groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_diazo():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C=[N+]=[N-]")
        result = PatternRecognition.MolPatterns.check_diazo(mol)
        assert result[0] is True, "Diazo groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CNN")
        result = PatternRecognition.MolPatterns.check_diazo(mol)
        assert result[0] is False, "Diazo groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_hydrazine():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC=C(C=C1)CCNN")
        result = PatternRecognition.MolPatterns.check_hydrazine(mol)
        assert result[0] is True, "Hydrazine groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[NX3][NX3]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1=CC=C(C=C1)CCN")
        result = PatternRecognition.MolPatterns.check_hydrazine(mol)
        assert result[0] is False, "Hydrazine groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[NX3][NX3]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_hydrazone():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC(=CC(=C1)Cl)NN=C(C#N)C#N")
        result = PatternRecognition.MolPatterns.check_hydrazone(mol)
        assert result[0] is True, "Hydrazone groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#7X3][#7D2]=[#6D3]([#6])[#6]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCN")
        result = PatternRecognition.MolPatterns.check_hydrazone(mol)
        assert result[0] is False, "Hydrazone groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#7X3][#7D2]=[#6D3]([#6])[#6]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_imine():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CC(C)=NC")
        result = PatternRecognition.MolPatterns.check_imine(mol)
        assert result[0] is True, "Imine groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCNC")
        result = PatternRecognition.MolPatterns.check_imine(mol)
        assert result[0] is False, "Imine groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_iminium():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CN(C)C1=CC=C(C=C1)C(=C2C=CC(=[N+](C)C)C=C2)C3=CC=C(C=C3)N(C)C")
        result = PatternRecognition.MolPatterns.check_iminium(mol)
        assert result[0] is True, "Iminium groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[NX3+]=[CX3]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCN")
        result = PatternRecognition.MolPatterns.check_iminium(mol)
        assert result[0] is False, "Iminium groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[NX3+]=[CX3]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_imide():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1CC(=O)NC(=O)C1N2C(=O)C3=C(C2=O)C(=CC=C3)N")
        result = PatternRecognition.MolPatterns.check_imide(mol)
        assert result[0] is True, "Imide groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6][#6D3](=[#8D1])[#7X3][#6D3](=[#8D1])[#6]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCNCCO")
        result = PatternRecognition.MolPatterns.check_imide(mol)
        assert result[0] is False, "Imide groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#6][#6D3](=[#8D1])[#7X3][#6D3](=[#8D1])[#6]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_nitrate():
        # Test with a valid input
        mol = Chem.MolFromSmiles("[N+](=O)([O-])[O-].[Ag+]")
        result = PatternRecognition.MolPatterns.check_nitrate(mol)
        assert result[0] is True, "Nitrate groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1=CC(=CC=C1[N+](=S)[O-])Cl")
        result = PatternRecognition.MolPatterns.check_nitrate(mol)
        assert result[0] is False, "Nitrate groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_nitro():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCOP(=S)(OCC)OC1=CC=C(C=C1)[N+](=O)[O-]")
        result = PatternRecognition.MolPatterns.check_nitro(mol)
        assert result[0] is True, "Nitro groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CNCC")
        result = PatternRecognition.MolPatterns.check_nitro(mol)
        assert result[0] is False, "Nitro groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_nitrile():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C(#N)C1=C(C(=C(C(=C1Cl)Cl)Cl)C#N)Cl  ")
        result = PatternRecognition.MolPatterns.check_nitrile(mol)
        assert result[0] is True, "Nitrile groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[NX1]#[CX2]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCN")
        result = PatternRecognition.MolPatterns.check_nitrile(mol)
        assert result[0] is False, "Nitrile groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[NX1]#[CX2]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_isonitrile():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CC1=CC=C(C=C1)S(=O)(=O)C[N+]#[C-]")
        result = PatternRecognition.MolPatterns.check_isonitrile(mol)
        assert result[0] is True, "Isonitrile groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX1-]#[NX2+]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCCN")
        result = PatternRecognition.MolPatterns.check_isonitrile(mol)
        assert result[0] is False, "Isonitrile groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX1-]#[NX2+]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_nitroso():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CN(C)C1=CC=C(C=C1)N=O")
        result = PatternRecognition.MolPatterns.check_nitroso(mol)
        assert result[0] is True, "Nitroso groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[NX2]=[OX1]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCCN")
        result = PatternRecognition.MolPatterns.check_nitroso(mol)
        assert result[0] is False, "Nitroso groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[NX2]=[OX1]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_n_oxide():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CN(CCCC(=O)C1=C[N+](=CC=C1)[O-])N=O")
        result = PatternRecognition.MolPatterns.check_n_oxide(mol)
        assert result[0] is True, "N-oxide groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([#7X3H1,#7X3&!#7X3H2,#7X3H0,#7X4+][#8]);!$([#7](~[O])~[O]);!$([#7]=[#7])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCNCO")
        result = PatternRecognition.MolPatterns.check_n_oxide(mol)
        assert result[0] is False, "N-oxide groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([#7X3H1,#7X3&!#7X3H2,#7X3H0,#7X4+][#8]);!$([#7](~[O])~[O]);!$([#7]=[#7])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_cyanamide():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCC(C)(C)N=C(NC#N)NC1=CN=CC=C1 ")
        result = PatternRecognition.MolPatterns.check_cyanamide(mol)
        assert result[0] is True, "Cyanamide groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[NX3][CX2]#[NX1]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCNCO")
        result = PatternRecognition.MolPatterns.check_cyanamide(mol)
        assert result[0] is False, "Cyanamide groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[NX3][CX2]#[NX1]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_cyanate():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CC(C1=CC=C(C=C1)OC#N)C2=CC=C(C=C2)OC#N")
        result = PatternRecognition.MolPatterns.check_cyanate(mol)
        assert result[0] is True, "Cyanate groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#8D2][#6D2]#[#7D1]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCN")
        result = PatternRecognition.MolPatterns.check_cyanate(mol)
        assert result[0] is False, "Cyanate groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#8D2][#6D2]#[#7D1]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_isocyanate():
        # Test with a valid input
        mol = Chem.MolFromSmiles("COC1=C(C=CC(=C1)C2=CC(=C(C=C2)N=C=O)OC)N=C=O")
        result = PatternRecognition.MolPatterns.check_isocyanate(mol)
        assert result[0] is True, "Isocyanate groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#7D2]=[#6D2]=[#8D1]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCN")
        result = PatternRecognition.MolPatterns.check_isocyanate(mol)
        assert result[0] is False, "Isocyanate groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#7D2]=[#6D2]=[#8D1]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_oxygen():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_oxygen(mol)
        assert result[0] is True, "Oxygen atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#8]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCN")
        result = PatternRecognition.MolPatterns.check_oxygen(mol)
        assert result[0] is False, "Oxygen atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#8]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_alcohol():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCCO")
        result = PatternRecognition.MolPatterns.check_alcohol(mol)
        assert result[0] is True, "Alcohol groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6][OX2H]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCC(=O)C")
        result = PatternRecognition.MolPatterns.check_alcohol(mol)
        assert result[0] is False, "Alcohol groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#6][OX2H]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_enol():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C=CO")
        result = PatternRecognition.MolPatterns.check_enol(mol)
        assert result[0] is True, "Enol groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([OX2H][#6X3]=[#6]),$([OX1-][#6X3]=[#6])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCO")
        result = PatternRecognition.MolPatterns.check_enol(mol)
        assert result[0] is False, "Enol groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([OX2H][#6X3]=[#6]),$([OX1-][#6X3]=[#6])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_phosphorus():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCOC(=O)CC(C(=O)OCC)SP(=S)(OC)OC  ")
        result = PatternRecognition.MolPatterns.check_phosphorus(mol)
        assert result[0] is True, "Phosphorus atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[P]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCC")
        result = PatternRecognition.MolPatterns.check_phosphorus(mol)
        assert result[0] is False, "Phosphorus atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[P]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_phosphoric_acid():
        # Test with a valid input
        mol = Chem.MolFromSmiles("OP(=O)(O)O")
        result = PatternRecognition.MolPatterns.check_phosphoric_acid(mol)
        assert result[0] is True, "Phosphoric acid groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCCPO")
        result = PatternRecognition.MolPatterns.check_phosphoric_acid(mol)
        assert result[0] is False, "Phosphoric acid groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_phosphoric_ester():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)O)O)O)N")
        result = PatternRecognition.MolPatterns.check_phosphoric_ester(mol)
        assert result[0] is True, "Phosphoric ester groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == ("[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),"
        "$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)]),"
        "$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),"
        "$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]"), "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("OP(=O)(O)O")
        result = PatternRecognition.MolPatterns.check_phosphoric_ester(mol)
        assert result[0] is False, "Phosphoric ester groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == ("[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),"
        "$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)]),"
        "$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),"
        "$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]"), "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphur():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCCS")
        result = PatternRecognition.MolPatterns.check_sulphur(mol)
        assert result[0] is True, "Sulphur atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#16]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCN")
        result = PatternRecognition.MolPatterns.check_sulphur(mol)
        assert result[0] is False, "Sulphur atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#16]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_thiol():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCCCCS")
        result = PatternRecognition.MolPatterns.check_thiol(mol)
        assert result[0] is True, "Thiol groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#16X2H]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("S")
        result = PatternRecognition.MolPatterns.check_thiol(mol)
        assert result[0] is False, "Thiol groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#16X2H]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_thiocarbonyl():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C(=S)(N)N")
        result = PatternRecognition.MolPatterns.check_thiocarbonyl(mol)
        assert result[0] is True, "Thiocarbonyl groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6X3]=[#16X1]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCS")
        result = PatternRecognition.MolPatterns.check_thiocarbonyl(mol)
        assert result[0] is False, "Thiocarbonyl groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#6X3]=[#16X1]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_thioketone():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CC1=CC(=S)C2=CC=CC=C2N1")
        result = PatternRecognition.MolPatterns.check_thioketone(mol)
        assert result[0] is True, "Thioketone groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6][#6D3]([#6])=[#16X1]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCS")
        result = PatternRecognition.MolPatterns.check_thioketone(mol)
        assert result[0] is False, "Thioketone groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#6][#6D3]([#6])=[#16X1]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_thioaldehyde():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C=CC=S ")
        result = PatternRecognition.MolPatterns.check_thioaldehyde(mol)
        assert result[0] is True, "Thioaldehyde groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6][#6X3H1]=[#16X1]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCS")
        result = PatternRecognition.MolPatterns.check_thioaldehyde(mol)
        assert result[0] is False, "Thioaldehyde groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#6][#6X3H1]=[#16X1]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_thioanhydride():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC=C(C=C1)C(=O)SC(=O)C2=CC=CC=C2")
        result = PatternRecognition.MolPatterns.check_thioanhydride(mol)
        assert result[0] is True, "Thioanhydride groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX3](=[OX1])[SX2][CX3](=[OX1])", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCC(=S)CC")
        result = PatternRecognition.MolPatterns.check_thioanhydride(mol)
        assert result[0] is False, "Thioanhydride groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX3](=[OX1])[SX2][CX3](=[OX1])", "SMARTS pattern does not match"

    @staticmethod
    def test_check_thiocarboxylic():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C(=O)S")
        result = PatternRecognition.MolPatterns.check_thiocarboxylic(mol)
        assert result[0] is True, "Thiocarboxylic groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == ("[$([$([CX3](=[SX1])[OX2H1]),$([CX3](=[SX1])[OX1-])]),"
        "$([$([CX3](=[SX1])[SX2H1]),$([CX3](=[SX1])[SX1-])]),"
        "$([$([CX3](=[OX1])[SX2H1]),$([CX3](=[OX1])[SX1-])])]"), "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCS")
        result = PatternRecognition.MolPatterns.check_thiocarboxylic(mol)
        assert result[0] is False, "Thiocarboxylic groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == ("[$([$([CX3](=[SX1])[OX2H1]),$([CX3](=[SX1])[OX1-])]),"
        "$([$([CX3](=[SX1])[SX2H1]),$([CX3](=[SX1])[SX1-])]),"
        "$([$([CX3](=[OX1])[SX2H1]),$([CX3](=[OX1])[SX1-])])]"), "SMARTS pattern does not match"

    @staticmethod
    def test_check_thioester():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CN1CN(C(=S)SC1)C")
        result = PatternRecognition.MolPatterns.check_thioester(mol)
        assert result[0] is True, "Thioester groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$(S([#6])[CX3](=O)),$(O([#6])[CX3](=S)),$([#16]([#6])[CX3](=S))]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCSO")
        result = PatternRecognition.MolPatterns.check_thioester(mol)
        assert result[0] is False, "Thioester groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$(S([#6])[CX3](=O)),$(O([#6])[CX3](=S)),$([#16]([#6])[CX3](=S))]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphide():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCOP(=S)(OCC)SCSC(C)(C)C ")
        result = PatternRecognition.MolPatterns.check_sulphide(mol)
        assert result[0] is True, "Sulphide groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6][#16D2][#6]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("S")
        result = PatternRecognition.MolPatterns.check_sulphide(mol)
        assert result[0] is False, "Sulphide groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#6][#16D2][#6]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_disulphide():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CSSC")
        result = PatternRecognition.MolPatterns.check_disulphide(mol)
        assert result[0] is True, "Disulphide groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#16X2H0][#16X2H0]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCS")
        result = PatternRecognition.MolPatterns.check_disulphide(mol)
        assert result[0] is False, "Disulphide groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#16X2H0][#16X2H0]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_thiocarbamate():
        # Test with a valid input
        mol = Chem.MolFromSmiles(" CCN(CC)C(=O)SCC1=CC=C(C=C1)Cl")
        result = PatternRecognition.MolPatterns.check_thiocarbamate(mol)
        assert result[0] is True, "Thiocarbamate groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([#6][#8D2][CD3](=[S])[#7X3,#7X4+]),$([#6][#16D2][CD3](=[O])[#7X3,#7X4+])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCC")
        result = PatternRecognition.MolPatterns.check_thiocarbamate(mol)
        assert result[0] is False, "Thiocarbamate groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([#6][#8D2][CD3](=[S])[#7X3,#7X4+]),$([#6][#16D2][CD3](=[O])[#7X3,#7X4+])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_thiocyanate():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C(N(C(F)(F)F)OS#N)(F)(F)F")
        result = PatternRecognition.MolPatterns.check_thiocyanate(mol)
        assert result[0] is True, "Thiocyanate groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#16D2]#[#7]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCC")
        result = PatternRecognition.MolPatterns.check_thiocyanate(mol)
        assert result[0] is False, "Thiocyanate groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#16D2]#[#7]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_isothiocyanate():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C=CCN=C=S")
        result = PatternRecognition.MolPatterns.check_isothiocyanate(mol)
        assert result[0] is True, "Isothiocyanate groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#7D2]=[#6]=[#16D1]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCSCN")
        result = PatternRecognition.MolPatterns.check_isothiocyanate(mol)
        assert result[0] is False, "Isothiocyanate groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#7D2]=[#6]=[#16D1]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphinic_acid():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CC1=CC=C(C=C1)S(=O)O")
        result = PatternRecognition.MolPatterns.check_sulphinic_acid(mol)
        assert result[0] is True, "Sulphinic acid groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([#6][#16X3](=[OX1])[OX2H,OX1H0-]),$([#6][#16X3+]([OX1-])[OX2H,OX1H0-])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCSO")
        result = PatternRecognition.MolPatterns.check_sulphinic_acid(mol)
        assert result[0] is False, "Sulphinic acid groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([#6][#16X3](=[OX1])[OX2H,OX1H0-]),$([#6][#16X3+]([OX1-])[OX2H,OX1H0-])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphinic_ester():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCCOS(=O)CCC")
        result = PatternRecognition.MolPatterns.check_sulphinic_ester(mol)
        assert result[0] is True, "Sulphinic ester groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([#6][#16X3](=[OX1])[OX2][#6]),$([#6][#16X3+]([OX1-])[OX2][#6])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCC")
        result = PatternRecognition.MolPatterns.check_sulphinic_ester(mol)
        assert result[0] is False, "Sulphinic ester groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([#6][#16X3](=[OX1])[OX2][#6]),$([#6][#16X3+]([OX1-])[OX2][#6])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphone():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC(=CC=C1N)S(=O)(=O)N")
        result = PatternRecognition.MolPatterns.check_sulphone(mol)
        assert result[0] is True, "Sulphone groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("SSO")
        result = PatternRecognition.MolPatterns.check_sulphone(mol)
        assert result[0] is False, "Sulphone groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_carbosulphone():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC(=CC=C1N)S(=O)(=O)C2=CC=C(C=C2)N")
        result = PatternRecognition.MolPatterns.check_carbosulphone(mol)
        assert result[0] is True, "Carbosulphone groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1=CC(=CC=C1N)S(=O)(=O)N")
        result = PatternRecognition.MolPatterns.check_carbosulphone(mol)
        assert result[0] is False, "Carbosulphone groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphonic_acid():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C(CS(=O)(=O)O)N")
        result = PatternRecognition.MolPatterns.check_sulphonic_acid(mol)
        assert result[0] is True, "Sulphonic acid groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([#16X4](=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])[OX2H,OX1H0-])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("SSSSSO")
        result = PatternRecognition.MolPatterns.check_sulphonic_acid(mol)
        assert result[0] is False, "Sulphonic acid groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([#16X4](=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])[OX2H,OX1H0-])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphonic_ester():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]")
        result = PatternRecognition.MolPatterns.check_sulphonic_ester(mol)
        assert result[0] is True, "Sulphonic ester groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([#16X4](=[OX1])(=[OX1])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])[OX2H0])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1=C(N=C(S1)N=C(N)N)CSCCC(=NS(=O)(=O)N)N ")
        result = PatternRecognition.MolPatterns.check_sulphonic_ester(mol)
        assert result[0] is False, "Sulphonic ester groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([#16X4](=[OX1])(=[OX1])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])[OX2H0])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphonamide():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC(=CC=C1N)S(=O)(=O)N")
        result = PatternRecognition.MolPatterns.check_sulphonamide(mol)
        assert result[0] is True, "Sulphonamide groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCS(=O)(=O)CCN1C(=NC=C1[N+](=O)[O-])C")
        result = PatternRecognition.MolPatterns.check_sulphonamide(mol)
        assert result[0] is False, "Sulphonamide groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphoxide():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC=C(C=C1)C(C2=CC=CC=C2)S(=O)CC(=O)N")
        result = PatternRecognition.MolPatterns.check_sulphoxide(mol)
        assert result[0] is True, "Sulphoxide groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCSO")
        result = PatternRecognition.MolPatterns.check_sulphoxide(mol)
        assert result[0] is False, "Sulphoxide groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_carbosulphoxide():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC=C(C=C1)C(C2=CC=CC=C2)S(=O)CC(=O)N")
        result = PatternRecognition.MolPatterns.check_carbosulphoxide(mol)
        assert result[0] is True, "Carbosulphoxide groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CC(C)(C)C1=CC=C(C=C1)OC2CCCCC2OS(=O)OCC#C")
        result = PatternRecognition.MolPatterns.check_carbosulphoxide(mol)
        assert result[0] is False, "Carbosulphoxide groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphuric_acid():
        # Test with a valid input
        mol = Chem.MolFromSmiles("OS(=O)(=O)O")
        result = PatternRecognition.MolPatterns.check_sulphuric_acid(mol)
        assert result[0] is True, "Sulphuric acid groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([SX4](=[OX1])(=[OX1])([OX2H1,OX1-])[OX2H1,OX1-])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CSO")
        result = PatternRecognition.MolPatterns.check_sulphuric_acid(mol)
        assert result[0] is False, "Sulphuric acid groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([SX4](=[OX1])(=[OX1])([OX2H1,OX1-])[OX2H1,OX1-])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphuric_ester():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C(COS(=O)(=O)O)N")
        result = PatternRecognition.MolPatterns.check_sulphuric_ester(mol)
        assert result[0] is True, "Sulphuric ester groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([SX4](=[OX1])(=[OX1])([OX2H1])[OX2H0][#6]),$([SX4](=[OX1])(=[OX1])([OX2H0][#6])[OX2H0][#6])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCCOS")
        result = PatternRecognition.MolPatterns.check_sulphuric_ester(mol)
        assert result[0] is False, "Sulphuric ester groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([SX4](=[OX1])(=[OX1])([OX2H1])[OX2H0][#6]),$([SX4](=[OX1])(=[OX1])([OX2H0][#6])[OX2H0][#6])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphamic_acid():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1CCC(CC1)NS(=O)(=O)[O-].[Na+]")
        result = PatternRecognition.MolPatterns.check_sulphamic_acid(mol)
        assert result[0] is True, "Sulphamic acid groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([#16X4]([NX3,NX4+])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCCSOCCN")
        result = PatternRecognition.MolPatterns.check_sulphamic_acid(mol)
        assert result[0] is False, "Sulphamic acid groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([#16X4]([NX3,NX4+])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphamic_ester():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC=C2C=C(C=CC2=C1)OS(=O)(=O)N ")
        result = PatternRecognition.MolPatterns.check_sulphamic_ester(mol)
        assert result[0] is True, "Sulphamic ester groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1CCC(CC1)NS(=O)(=O)[O-].[Na+]")
        result = PatternRecognition.MolPatterns.check_sulphamic_ester(mol)
        assert result[0] is False, "Sulphamic ester groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphenic_acid():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C(#N)SO")
        result = PatternRecognition.MolPatterns.check_sulphenic_acid(mol)
        assert result[0] is True, "Sulphenic acid groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#16X2][OX2H,OX1H0-]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1=CC=C2C=C(C=CC2=C1)OS(=O)(=O)N")
        result = PatternRecognition.MolPatterns.check_sulphenic_acid(mol)
        assert result[0] is False, "Sulphenic acid groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#16X2][OX2H,OX1H0-]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_sulphenic_ester():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCOSC1=CC=CC=C1")
        result = PatternRecognition.MolPatterns.check_sulphenic_ester(mol)
        assert result[0] is True, "Sulphenic ester groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#16X2][OX2H0][#6]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C(#N)SO")
        result = PatternRecognition.MolPatterns.check_sulphenic_ester(mol)
        assert result[0] is False, "Sulphenic ester groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#16X2][OX2H0][#6]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_halogen():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CF")
        result = PatternRecognition.MolPatterns.check_halogen(mol)
        assert result[0] is True, "Halogens should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[F,Cl,Br,I]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CN")
        result = PatternRecognition.MolPatterns.check_halogen(mol)
        assert result[0] is False, "Halogens should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[F,Cl,Br,I]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_fluorine():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CF")
        result = PatternRecognition.MolPatterns.check_fluorine(mol)
        assert result[0] is True, "Fluorine atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[F]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCl")
        result = PatternRecognition.MolPatterns.check_fluorine(mol)
        assert result[0] is False, "Fluorine atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[F]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_chlorine():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCl")
        result = PatternRecognition.MolPatterns.check_chlorine(mol)
        assert result[0] is True, "Chlorine atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[Cl]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CF")
        result = PatternRecognition.MolPatterns.check_chlorine(mol)
        assert result[0] is False, "Chlorine atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[Cl]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_bromine():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CBr")
        result = PatternRecognition.MolPatterns.check_bromine(mol)
        assert result[0] is True, "Bromine atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[Br]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCl")
        result = PatternRecognition.MolPatterns.check_bromine(mol)
        assert result[0] is False, "Bromine atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[Br]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_iodine():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CI")
        result = PatternRecognition.MolPatterns.check_iodine(mol)
        assert result[0] is True, "Iodine atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[I]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CBr")
        result = PatternRecognition.MolPatterns.check_iodine(mol)
        assert result[0] is False, "Iodine atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[I]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_halogen_carbon():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCCF")
        result = PatternRecognition.MolPatterns.check_halogen_carbon(mol)
        assert result[0] is True, "Carbons connected to halogens should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6]~[F,Cl,Br,I]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCO")
        result = PatternRecognition.MolPatterns.check_halogen_carbon(mol)
        assert result[0] is False, "Carbons connected to halogens should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#6]~[F,Cl,Br,I]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_halogen_nitrogen():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCNF")
        result = PatternRecognition.MolPatterns.check_halogen_nitrogen(mol)
        assert result[0] is True, "Nitrogens connected to halogens should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#7]~[F,Cl,Br,I]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCF")
        result = PatternRecognition.MolPatterns.check_halogen_nitrogen(mol)
        assert result[0] is False, "Nitrogens connected to halogens should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#7]~[F,Cl,Br,I]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_halogen_oxygen():
        # Test with a valid input
        mol = Chem.MolFromSmiles("OCl(=O)(=O)=O")
        result = PatternRecognition.MolPatterns.check_halogen_oxygen(mol)
        assert result[0] is True, "Oxygens connected to halogens should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#8]~[F,Cl,Br,I]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1CC(=O)N(C1=O)Br")
        result = PatternRecognition.MolPatterns.check_halogen_oxygen(mol)
        assert result[0] is False, "Oxygens connected to halogens should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#8]~[F,Cl,Br,I]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_haloalkane():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC(=CC=C1C(C2=CC=C(C=C2)Cl)C(Cl)Cl)Cl")
        result = PatternRecognition.MolPatterns.check_haloalkane(mol)
        assert result[0] is True, "Haloalkane groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX4]-[F,Cl,Br,I]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCCC")
        result = PatternRecognition.MolPatterns.check_haloalkane(mol)
        assert result[0] is False, "Haloalkane groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX4]-[F,Cl,Br,I]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_haloalkane_primary():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C(CCl)Cl")
        result = PatternRecognition.MolPatterns.check_haloalkane_primary(mol)
        assert result[0] is True, "Primary haloalkane groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX4H3,CX4H2]-[F,Cl,Br,I]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C(C(=O)O)(Cl)Cl")
        result = PatternRecognition.MolPatterns.check_haloalkane_primary(mol)
        assert result[0] is False, "Primary haloalkane groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX4H3,CX4H2]-[F,Cl,Br,I]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_haloalkane_secondary():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C(C(=O)O)(Cl)Cl")
        result = PatternRecognition.MolPatterns.check_haloalkane_secondary(mol)
        assert result[0] is True, "Secondary haloalkane groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX4H1]-[F,Cl,Br,I]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1=CC(=CC=C1C(C2=CC=C(C=C2)Cl)C(Cl)(C)C)Cl")
        result = PatternRecognition.MolPatterns.check_haloalkane_secondary(mol)
        assert result[0] is False, "Secondary haloalkane groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX4H1]-[F,Cl,Br,I]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_haloalkane_tertiary():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC(=CC=C1C(C2=CC=C(C=C2)Cl)C(Cl)(C)C)Cl")
        result = PatternRecognition.MolPatterns.check_haloalkane_tertiary(mol)
        assert result[0] is True, "Tertiary haloalkane groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[CX4H0]-[F,Cl,Br,I]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCCl")
        result = PatternRecognition.MolPatterns.check_haloalkane_tertiary(mol)
        assert result[0] is False, "Tertiary haloalkane groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[CX4H0]-[F,Cl,Br,I]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_haloalkene():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C(=C(Cl)Cl)Cl")
        result = PatternRecognition.MolPatterns.check_haloalkene(mol)
        assert result[0] is True, "Haloalkene groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[C&!c]=[C&!c][F,Cl,Br,I]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCl")
        result = PatternRecognition.MolPatterns.check_haloalkene(mol)
        assert result[0] is False, "Haloalkene groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[C&!c]=[C&!c][F,Cl,Br,I]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_oxohalide():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CC(C(C)(C)C)OP(=O)(C)F")
        result = PatternRecognition.MolPatterns.check_oxohalide(mol)
        assert result[0] is True, "Oxohalide groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#8]=[*H0]~[F,Cl,Br,I]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCC(O)F")
        result = PatternRecognition.MolPatterns.check_oxohalide(mol)
        assert result[0] is False, "Oxohalide groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#8]=[*H0]~[F,Cl,Br,I]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_alkali_metals():
        # Test with a valid input
        mol = Chem.MolFromSmiles("[Li]")
        result = PatternRecognition.MolPatterns.check_alkali_metals(mol)
        assert result[0] is True, "Alkali metals should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[Li,Na,K,Rb,Cs,Fr]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("[Ca]")
        result = PatternRecognition.MolPatterns.check_alkali_metals(mol)
        assert result[0] is False, "Alkali metals should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[Li,Na,K,Rb,Cs,Fr]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_alkaline_earth_metals():
        # Test with a valid input
        mol = Chem.MolFromSmiles("[O-]S(=O)(=O)[O-].[Ba+2]")
        result = PatternRecognition.MolPatterns.check_alkaline_earth_metals(mol)
        assert result[0] is True, "Alkaline earth metals should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[Be,Mg,Ca,Sr,Ba,Ra]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]")
        result = PatternRecognition.MolPatterns.check_alkaline_earth_metals(mol)
        assert result[0] is False, "Alkaline earth metals should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[Be,Mg,Ca,Sr,Ba,Ra]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_transition_metals():
        # Test with a valid input
        mol = Chem.MolFromSmiles("[O-][Cr](=O)(=O)O[Cr](=O)(=O)[O-].[K+].[K+]")
        result = PatternRecognition.MolPatterns.check_transition_metals(mol)
        assert result[0] is True, "Transition metals should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert len(result[2]) == 175, "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("[O-]S(=O)(=O)[O-].[Ba+2]")
        result = PatternRecognition.MolPatterns.check_transition_metals(mol)
        assert result[0] is False, "Transition metals should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert len(result[2]) == 175, "SMARTS pattern does not match"

    @staticmethod
    def test_check_boron_group_elements():
        # Test with a valid input
        mol = Chem.MolFromSmiles("B(O)(O)O")
        result = PatternRecognition.MolPatterns.check_boron_group_elements(mol)
        assert result[0] is True, "Boron group elements should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[B,Al,Ga,In,Ti,Nh]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCC")
        result = PatternRecognition.MolPatterns.check_boron_group_elements(mol)
        assert result[0] is False, "Boron group elements should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[B,Al,Ga,In,Ti,Nh]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_carbon_group_elements():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCCC[Sn](CCCC)(Cl)Cl")
        result = PatternRecognition.MolPatterns.check_carbon_group_elements(mol)
        assert result[0] is True, "Carbon group elements should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[Si,Ge,Sn,Pb,Fl]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCC")
        result = PatternRecognition.MolPatterns.check_carbon_group_elements(mol)
        assert result[0] is False, "Carbon group elements should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[Si,Ge,Sn,Pb,Fl]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_nitrogen_group_elements():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C[As](=O)(C)O")
        result = PatternRecognition.MolPatterns.check_nitrogen_group_elements(mol)
        assert result[0] is True, "Nitrogen group elements should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[As,Sb,Bi]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCC")
        result = PatternRecognition.MolPatterns.check_nitrogen_group_elements(mol)
        assert result[0] is False, "Nitrogen group elements should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[As,Sb,Bi]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_chalcogens():
        # Test with a valid input
        mol = Chem.MolFromSmiles("O[Se](=O)O")
        result = PatternRecognition.MolPatterns.check_chalcogens(mol)
        assert result[0] is True, "Chalcogens should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[Se,Te,Po,Lv]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCC")
        result = PatternRecognition.MolPatterns.check_chalcogens(mol)
        assert result[0] is False, "Chalcogens should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[Se,Te,Po,Lv]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_noble_gases():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C(F)(F)(F)[Xe]C(F)(F)F")
        result = PatternRecognition.MolPatterns.check_noble_gases(mol)
        assert result[0] is True, "Noble gases should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[He,Ne,Ar,Kr,Xe,Rn]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCC")
        result = PatternRecognition.MolPatterns.check_noble_gases(mol)
        assert result[0] is False, "Noble gases should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[He,Ne,Ar,Kr,Xe,Rn]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_pos_charge_1():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC(=CC=C1[N+](=O)[O-])O")
        result = PatternRecognition.MolPatterns.check_pos_charge_1(mol)
        assert result[0] is True, "Positively charged atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[+]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CC(=O)C")
        result = PatternRecognition.MolPatterns.check_pos_charge_1(mol)
        assert result[0] is False, "Positively charged atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[+]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_pos_charge_2():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CC1=CC(=CC(=C1O)[N+](=O)[O-])[N+](=O)[O-]")
        result = PatternRecognition.MolPatterns.check_pos_charge_2(mol)
        assert result[0] is True, "Two positively charged atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[+].[+]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCC")
        result = PatternRecognition.MolPatterns.check_pos_charge_2(mol)
        assert result[0] is False, "Two positively charged atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[+].[+]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_pos_charge_3():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]")
        result = PatternRecognition.MolPatterns.check_pos_charge_3(mol)
        assert result[0] is True, "Three positively charged atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[+].[+].[+]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCC")
        result = PatternRecognition.MolPatterns.check_pos_charge_3(mol)
        assert result[0] is False, "Three positively charged atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[+].[+].[+]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_neg_charge_1():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CC(=O)[O-]")
        result = PatternRecognition.MolPatterns.check_neg_charge_1(mol)
        assert result[0] is True, "Negatively charged atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[-]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CC(=O)O")
        result = PatternRecognition.MolPatterns.check_neg_charge_1(mol)
        assert result[0] is False, "Negatively charged atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[-]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_neg_charge_2():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CC(=O)[O-].CC(=O)[O-].[Zn+2]")
        result = PatternRecognition.MolPatterns.check_neg_charge_2(mol)
        assert result[0] is True, "Two negatively charged atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[-].[-]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C/C=C(/C(=C/C)/C1=CC=C(C=C1)O)\C2=CC=C(C=C2)O")
        result = PatternRecognition.MolPatterns.check_neg_charge_2(mol)
        assert result[0] is False, "Two negatively charged atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[-].[-]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_neg_charge_3():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C(C(=O)[O-])C(CC(=O)[O-])(C(=O)[O-])O")
        result = PatternRecognition.MolPatterns.check_neg_charge_3(mol)
        assert result[0] is True, "Three negatively charged atoms should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[-].[-].[-]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C(C(=O)O)C(CC(=O)O)(C(=O)O)O")
        result = PatternRecognition.MolPatterns.check_neg_charge_3(mol)
        assert result[0] is False, "Three negatively charged atoms should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[-].[-].[-]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_zwitterion():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CCC(CC)NC1=C(C=C(C(=C1[N+](=O)[O-])C)C)[N+](=O)[O-]")
        result = PatternRecognition.MolPatterns.check_zwitterion(mol)
        assert result[0] is True, "Zwitterions should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert len(result[2]) == 442, "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCC")
        result = PatternRecognition.MolPatterns.check_zwitterion(mol)
        assert result[0] is False, "Zwitterions should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert len(result[2]) == 442, "SMARTS pattern does not match"


    @staticmethod
    def test_check_hbond_acceptors():
        # Test with a valid input
        mol = Chem.MolFromSmiles("CC(=CCC/C(=C/CO)/C)C")
        result = PatternRecognition.MolPatterns.check_hbond_acceptors(mol)
        assert result[0] is True, "Hydrogen bond acceptors should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCC")
        result = PatternRecognition.MolPatterns.check_hbond_acceptors(mol)
        assert result[0] is False, "Hydrogen bond acceptors should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_hbond_acceptors_higher_than():
        # Test with a valid input and a threshold
        mol = Chem.MolFromSmiles("CC(=C)[C@H]1CC2=C(O1)C=CC3=C2O[C@@H]4COC5=CC(=C(C=C5[C@@H]4C3=O)OC)OC")
        n = 5  # Example threshold
        result, atoms, pattern = PatternRecognition.MolPatterns.check_hbond_acceptors_higher_than(mol, n)
        assert result is True, f"Number of hydrogen bond acceptors should be higher than {n}"
        assert len(atoms) > n, f"Expected more than {n} hydrogen bond acceptors but got {len(atoms)}"
        assert len(pattern) == 63, "SMARTS pattern does not match"

        # Test with an invalid input and a threshold
        mol = Chem.MolFromSmiles("CCCO")
        result, atoms, pattern = PatternRecognition.MolPatterns.check_hbond_acceptors_higher_than(mol, n)
        assert result is False, f"Number of hydrogen bond acceptors should not be higher than {n}"
        assert len(atoms) <= n, f"Expected less than or equal to {n} hydrogen bond acceptors but got {len(atoms)}"
        assert len(pattern) == 63, "SMARTS pattern does not match"

    @staticmethod
    def test_check_hbond_acceptors_lower_than():
        # Test with a valid input and a threshold
        mol = Chem.MolFromSmiles("CCCO")
        n = 3  # Example threshold
        result, atoms, pattern = PatternRecognition.MolPatterns.check_hbond_acceptors_lower_than(mol, n)
        assert result is True, f"Number of hydrogen bond acceptors should be lower than {n}"
        assert len(atoms) < n, f"Expected less than {n} hydrogen bond acceptors but got {len(atoms)}"
        assert len(pattern) == 63, "SMARTS pattern does not match"

        # Test with an invalid input and a threshold
        mol = Chem.MolFromSmiles("CC(=C)[C@H]1CC2=C(O1)C=CC3=C2O[C@@H]4COC5=CC(=C(C=C5[C@@H]4C3=O)OC)OC")
        result, atoms, pattern = PatternRecognition.MolPatterns.check_hbond_acceptors_lower_than(mol, n)
        assert result is False, f"Number of hydrogen bond acceptors should not be lower than {n}"
        assert len(atoms) >= n, f"Expected more than or equal to {n} hydrogen bond acceptors but got {len(atoms)}"
        assert len(pattern) == 63, "SMARTS pattern does not match"

    @staticmethod
    def test_check_hbond_donors():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC=C2C(=C1)NC(=N2)C3=CSC=N3")
        result = PatternRecognition.MolPatterns.check_hbond_donors(mol)
        assert result[0] is True, "Hydrogen bond donors should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[!$([#6,H0,-,-2,-3])]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCC")
        result = PatternRecognition.MolPatterns.check_hbond_donors(mol)
        assert result[0] is False, "Hydrogen bond donors should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[!$([#6,H0,-,-2,-3])]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_hbond_donors_higher_than():
        # Test with a valid input and a threshold
        mol = Chem.MolFromSmiles("C([C@@H]([C@@H]1C(=C(C(=O)O1)O)O)O)O")
        n = 3  # Example threshold
        result, atoms, pattern = PatternRecognition.MolPatterns.check_hbond_donors_higher_than(mol, n)
        assert result is True, f"Number of hydrogen bond donors should be higher than {n}"
        assert len(atoms) > n, f"Expected more than {n} hydrogen bond donors but got {len(atoms)}"
        assert len(pattern) == 21, "SMARTS pattern does not match"

        # Test with an invalid input and a threshold
        mol = Chem.MolFromSmiles("CC(=O)O")
        result, atoms, pattern = PatternRecognition.MolPatterns.check_hbond_donors_higher_than(mol, n)
        assert result is False, f"Number of hydrogen bond donors should not be higher than {n}"
        assert len(atoms) <= n, f"Expected less than or equal to {n} hydrogen bond donors but got {len(atoms)}"
        assert len(pattern) == 21, "SMARTS pattern does not match"

    @staticmethod
    def test_check_hbond_donors_lower_than():
        # Test with a valid input and a threshold
        mol = Chem.MolFromSmiles("CC(=O)O")
        n = 3  # Example threshold
        result, atoms, pattern = PatternRecognition.MolPatterns.check_hbond_donors_lower_than(mol, n)
        assert result is True, f"Number of hydrogen bond donors should be lower than {n}"
        assert len(atoms) < n, f"Expected less than {n} hydrogen bond donors but got {len(atoms)}"
        assert len(pattern) == 21, "SMARTS pattern does not match"

        # Test with an invalid input and a threshold
        mol = Chem.MolFromSmiles("C([C@@H]([C@@H]1C(=C(C(=O)O1)O)O)O)O")
        result, atoms, pattern = PatternRecognition.MolPatterns.check_hbond_donors_lower_than(mol, n)
        assert result is False, f"Number of hydrogen bond donors should not be lower than {n}"
        assert len(atoms) >= n, f"Expected more than or equal to {n} hydrogen bond donors but got {len(atoms)}"
        assert len(pattern) == 21, "SMARTS pattern does not match"

    @staticmethod
    def test_check_unbranched_rotatable_chain():
        # Test with a valid input and a number of units
        mol = Chem.MolFromSmiles("CNONONO")
        n_units = 3  # Example number of units
        result = PatternRecognition.MolPatterns.check_unbranched_rotatable_chain(mol, n_units)
        assert result[0] is True, f"Unbranched rotatable chains with {n_units} units should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == ["[R0;D2]" *n_units][0], "SMARTS pattern does not match"

        # Test with an invalid input and a number of units
        mol = Chem.MolFromSmiles("CC")
        result = PatternRecognition.MolPatterns.check_unbranched_rotatable_chain(mol, n_units)
        assert result[0] is False, f"Unbranched rotatable chains with {n_units} units should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == ["[R0;D2]" * n_units][0], "SMARTS pattern does not match"

    @staticmethod
    def test_check_unbranched_rotatable_carbons():
        # Test with a valid input and a number of units
        mol = Chem.MolFromSmiles("CCCCCC")
        n_units = 3  # Example number of units
        result = PatternRecognition.MolPatterns.check_unbranched_rotatable_carbons(mol, n_units)
        assert result[0] is True, f"Unbranched rotatable carbons with {n_units} units should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == ["[R0;CD2]" * n_units][0], "SMARTS pattern does not match"

        # Test with an invalid input and a number of units
        mol = Chem.MolFromSmiles("CC")
        result = PatternRecognition.MolPatterns.check_unbranched_rotatable_carbons(mol, n_units)
        assert result[0] is False, f"Unbranched rotatable carbons with {n_units} units should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == ["[R0;CD2]" * n_units][0], "SMARTS pattern does not match"

    @staticmethod
    def test_check_unbranched_structure():
        # Test with a valid input and a number of units
        mol = Chem.MolFromSmiles("CC(=CCC/C(=C/CO)/C)C")
        n_units = 3  # Example number of units
        result = PatternRecognition.MolPatterns.check_unbranched_structure(mol, n_units)
        assert result[0] is True, f"Unbranched structures with {n_units} units should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == ["[R0;D2]" * n_units][0], "SMARTS pattern does not match"

        # Test with an invalid input and a number of units
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = PatternRecognition.MolPatterns.check_unbranched_structure(mol, n_units)
        assert result[0] is False, f"Unbranched structures with {n_units} units should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == ["[R0;D2]" * n_units][0], "SMARTS pattern does not match"

    @staticmethod
    def test_check_ring():
        # Test with a valid input
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = PatternRecognition.Rings.check_ring(mol)
        assert result[0] is True, "Rings should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[R]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CCCCCC")
        result = PatternRecognition.Rings.check_ring(mol)
        assert result[0] is False, "Rings should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[R]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_pattern_cyclic():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1CCCCN1")
        pattern_function = PatternRecognition.MolPatterns.check_nitrogen  # Example pattern function
        result, atoms, patterns = PatternRecognition.Rings.check_pattern_cyclic(mol, pattern_function)
        assert result is True, "Pattern should share atoms with ring structures in the molecule"
        assert len(atoms) > 0, "No atom indices returned"
        assert patterns == ("[#7]", "[R]"), "SMARTS patterns do not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1CCCCO1")
        result, atoms, patterns = PatternRecognition.Rings.check_pattern_cyclic(mol, pattern_function)
        assert result is False, "Pattern should not share atoms with ring structures in the molecule"
        assert len(atoms) == 0, "Atom indices should be empty"
        assert patterns == ("[#7]", "[R]"), "SMARTS patterns do not match"

    @staticmethod
    def test_check_pattern_cyclic_substituent():
        # Test with a valid input
        mol = Chem.MolFromSmiles("OC1CCCCC1")
        pattern_function = PatternRecognition.MolPatterns.check_oxygen  # Example pattern function
        result, atoms, patterns = PatternRecognition.Rings.check_pattern_cyclic_substituent(mol, pattern_function)
        assert result is True, "Pattern should share atoms with ring structures connected to something not via a ring bond"
        assert len(atoms) > 0, "No atom indices returned"
        assert patterns == ('[#8]', '[R]!@[*]'), "SMARTS patterns do not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1CCCCC1CC")
        result, atoms, patterns = PatternRecognition.Rings.check_pattern_cyclic_substituent(mol, pattern_function)
        assert result is False, "Pattern should not share atoms with ring structures connected to something not via a ring bond"
        assert len(atoms) == 0, "Atom indices should be empty"
        assert patterns == ('[#8]', '[R]!@[*]'), "SMARTS patterns do not match"

    @staticmethod
    def test_check_ortho_substituted_aromatic_r6():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC(=C(C=C1Cl)Cl)OCC(=O)O")
        result = PatternRecognition.Rings.check_ortho_substituted_aromatic_r6(mol)
        assert result[0] is True, "Ortho-substituted benzene groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "a1(-[*&!#1&!a&!R])a(-[*&!#1&!a&!R])aaaa1", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("C1=CC(=CC=C1C2=COC3=CC(=CC(=C3C2=O)O)O)O")
        result = PatternRecognition.Rings.check_ortho_substituted_aromatic_r6(mol)
        assert result[0] is False, "Ortho-substituted benzene groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "a1(-[*&!#1&!a&!R])a(-[*&!#1&!a&!R])aaaa1", "SMARTS pattern does not match"

    @staticmethod
    def test_check_meta_substituted_aromatic_r6():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC(=C(C=C1Cl)Cl)OCC(=O)O")
        result = PatternRecognition.Rings.check_meta_substituted_aromatic_r6(mol)
        assert result[0] is True, "Meta-substituted benzene groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "a1(-[*&!#1&!a&!R])aa(-[*&!#1&!a&!R])aaa1", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CC1=CC=CC=C1C")
        result = PatternRecognition.Rings.check_meta_substituted_aromatic_r6(mol)
        assert result[0] is False, "Meta-substituted benzene groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "a1(-[*&!#1&!a&!R])aa(-[*&!#1&!a&!R])aaa1", "SMARTS pattern does not match"

    @staticmethod
    def test_check_para_substituted_aromatic_r6():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC(=C(C=C1Cl)Cl)OCC(=O)O")
        result = PatternRecognition.Rings.check_para_substituted_aromatic_r6(mol)
        assert result[0] is True, "Para-substituted benzene groups should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "a1(-[*&!#1&!a&!R])aaa(-[*&!#1&!a&!R])aa1", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("CC1=CC=CC=C1C")
        result = PatternRecognition.Rings.check_para_substituted_aromatic_r6(mol)
        assert result[0] is False, "Para-substituted benzene groups should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "a1(-[*&!#1&!a&!R])aaa(-[*&!#1&!a&!R])aa1", "SMARTS pattern does not match"

    @staticmethod
    def test_check_macrocycle():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C[C@@]12CC[C@@H]3[C@H](CC3(C)C)C(=C)CC[C@H]1O2")
        result = PatternRecognition.Rings.check_macrocycle(mol)
        assert result[0] is True, "Macrocycles should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[r;!r3;!r4;!r5;!r6;!r7]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = PatternRecognition.Rings.check_macrocycle(mol)
        assert result[0] is False, "Macrocycles should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[r;!r3;!r4;!r5;!r6;!r7]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_ring_size():
        # Test with a valid input and a ring size
        mol = Chem.MolFromSmiles("c1ccccc1")
        size = 6  # Example ring size
        result = PatternRecognition.Rings.check_ring_size(mol, size)
        assert result[0] is True, f"Rings of size {size} should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[r6]", "SMARTS pattern does not match"

        # Test with an invalid input and a ring size
        mol = Chem.MolFromSmiles("C1CCCC1")
        result = PatternRecognition.Rings.check_ring_size(mol, size)
        assert result[0] is False, f"Rings of size {size} should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[r6]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_ring_fusion():
        # Test with a valid input
        mol = Chem.MolFromSmiles("C1=CC=C2C=CC=CC2=C1")
        result = PatternRecognition.Rings.check_ring_fusion(mol)
        assert result[0] is True, "Fused ring systems should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#6R2,#6R3,#6R4]", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("c1ccccc1c2ccccc2")
        result = PatternRecognition.Rings.check_ring_fusion(mol)
        assert result[0] is False, "Fused ring systems should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#6R2,#6R3,#6R4]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_heterocycle():
        # Test with a valid input
        mol = Chem.MolFromSmiles("c1ccncc1")
        result = PatternRecognition.Rings.check_heterocycle(mol)
        assert result[0] is True, "Heterocycles should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "* |$CHC$|", "SMARTS pattern does not match"

        # Test with an invalid input
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = PatternRecognition.Rings.check_heterocycle(mol)
        assert result[0] is False, "Heterocycles should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "* |$CHC$|", "SMARTS pattern does not match"

    @staticmethod
    def test_check_heterocycle_N():
        # Test with a valid input
        smi = "c1ccncc1"
        result = PatternRecognition.Rings.check_heterocycle_N(smi)
        assert result[0] is True, "Nitrogen-containing heterocycles should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#7R]", "SMARTS pattern does not match"

        # Test with an invalid input
        smi = "c1ccccc1"
        result = PatternRecognition.Rings.check_heterocycle_N(smi)
        assert result[0] is False, "Nitrogen-containing heterocycles should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#7R]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_heterocycle_O():
        # Test with a valid input
        smi = "C1CCOCC1"
        result = PatternRecognition.Rings.check_heterocycle_O(smi)
        assert result[0] is True, "Oxygen-containing heterocycles should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#8R]", "SMARTS pattern does not match"

        # Test with an invalid input
        smi = "c1ccccc1"
        result = PatternRecognition.Rings.check_heterocycle_O(smi)
        assert result[0] is False, "Oxygen-containing heterocycles should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#8R]", "SMARTS pattern does not match"

    @staticmethod
    def test_check_heterocycle_S():
        # Test with a valid input
        smi = "C1CCCSC1"
        result = PatternRecognition.Rings.check_heterocycle_S(smi)
        assert result[0] is True, "Sulphur-containing heterocycles should be found in the molecule"
        assert len(result[1]) > 0, "No atom indices returned"
        assert result[2] == "[#16R]", "SMARTS pattern does not match"

        # Test with an invalid input
        smi = "c1ccccc1"
        result = PatternRecognition.Rings.check_heterocycle_S(smi)
        assert result[0] is False, "Sulphur-containing heterocycles should not be found in the molecule"
        assert len(result[1]) == 0, "Atom indices should be empty"
        assert result[2] == "[#16R]", "SMARTS pattern does not match"

    if __name__ == "__main__":
        pytest.main()