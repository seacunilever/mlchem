import numpy as np
import pandas as pd
from rdkit import Chem
from typing import Literal, Iterable


def get_rdkitDesc(mol_input_list: Iterable[str | Chem.rdchem.Mol],
                  include_3D: bool = False) -> pd.DataFrame:
    """
Calculate RDKit descriptors for a list of molecules.

This function computes 2D descriptors for each molecule in the input list.
If `include_3D` is True, it also calculates 3D descriptors and merges them
with the 2D descriptors.

Parameters
----------
mol_input_list : Iterable[str or rdkit.Chem.rdchem.Mol]
    List of molecules in SMILES format or as RDKit Mol objects.
include_3D : bool, optional
    Whether to include 3D descriptors. Default is False.

Returns
-------
pd.DataFrame
    DataFrame containing the descriptors for each molecule.

Examples
--------
>>> get_rdkitDesc(["CCO", "c1ccccc1"], include_3D=False)
"""


    from rdkit.Chem import Descriptors
    from mlchem.chem.manipulation import create_molecule
    from mlchem.helper import merge_dicts_with_duplicates

    # Define two inner helper functions

    def get_desc_2d(mol_input: str | Chem.rdchem.Mol) -> dict:
        """Calculate 2D descriptors for a single molecule."""
        return Descriptors.CalcMolDescriptors(create_molecule(mol_input))

    def get_desc_3d(mol_input: str | Chem.rdchem.Mol) -> dict:
        """Calculate 3D descriptors for a single molecule."""
        from rdkit.Chem.Descriptors3D import CalcMolDescriptors3D

        try:
            mol_input_h = create_molecule(mol_input=mol_input,
                                          add_hydrogens=True,
                                          show=False,
                                          solid_sticks=True,
                                          is_3d=True,
                                          optimise=True)
        except Exception as e:
            print(f"Problem encountered with: {mol_input}."
                               f"Error: {e}")
            pass
        return CalcMolDescriptors3D(mol_input_h)

    # Calculate descriptors for each molecule in a dictionary

    dict_desc = {
        (m if isinstance(m, str) else Chem.MolToSmiles(m)):
        (merge_dicts_with_duplicates(get_desc_2d(m),
                                     get_desc_3d(m)) if include_3D
            else get_desc_2d(m)) for m in mol_input_list
    }

    # Dataframes from dictionary need to be transposed

    df = pd.DataFrame(dict_desc).T

    # Remove Ipc descriptor as it returns innatural values
    df = df[[c for c in df.columns if c != 'Ipc']]
    return df


def get_mordredDesc(mol_input_list: list | np.ndarray[str | Chem.rdchem.Mol],
                    include_3D: bool = False) -> pd.DataFrame:
    """
Calculate Mordred descriptors for a list of molecules.

This function computes Mordred descriptors for each molecule in the input list.
If `include_3D` is True, 3D descriptors are included.

Parameters
----------
mol_input_list : list or np.ndarray of str or rdkit.Chem.rdchem.Mol
    List or array of molecules in SMILES format or as RDKit Mol objects.
include_3D : bool, optional
    Whether to include 3D descriptors. Default is False.

Returns
-------
pd.DataFrame
    DataFrame containing the descriptors for each molecule.

Examples
--------
>>> get_mordredDesc(["CCO", "c1ccccc1"], include_3D=True)
"""

    from mlchem.chem.manipulation import create_molecule
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=1-include_3D)

    # Define inner helper function

    def get_desc(mol_input: str | Chem.rdchem.Mol, calculator=calc) -> list:
        """Calculate descriptors for a single molecule."""
        try:
            mol = create_molecule(mol_input)
            return calculator(mol)
        except Exception as e:
            print(f"Problem encountered with: {mol_input}."
                  f"Error: {e}")
            return [None] * len(calculator.descriptors)
        

    # Calculate descriptors for each molecule in a dictionary

    dict_desc = {
        (m if isinstance(m, str) else Chem.MolToSmiles(m)):
        get_desc(m, calculator=calc) for m in mol_input_list
    }

    df_desc = pd.DataFrame(dict_desc).T
    df_desc.columns = [str(d) for d in calc.descriptors]
    return df_desc


def get_allDesc(mol_input_list: list[str | Chem.rdchem.Mol] |
                np.ndarray[str | Chem.rdchem.Mol],
                include_3D: bool = False) -> pd.DataFrame:
    """
Calculate both Mordred and RDKit descriptors for a list of molecules.

This function computes both Mordred and RDKit descriptors for each molecule
in the input list. If `include_3D` is True, 3D descriptors are included
in both sets.

Parameters
----------
mol_input_list : list or np.ndarray of str or rdkit.Chem.rdchem.Mol
    List or array of molecules in SMILES format or as RDKit Mol objects.
include_3D : bool, optional
    Whether to include 3D descriptors. Default is False.

Returns
-------
pd.DataFrame
    DataFrame containing the combined descriptors for each molecule.

Examples
--------
>>> get_allDesc(["CCO", "c1ccccc1"], include_3D=True)
"""


    import pandas as pd

    desc_rdkit = get_rdkitDesc(mol_input_list, include_3D=include_3D)
    desc_mordred = get_mordredDesc(mol_input_list, include_3D=include_3D)
    desc_both = pd.concat([desc_rdkit, desc_mordred], axis=1)

    # Remove duplicates when they come from mordred
    desc_both = desc_both.loc[:, ~desc_both.columns.duplicated(keep='first')]
    return desc_both


def get_atomicDesc(mol_input: str | Chem.rdchem.Mol,
                   atom_index: int) -> pd.DataFrame:
    """
Calculate atomic descriptors for a specific atom in a molecule.

This function computes a comprehensive set of atomic-level descriptors
for a given atom in a molecule. These include properties related to
bond types, hybridisation, charges, ring membership, and statistics
on neighbouring atoms up to the third order.

Parameters
----------
mol_input : str or rdkit.Chem.rdchem.Mol
    Molecule in SMILES format or as an RDKit Mol object.
atom_index : int
    Index of the atom for which descriptors are calculated.

Returns
-------
pd.DataFrame
    A DataFrame containing the descriptors for the specified atom.

Raises
------
RuntimeError
    If the molecule cannot be created from the input.
IndexError
    If the atom index is out of bounds.

Examples
--------
>>> get_atomicDesc("CC(=O)O", atom_index=1)
"""

    from mlchem.chem.manipulation import create_molecule
    from mlchem.chem.manipulation import PatternRecognition as pr
    from mlchem.chem.manipulation import PropManager as pm

    prA = pr.Atoms
    prBn = pr.Bonds

    if isinstance(mol_input, str):
        try:
            mol = create_molecule(mol_input)
            smiles = mol_input
        except Exception as e:
            raise RuntimeError(
                f"Error creating molecule from input: {mol_input}. Error: {e}"
                )
    else:
        mol = mol_input
        smiles = Chem.MolToSmiles(mol)

    mol_h = create_molecule(mol_input, is_3d=True, add_hydrogens=True)
    smiles_h = Chem.MolToSmiles(mol_h)

    mol.ComputeGasteigerCharges()

    distmat = pm.Mol.get_distance_matrix(mol_h)

    tot_atoms = mol.GetNumAtoms()
    if atom_index >= tot_atoms:
        raise IndexError(
        f"Atom index ({atom_index}) larger than total atoms ({tot_atoms})"
        )

    a = mol.GetAtomWithIdx(atom_index)

    bonds = list(a.GetBonds())
    symbol = a.GetSymbol()
    neighbours = pm.Atom.get_neighbours(a, 1)
    neighbours_2nd_order = pm.Atom.get_neighbours(a, 2)
    neighbours_3rd_order = pm.Atom.get_neighbours(a, 3)

    dict_properties = {'SMILES': smiles,
                       'SMILES_H': smiles_h,
                       'SYMBOL': symbol,
                       'total_degree': a.GetTotalDegree(),
                       'total_valence': a.GetTotalValence(),
                       'formal_charge': a.GetFormalCharge(),
                       'is_SP': prA.is_SP(a),
                       'is_SP2': prA.is_SP2(a),
                       'is_SP3': prA.is_SP3(a),
                       'tot_single_b': np.sum(
                           [prBn.is_single_bond(b) for b in bonds]),
                       'avg_single_b': np.mean(
                           [prBn.is_single_bond(b) for b in bonds]),
                       'tot_double_b': np.sum(
                           [prBn.is_double_bond(b) for b in bonds]),
                       'avg_double_b': np.mean(
                           [prBn.is_double_bond(b) for b in bonds]),
                       'tot_triple_b': np.sum(
                           [prBn.is_triple_bond(b) for b in bonds]),
                       'avg_triple_b': np.mean(
                           [prBn.is_triple_bond(b) for b in bonds]),
                       'tot_dative_b': np.sum(
                           [prBn.is_dative_bond(b) for b in bonds]),
                       'avg_dative_b': np.mean(
                           [prBn.is_dative_bond(b) for b in bonds]),
                       'is_aromatic': int(a.GetIsAromatic()),
                       'H_bonded': a.GetTotalNumHs(includeNeighbors=True),
                       'is_in_ring': int(a.IsInRing()),
                       'ring_size': prA.get_ring_size(a),
                       'gasteiger_charge': a.GetDoubleProp("_GasteigerCharge"),
                       'avg_deg_neighbours': np.mean(
                           [atom.GetTotalDegree() for atom in neighbours]),
                       'tot_deg_neighbours': np.sum(
                           [atom.GetTotalDegree() for atom in neighbours]),
                       'avg_deg_neighbours2': np.mean(
                           [atom.GetTotalDegree() for atom in
                            neighbours_2nd_order]),
                       'tot_deg_neighbours2': np.sum(
                           [atom.GetTotalDegree() for atom in
                            neighbours_2nd_order]),
                       'avg_degree_neighbours3': np.mean(
                           [atom.GetTotalDegree() for atom in
                            neighbours_3rd_order]),
                       'tot_deg_neighbours3': np.sum(
                           [atom.GetTotalDegree() for atom in
                            neighbours_3rd_order]),
                       'avg_val_neighbours': np.mean(
                           [atom.GetTotalValence() for atom in neighbours]),
                       'tot_val_neighbours': np.sum(
                           [atom.GetTotalValence() for atom in neighbours]),
                       'avg_val_neighbours2': np.mean(
                           [atom.GetTotalValence() for atom in
                            neighbours_2nd_order]),
                       'tot_val_neighbours2': np.sum(
                           [atom.GetTotalValence() for atom in
                            neighbours_2nd_order]),
                       'avg_val_neighbours3': np.mean(
                           [atom.GetTotalValence() for atom in
                            neighbours_3rd_order]),
                       'tot_val_neighbours3': np.sum(
                           [atom.GetTotalValence() for atom in
                            neighbours_3rd_order]),
                       'avg_formal_charge_neighbours': np.mean(
                           [atom.GetFormalCharge() for atom in neighbours]),
                       'tot_formal_charge_neighbours': np.sum(
                           [atom.GetFormalCharge() for atom in neighbours]),
                       'avg formal_charge_neighbours2': np.mean(
                           [atom.GetFormalCharge() for atom in
                            neighbours_2nd_order]),
                       'tot_formal_charge_neighbours2': np.sum(
                           [atom.GetFormalCharge() for atom in
                            neighbours_2nd_order]),
                       'avg_formal_charge_neighbours3': np.mean(
                           [atom.GetFormalCharge() for atom in
                            neighbours_3rd_order]),
                       'tot_formal_charge_neighbours3': np.sum(
                           [atom.GetFormalCharge() for atom in
                            neighbours_3rd_order]),
                       'avg SP1 degree of neighbours': np.mean(
                           [prA.is_SP(atom) for atom in neighbours]),
                       'tot_SP1_deg_neighbours': np.sum(
                           [prA.is_SP(atom) for atom in neighbours]),
                       'avg_SP1_deg_neighbours2': np.mean(
                           [prA.is_SP(atom) for atom in neighbours_2nd_order]),
                       'tot_SP1_deg_neighbours2': np.sum(
                           [prA.is_SP(atom) for atom in neighbours_2nd_order]),
                       'avg_SP1_deg_neighbours3': np.mean(
                           [prA.is_SP(atom) for atom in neighbours_3rd_order]),
                       'tot_SP1_deg_neighbours3': np.sum(
                           [prA.is_SP(atom) for atom in neighbours_3rd_order]),
                       'avg_SP2_deg_neighbours': np.mean(
                           [prA.is_SP2(atom) for atom in neighbours]),
                       'tot_SP2_deg_neighbours': np.sum(
                           [prA.is_SP2(atom) for atom in neighbours]),
                       'avg_SP2_deg_neighbours2': np.mean(
                           [prA.is_SP2(atom) for atom in
                            neighbours_2nd_order]),
                       'tot_SP2_deg_neighbours2': np.sum(
                           [prA.is_SP2(atom) for atom in
                            neighbours_2nd_order]),
                       'avg_SP2_deg_neighbours3': np.mean(
                           [prA.is_SP2(atom) for atom in
                            neighbours_3rd_order]),
                       'tot_SP2_deg_neighbours3': np.sum(
                           [prA.is_SP2(atom) for atom in
                            neighbours_3rd_order]),
                       'avg_SP3_deg_neighbours': np.mean(
                           [prA.is_SP3(atom) for atom in
                            neighbours]),
                       'tot_SP3_deg_neighbours': np.sum(
                           [prA.is_SP3(atom) for atom in
                            neighbours]),
                       'avg_SP3_deg_neighbours2': np.mean(
                           [prA.is_SP3(atom) for atom in
                            neighbours_2nd_order]),
                       'tot_SP3_deg_neighbours2': np.sum(
                           [prA.is_SP3(atom) for atom in
                            neighbours_2nd_order]),
                       'avg_SP3_deg_neighbours3': np.mean(
                           [prA.is_SP3(atom) for atom in
                            neighbours_3rd_order]),
                       'tot_SP3_deg_neighbours3': np.sum(
                           [prA.is_SP3(atom) for atom in
                            neighbours_3rd_order]),
                       'avg_arom_neighbours': np.mean(
                           [atom.GetIsAromatic() for atom in neighbours]),
                       'tot_arom_neighbours': np.sum(
                           [atom.GetIsAromatic() for atom in neighbours]),
                       'avg_arom_neighbours2': np.mean(
                           [atom.GetIsAromatic() for atom in
                            neighbours_2nd_order]),
                       'tot_arom_neighbours2': np.sum(
                           [atom.GetIsAromatic() for atom in
                            neighbours_2nd_order]),
                       'avg_arom_neighbours3': np.mean(
                           [atom.GetIsAromatic() for atom in
                            neighbours_3rd_order]),
                       'tot_arom_neighbours3': np.sum(
                           [atom.GetIsAromatic() for atom in
                            neighbours_3rd_order]),
                       'avgmass_neighbours': np.mean(
                           [atom.GetMass() for atom in neighbours]),
                       'tot_mass_neighbours': np.sum(
                           [atom.GetMass() for atom in neighbours]),
                       'avg_mass_neighbours2': np.mean(
                           [atom.GetMass() for atom in neighbours_2nd_order]),
                       'tot_mass_neighbours2': np.sum(
                           [atom.GetMass() for atom in neighbours_2nd_order]),
                       'avg_mass_neighbours3': np.mean(
                           [atom.GetMass() for atom in neighbours_3rd_order]),
                       'tot_mass_neighbours3': np.sum(
                           [atom.GetMass() for atom in neighbours_3rd_order]),
                       'avg_H_bonded_neighbours': np.mean(
                           [atom.GetTotalNumHs(includeNeighbors=True) for
                            atom in neighbours]),
                       'tot_H_bonded_neighbours': np.sum(
                           [atom.GetTotalNumHs(includeNeighbors=True) for
                            atom in neighbours]),
                       'avg_H_bonded_neighbours2': np.mean(
                           [atom.GetTotalNumHs(includeNeighbors=True) for
                            atom in neighbours_2nd_order]),
                       'total_H_bonded_neighbours2': np.sum(
                           [atom.GetTotalNumHs(includeNeighbors=True) for
                            atom in neighbours_2nd_order]),
                       'avg_H_bonded_neighbours3': np.mean(
                           [atom.GetTotalNumHs(includeNeighbors=True) for
                            atom in neighbours_3rd_order]),
                       'total_H_bonded_neighbours3': np.sum(
                           [atom.GetTotalNumHs(includeNeighbors=True) for
                            atom in neighbours_3rd_order]),
                       'avg_ring_size_neighbours': np.mean(
                        [prA.get_ring_size(atom) for atom in neighbours]),
                       'tot_ring_size_neighbours': np.sum(
                        [prA.get_ring_size(atom) for atom in neighbours]),
                       'avg_ring_size_neighbours2': np.mean(
                        [prA.get_ring_size(atom) for atom in
                         neighbours_2nd_order]),
                       'tot_ring_size_neighbours2': np.sum(
                        [prA.get_ring_size(atom) for atom in
                         neighbours_2nd_order]),
                       'avg_ring_size_neighbours3': np.mean(
                        [prA.get_ring_size(atom) for atom in
                         neighbours_3rd_order]),
                       'tot_ring_size_neighbours3': np.sum(
                        [prA.get_ring_size(atom) for atom in
                         neighbours_3rd_order]),
                       'avg_gasteiger_charge_neighbours': np.mean(
                        [atom.GetDoubleProp('_GasteigerCharge') for
                         atom in neighbours]),
                       'tot_gasteiger_charge_neighbours': np.sum(
                        [atom.GetDoubleProp('_GasteigerCharge') for
                         atom in neighbours]),
                       'max_gasteiger_charge_neighbours': np.max(
                        [atom.GetDoubleProp('_GasteigerCharge') for
                         atom in neighbours]),
                       'min_gasteiger_charge_neighbours': np.min(
                        [atom.GetDoubleProp('_GasteigerCharge') for
                         atom in neighbours]),
                       'avg_gasteiger_charge_neighbours2': np.mean(
                        [atom.GetDoubleProp('_GasteigerCharge') for
                         atom in neighbours_2nd_order]),
                       'tot_gasteiger_charge_neighbours2': np.sum(
                        [atom.GetDoubleProp('_GasteigerCharge') for
                         atom in neighbours_2nd_order]),
                       'max_gasteiger_charge_neighbours2': np.max(
                        [atom.GetDoubleProp('_GasteigerCharge') for
                         atom in neighbours_2nd_order]),
                       'min_gasteiger_charge_neighbours2': np.min(
                        [atom.GetDoubleProp('_GasteigerCharge') for
                         atom in neighbours_2nd_order]),
                       'avg_gasteiger_charge_neighbours3': np.mean(
                        [atom.GetDoubleProp('_GasteigerCharge') for
                         atom in neighbours_3rd_order]),
                       'total_gasteiger_charge_neighbours3': np.sum(
                        [atom.GetDoubleProp('_GasteigerCharge') for
                         atom in neighbours_3rd_order]),
                       'max_gasteiger_charge_neighbours3': np.max(
                        [atom.GetDoubleProp('_GasteigerCharge') for
                         atom in neighbours_3rd_order]),
                       'min_gasteiger_charge_neighbours3': np.min(
                        [atom.GetDoubleProp('_GasteigerCharge') for
                         atom in neighbours_3rd_order]),
                       'average_eucl_dist_in_mol': distmat[atom_index].mean(),
                       }

    return pd.DataFrame(dict_properties, index=[smiles])


def get_chemotypes(mol_input_list: list | np.ndarray[str | Chem.rdchem.Mol],
                   chemotype_dict: dict | None = None) -> pd.DataFrame:
    """
Identify chemotypes for a list of molecules.

This function applies a dictionary of chemotype definitions to each
molecule in the input list. Each chemotype is defined by a function
and its arguments. If no dictionary is provided, a default one is used.

Parameters
----------
mol_input_list : list or np.ndarray of str or rdkit.Chem.rdchem.Mol
    List or array of molecules in SMILES format or as RDKit Mol objects.
chemotype_dict : dict, optional
    Dictionary of chemotype definitions. Each entry should be a key
    with a tuple of (function, argument_dict). If None, a default
    dictionary is used.

Returns
-------
pd.DataFrame
    DataFrame containing the identified chemotypes for each molecule.

Examples
--------
>>> get_chemotypes(["CCO", "c1ccccc1"])
"""


    matchers = Chem.SubstructMatchParameters()
    matchers.useGenericMatchers = True

    if chemotype_dict is None:
        from mlchem.importables import chemotype_dictionary
        chemotype_dict = chemotype_dictionary

    def identify_chemotypes(mol_input: str | Chem.rdchem.Mol,
                            chemotype_dict: dict) -> dict:
        """Identify chemotypes for a single molecule."""
        results = {}
        for key, value in chemotype_dict.items():

            # Case where there is a pattern function and its arguments

            if len(value) == 2:     # Pattern function returns a tuple
                try:
                    func = value[0]
                    args = value[1]
                    # [0] is True/False, [1] (if available) are atom indexes,
                    # [2] (if available) are either query strings or notes
                    results[key] = func(mol_input, **args)[0]
                except Exception:     # Pattern function returns a single value
                    func, args = value
                    results[key] = func(mol_input, **args)
            else:
                raise ValueError(
                    "expected 1 function and 1 dictionary of arguments, found "
                    f"{len(value)} total elements instead.")
        return results

    chemotype_results = [
        identify_chemotypes(mol, chemotype_dict) for mol in mol_input_list
    ]
    return pd.DataFrame(chemotype_results,index=mol_input_list)


def get_fingerprint(
    mol_input: Chem.rdchem.Mol | str,
    fp_type: Literal['m', 'ap', 'rk', 'tt', 'mac'] = 'm',
    radius: int = 2,
    nBits: int = 2048,
    include_chirality: bool = False,
    include_bit_info: bool = False
) -> tuple | Chem.rdchem.Mol:
    """
Generate a molecular fingerprint using RDKit.

This function generates a fingerprint for a molecule using one of
several RDKit-supported types. Optionally, bit information can be
returned for interpretability.

Parameters
----------
mol_input : str or rdkit.Chem.rdchem.Mol
    Molecule in SMILES format or as an RDKit Mol object.
fp_type : {'m', 'ap', 'rk', 'tt', 'mac'}, optional
    Type of fingerprint to generate:
    - 'm': Morgan
    - 'ap': Atom Pair
    - 'rk': RDKit
    - 'tt': Topological Torsion
    - 'mac': MACCS keys
    Default is 'm'.
radius : int, optional
    Radius or path length depending on fingerprint type. Default is 2.
nBits : int, optional
    Size of the fingerprint. Default is 2048.
include_chirality : bool, optional
    Whether to include chirality. Default is False.
include_bit_info : bool, optional
    Whether to return bit information. Default is False.

Returns
-------
tuple or rdkit.DataStructs.cDataStructs.ExplicitBitVect
    Fingerprint of the molecule. If `include_bit_info` is True,
    returns a tuple (fingerprint, bit_info_dict).

Examples
--------
>>> get_fingerprint("CCO", fp_type='m', include_bit_info=True)
"""

    from rdkit.Chem import rdFingerprintGenerator, AllChem
    from mlchem.chem.manipulation import create_molecule

    try:
        mol = create_molecule(mol_input)
    except Exception as e:
        print(f"Problem encountered with: {mol_input}."
              f"Error: {e}")
        pass

    ao = rdFingerprintGenerator.AdditionalOutput()
    if include_bit_info:
        ao.AllocateBitInfoMap()

    if fp_type == 'm':
        fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=nBits,
            includeChirality=include_chirality
        )
    elif fp_type == 'ap':
        fpgen = rdFingerprintGenerator.GetAtomPairGenerator(
            maxDistance=radius, fpSize=nBits,
            includeChirality=include_chirality
        )
    elif fp_type == 'rk':
        fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(
            maxPath=radius, fpSize=nBits,
        )
    elif fp_type == 'tt':
        fpgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            torsionAtomCount=radius, fpSize=nBits,
            includeChirality=include_chirality
        )
    elif fp_type == 'mac':
        return AllChem.GetMACCSKeysFingerprint(mol)

    fp = fpgen.GetFingerprint(mol, additionalOutput=ao)
    if include_bit_info:
        return fp, ao.GetBitInfoMap()
    else:
        return fp


def get_fingerprint_df(
    mol_input_list: list[str | Chem.rdchem.Mol] |
    np.ndarray[str | Chem.rdchem.Mol],
    fp_type: Literal['m', 'ap', 'rk', 'tt', 'mac'] = 'm',
    radius: int = 2,
    nBits: int = 2048,
    include_chirality: bool = False,
    include_bit_info: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
Generate a DataFrame of fingerprints for a list of molecules.

This function computes fingerprints for each molecule in the input list
and returns them as a DataFrame. Optionally, bit information can also
be returned.

Parameters
----------
mol_input_list : list or np.ndarray of str or rdkit.Chem.rdchem.Mol
    List or array of molecules in SMILES format or as RDKit Mol objects.
fp_type : {'m', 'ap', 'rk', 'tt', 'mac'}, optional
    Type of fingerprint to generate. Default is 'm'.
radius : int, optional
    Radius or path length depending on fingerprint type. Default is 2.
nBits : int, optional
    Size of the fingerprint. Default is 2048.
include_chirality : bool, optional
    Whether to include chirality. Default is False.
include_bit_info : bool, optional
    Whether to return bit information. Default is False.

Returns
-------
pd.DataFrame or tuple of (pd.DataFrame, dict)
    DataFrame of fingerprints. If `include_bit_info` is True,
    also returns a dictionary of bit information.

Examples
--------
>>> get_fingerprint_df(["CCO", "c1ccccc1"], fp_type='m')
"""

    from mlchem.helper import create_progressive_column_names

    dict_fp = {}
    dict_bit_info = {}
    for i,m in enumerate(mol_input_list):
        try:
            identifier = m if isinstance(m, str) else Chem.MolToSmiles(m)
        except Exception as e:
            raise ValueError(f"Reading problem with molecule # {i}: {m}."
                  f"Error: {e}")
        try:
            fp = get_fingerprint(m, fp_type, radius, nBits,
                                 include_chirality, include_bit_info)
            if not include_bit_info:
                dict_fp[identifier] = fp.ToList()
            else:
                dict_fp[identifier] = fp[0].ToList()
                dict_bit_info[identifier] = fp[1]
        except Exception as e:
            raise ValueError(f"Calculation problem with molecule # {i}: {m}."
                             f"Error: {e}")

    # Dataframe from dictionary is always transposed,
    # so needs to be transposed again

    fp_names = create_progressive_column_names(fp_type,
                                               len(dict_fp[identifier]))
    fp_dataframe = pd.DataFrame(dict_fp, index=fp_names).T

    if include_bit_info:
        return fp_dataframe, dict_bit_info
    else:
        return fp_dataframe


def get_EHT_descriptors(mol_input: Chem.rdchem.Mol,
                        conf_id: int = -1) -> dict:
    """
Calculate quantum chemistry descriptors using Extended Hückel Theory (EHT).

This function computes various quantum chemistry properties for a
3D-embedded molecule using RDKit's EHT implementation. It includes
orbital energies, overlap matrices, and Mulliken charges.

More information:
https://dasher.wustl.edu/chem478/reading/extended-huckel-lowe.pdf

Parameters
----------
mol_input : rdkit.Chem.rdchem.Mol
    RDKit Mol object with at least one conformer.
conf_id : int, optional
    Conformer ID to use. Default is -1 (use the first conformer).

Returns
-------
dict
    Dictionary containing quantum chemistry descriptors:
    - AtomicCharges
    - Hamiltonian
    - OrbitalEnergies
    - OverlapMatrix
    - ReducedChargeMatrix
    - ReducedOverlapPopulationMatrix
    - FermiEnergy
    - NumElectrons
    - NumOrbitals
    - TotalEnergy

Raises
------
ValueError
    If the molecule has no conformers.

Examples
--------
>>> get_EHT_descriptors(mol_with_conformer)
"""

    from rdkit.Chem import rdEHTTools

    if mol_input.GetNumConformers() == 0:
        raise ValueError("Provided molecule has no conformers.")

    try:
        _, res = rdEHTTools.RunMol(
            mol_input,
            keepOverlapAndHamiltonianMatrices=True,
            confId=conf_id
            )
    except Exception as e:
        print(f"Problem encountered with: {mol_input}."
              f"Error: {e}")
        pass

    dictionary = {
        'AtomicCharges': res.GetAtomicCharges(),
        'Hamiltonian': res.GetHamiltonian(),
        'OrbitalEnergies': res.GetOrbitalEnergies(),
        'OverlapMatrix': res.GetOverlapMatrix(),
        'ReducedChargeMatrix': res.GetReducedChargeMatrix(),
        'ReducedOverlapPopulationMatrix':
        res.GetReducedOverlapPopulationMatrix(),
        'FermiEnergy': res.fermiEnergy,
        'NumElectrons': res.numElectrons,
        'NumOrbitals': res.numOrbitals,
        'TotalEnergy': res.totalEnergy
    }
    return dictionary
