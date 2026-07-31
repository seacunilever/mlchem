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

import numpy as np
import pandas as pd
from rdkit import Chem
from typing import Literal, Iterable, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import os
import warnings


class _ChemotypeExecution:
    """Private helpers that orchestrate cached chemotype rule execution.

    Notes
    -----
    This class is internal to chemotype calculation and is not part of the
    public API. The methods are documented to improve editor hover help and
    maintainability.
    """

    @staticmethod
    def freeze_cache_value(value: Any) -> Any:
        """Convert nested arguments into a stable hashable cache key.

        Parameters
        ----------
        value : Any
            Nested input value (dict/list/set/callable/scalar).

        Returns
        -------
        Any
            A deterministic, hashable representation suitable for cache keys.

        Examples
        --------
        >>> _ChemotypeExecution.freeze_cache_value({'a': [1, 2]})
        (('a', (1, 2)),)
        """

        if isinstance(value, dict):
            return tuple(sorted((k, _ChemotypeExecution.freeze_cache_value(v))
                                for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(_ChemotypeExecution.freeze_cache_value(v)
                         for v in value)
        if isinstance(value, set):
            return tuple(sorted(_ChemotypeExecution.freeze_cache_value(v)
                                for v in value))
        if callable(value):
            return ("__callable__", getattr(value, "__module__", ""),
                    getattr(value,
                            "__qualname__",
                            getattr(value, "__name__", "")))
        return value

    @staticmethod
    def normalise_output(result: Any) -> bool:
        """Normalize chemotype rule outputs to a boolean.

        Tuple outputs follow legacy semantics where element 0 is the truth
        value (for example: ``(True, [atom_ids], smarts)``).

        Parameters
        ----------
        result : Any
            Rule output as bool-like scalar or tuple.

        Returns
        -------
        bool
            Normalized boolean result.

        Examples
        --------
        >>> _ChemotypeExecution.normalise_output((True, [0, 1]))
        True
        >>> _ChemotypeExecution.normalise_output(False)
        False
        """

        if isinstance(result, tuple):
            if len(result) == 0:
                return False
            return bool(result[0])
        return bool(result)

    @staticmethod
    def call_rule_with_cache(func: Callable[..., Any],
                             mol: Chem.rdchem.Mol,
                             kwargs: dict[str, Any],
                             rule_cache: dict[tuple[Any, Any], Any]) -> Any:
        """Call a chemotype rule once per molecule for a specific arg signature.

        Parameters
        ----------
        func : Callable[..., Any]
            Chemotype rule function.
        mol : rdkit.Chem.rdchem.Mol
            Target molecule.
        kwargs : dict[str, Any]
            Keyword arguments passed to the rule function.
        rule_cache : dict[tuple[Any, Any], Any]
            Per-molecule cache mapping function/signature to computed result.

        Returns
        -------
        Any
            Cached or freshly computed rule output.

        Notes
        -----
        The method first tries ``func(target=mol, **kwargs)`` and falls back
        to ``func(mol, **kwargs)`` for backward compatibility with rule
        functions that only accept positional target input.

        Raises
        ------
        Exception
            Re-raises exceptions from the rule function if evaluation fails.

        Examples
        --------
        >>> # Internal use inside get_chemotypes evaluation loop
        >>> # _ChemotypeExecution.call_rule_with_cache(rule, mol, {}, cache)
        """

        cache_key = (func, _ChemotypeExecution.freeze_cache_value(kwargs))
        if cache_key in rule_cache:
            return rule_cache[cache_key]

        try:
            result = func(target=mol, **kwargs)
        except TypeError:
            result = func(mol, **kwargs)

        rule_cache[cache_key] = result
        return result

    @staticmethod
    def to_atom_count(result: Any) -> int:
        """Extract atom-count semantics from chemotype rule outputs.

        Parameters
        ----------
        result : Any
            Rule output as tuple ``(bool, atom_ids, ...)`` or scalar bool-like.

        Returns
        -------
        int
            Number of matched atoms if available, otherwise 1/0 from bool cast.

        Examples
        --------
        >>> _ChemotypeExecution.to_atom_count((True, [2, 5, 9]))
        3
        >>> _ChemotypeExecution.to_atom_count(False)
        0
        """

        if isinstance(result, tuple) and len(result) > 1:
            return len(result[1])
        return int(bool(result))

    @staticmethod
    def evaluate_rule(func: Callable[..., Any],
                      args: dict[str, Any],
                      mol: Chem.rdchem.Mol,
                      rule_cache: dict[tuple[Any, Any], Any],
                      abs_rule: Callable[..., Any],
                      rel_rule: Callable[..., Any]) -> Any:
        """Evaluate one chemotype rule with memoization-aware fast paths.

        This method optimizes known derived rules
        (absolute/relative fraction thresholds) by reusing cached base-pattern
        evaluations rather than re-running equivalent pattern checks.

        Parameters
        ----------
        func : Callable[..., Any]
            Rule function to evaluate.
        args : dict[str, Any]
            Rule argument mapping from the chemotype dictionary.
        mol : rdkit.Chem.rdchem.Mol
            Molecule to evaluate.
        rule_cache : dict[tuple[Any, Any], Any]
            Per-molecule cache of previously evaluated rule calls.
        abs_rule : Callable[..., Any]
            Reference absolute-fraction rule function.
        rel_rule : Callable[..., Any]
            Reference relative-fraction rule function.

        Returns
        -------
        Any
            Raw rule output (later normalized by ``normalise_output``).

        Raises
        ------
        KeyError
            If a derived rule is missing required dictionary keys, for example
            ``func``, ``func1``, ``func2``, or ``threshold``.
        Exception
            Re-raises exceptions from nested rule evaluations.

        Examples
        --------
        >>> # Internal use in get_chemotypes:
        >>> # raw = _ChemotypeExecution.evaluate_rule(func, args, mol, cache, abs_rule, rel_rule)
        """

        if func is abs_rule:
            hidden_func = args.get('hidden_pattern_function')
            numerator_kwargs = {}
            if hidden_func is not None:
                numerator_kwargs['pattern_function'] = hidden_func

            numerator_result = _ChemotypeExecution.call_rule_with_cache(
                args['func'], mol, numerator_kwargs, rule_cache
            )
            pattern_atoms = _ChemotypeExecution.to_atom_count(numerator_result)
            total_atoms = mol.GetNumAtoms()
            threshold = args['threshold']
            return total_atoms > 0 and (pattern_atoms / total_atoms) > threshold

        if func is rel_rule:
            hidden_func = args.get('hidden_pattern_function')
            numerator_kwargs = {}
            if hidden_func is not None:
                numerator_kwargs['pattern_function'] = hidden_func

            denominator_result = _ChemotypeExecution.call_rule_with_cache(
                args['func2'], mol, {}, rule_cache
            )
            numerator_result = _ChemotypeExecution.call_rule_with_cache(
                args['func1'], mol, numerator_kwargs, rule_cache
            )

            denominator_atoms = _ChemotypeExecution.to_atom_count(
                denominator_result
            )
            numerator_atoms = _ChemotypeExecution.to_atom_count(numerator_result)
            threshold = args['threshold']
            return denominator_atoms > 0 and \
                (numerator_atoms / denominator_atoms) > threshold

        return _ChemotypeExecution.call_rule_with_cache(
            func, mol, args, rule_cache
        )


def _build_fingerprint_generator(
    fp_type: Literal['m', 'ap', 'rk', 'tt', 'mac'],
    radius: int,
    nBits: int,
    include_chirality: bool
):
    """Create an RDKit fingerprint generator for supported fingerprint types."""

    from rdkit.Chem import rdFingerprintGenerator

    if fp_type == 'm':
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=nBits,
            includeChirality=include_chirality
        )
    if fp_type == 'ap':
        return rdFingerprintGenerator.GetAtomPairGenerator(
            maxDistance=radius, fpSize=nBits,
            includeChirality=include_chirality
        )
    if fp_type == 'rk':
        return rdFingerprintGenerator.GetRDKitFPGenerator(
            maxPath=radius, fpSize=nBits,
        )
    if fp_type == 'tt':
        return rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            torsionAtomCount=radius, fpSize=nBits,
            includeChirality=include_chirality
        )
    if fp_type == 'mac':
        return None

    raise ValueError(
        f"Unsupported fp_type: {fp_type}. "
        "Use one of {'m', 'ap', 'rk', 'tt', 'mac'}."
    )


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

        mol_input_h = create_molecule(mol_input=mol_input,
                                      add_hydrogens=True,
                                      show=False,
                                      solid_sticks=True,
                                      is_3d=True,
                                      optimise=True)
        return CalcMolDescriptors3D(mol_input_h)

    rows = []
    identifiers = []
    for i, mol_input in enumerate(mol_input_list):
        try:
            identifier = mol_input if isinstance(mol_input, str) \
                else Chem.MolToSmiles(mol_input)
        except Exception as e:
            raise ValueError(f"Reading problem with molecule # {i}: {mol_input}."
                             f"Error: {e}") from e

        try:
            desc_2d = get_desc_2d(mol_input)
            if include_3D:
                row = merge_dicts_with_duplicates(desc_2d,
                                                  get_desc_3d(mol_input))
            else:
                row = desc_2d
        except Exception as e:
            raise ValueError(
                f"Descriptor calculation problem with molecule # {i}: "
                f"{mol_input}. Error: {e}"
            ) from e

        rows.append(row)
        identifiers.append(identifier)

    df = pd.DataFrame(rows, index=identifiers)

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
            warnings.warn(
                f"Problem encountered with: {mol_input}. Error: {e}",
                RuntimeWarning,
                stacklevel=2
            )
            return [None] * len(calculator.descriptors)

    rows = []
    identifiers = []
    for i, mol_input in enumerate(mol_input_list):
        try:
            identifier = mol_input if isinstance(mol_input, str) \
                else Chem.MolToSmiles(mol_input)
        except Exception as e:
            raise ValueError(f"Reading problem with molecule # {i}: {mol_input}."
                             f"Error: {e}") from e

        rows.append(get_desc(mol_input, calculator=calc))
        identifiers.append(identifier)

    df_desc = pd.DataFrame(rows, index=identifiers)
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
    if atom_index < 0 or atom_index >= tot_atoms:
        raise IndexError(
            f"Atom index ({atom_index}) outside valid range [0, {tot_atoms - 1}]"
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
                   chemotype_dict: dict | None = None,
                   n_jobs: int = 1) -> pd.DataFrame:
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
n_jobs : int, optional
    Number of worker threads used to process molecules. Use values
    greater than 1 to enable parallel execution. Use -1 to consume
    all available CPU cores. Default is 1.

Returns
-------
pd.DataFrame
    DataFrame containing the identified chemotypes for each molecule.

Raises
------
ValueError
    If ``n_jobs`` is 0 or less than -1.
ValueError
    If a chemotype rule does not contain exactly two items
    (function, arguments).
TypeError
    If a chemotype rule function is not callable or its argument payload is
    not a dictionary.
Exception
    Propagates molecule parsing/evaluation exceptions from underlying rule
    functions and RDKit utilities.

Notes
-----
- Output row ordering is deterministic and follows ``mol_input_list`` even
  when ``n_jobs > 1`` because ``ThreadPoolExecutor.map`` preserves order.
- ``n_jobs=-1`` maps to ``os.cpu_count()`` (or 1 if unavailable).

Examples
--------
>>> get_chemotypes(["CCO", "c1ccccc1"])
>>> get_chemotypes(["CCO", "CCN", "COCC"], n_jobs=-1)
>>> custom = {'O_rule': [lambda target: target.HasSubstructMatch(Chem.MolFromSmarts('[#8]')), {}]}
>>> get_chemotypes([Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCC")], chemotype_dict=custom)
"""

    from mlchem.chem.manipulation import create_molecule
    from mlchem.chem.manipulation import PatternRecognition as pr

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    elif n_jobs < -1 or n_jobs == 0:
        raise ValueError("'n_jobs' must be -1 or a positive integer.")

    if chemotype_dict is None:
        from mlchem.importables import chemotype_dictionary
        chemotype_dict = chemotype_dictionary

    abs_rule = pr.Base.pattern_abs_fraction_greater_than
    rel_rule = pr.Base.pattern_rel_fraction_greater_than

    mol_input_list = list(mol_input_list)
    mol_list = [create_molecule(mol_input) if isinstance(mol_input, str)
                else mol_input for mol_input in mol_input_list]

    def identify_chemotypes(mol_input: Chem.rdchem.Mol,
                            chemotype_dict: dict,
                            abs_rule,
                            rel_rule) -> dict:
        """Identify chemotypes for one molecule.

        Parameters
        ----------
        mol_input : rdkit.Chem.rdchem.Mol
            Molecule to classify.
        chemotype_dict : dict
            Mapping of chemotype names to ``[function, kwargs]`` entries.
        abs_rule : callable
            Reference for absolute-fraction optimization path.
        rel_rule : callable
            Reference for relative-fraction optimization path.

        Returns
        -------
        dict
            Mapping of chemotype names to boolean presence flags.
        """
        results = {}
        rule_cache = {}
        for key, value in chemotype_dict.items():
            if len(value) != 2:
                raise ValueError(
                    "expected 1 function and 1 dictionary of arguments, found "
                    f"{len(value)} total elements instead.")

            func, args = value
            if not callable(func):
                raise TypeError(f"Chemotype rule '{key}' has a non-callable function.")
            if not isinstance(args, dict):
                raise TypeError(f"Chemotype rule '{key}' has non-dict arguments.")

            result = _ChemotypeExecution.evaluate_rule(
                func=func,
                args=args,
                mol=mol_input,
                rule_cache=rule_cache,
                abs_rule=abs_rule,
                rel_rule=rel_rule
            )
            results[key] = _ChemotypeExecution.normalise_output(result)
        return results

    if n_jobs == 1:
        chemotype_results = [
            identify_chemotypes(mol, chemotype_dict, abs_rule, rel_rule)
            for mol in mol_list
        ]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            chemotype_results = list(
                pool.map(
                    lambda mol: identify_chemotypes(mol,
                                                    chemotype_dict,
                                                    abs_rule,
                                                    rel_rule),
                    mol_list
                )
            )

    return pd.DataFrame(chemotype_results, index=mol_input_list)


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
        raise ValueError(f"Problem encountered with: {mol_input}. Error: {e}") from e

    fpgen = _build_fingerprint_generator(fp_type,
                                         radius,
                                         nBits,
                                         include_chirality)
    if fpgen is None:
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        if include_bit_info:
            return fp, {}
        return fp

    ao = rdFingerprintGenerator.AdditionalOutput()
    if include_bit_info:
        ao.AllocateBitInfoMap()

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

    from rdkit.Chem import AllChem
    from mlchem.chem.manipulation import create_molecule
    from mlchem.helper import create_progressive_column_names

    mol_input_list = list(mol_input_list)
    fpgen = _build_fingerprint_generator(fp_type,
                                         radius,
                                         nBits,
                                         include_chirality)
    n_columns = 167 if fp_type == 'mac' else nBits
    fp_names = create_progressive_column_names(fp_type, n_columns)

    if len(mol_input_list) == 0:
        empty_df = pd.DataFrame(columns=fp_names)
        if include_bit_info:
            return empty_df, {}
        return empty_df

    rows = []
    identifiers = []
    dict_bit_info = {}
    duplicate_counts = {}
    for i, m in enumerate(mol_input_list):
        try:
            identifier = m if isinstance(m, str) else Chem.MolToSmiles(m)
        except Exception as e:
            raise ValueError(f"Reading problem with molecule # {i}: {m}."
                             f"Error: {e}") from e
        try:
            mol = create_molecule(m)
            if fpgen is None:
                fp = AllChem.GetMACCSKeysFingerprint(mol)
                bit_info = {}
            else:
                fp = fpgen.GetFingerprint(mol)
                bit_info = {}
                if include_bit_info:
                    from rdkit.Chem import rdFingerprintGenerator
                    ao = rdFingerprintGenerator.AdditionalOutput()
                    ao.AllocateBitInfoMap()
                    fp = fpgen.GetFingerprint(mol, additionalOutput=ao)
                    bit_info = ao.GetBitInfoMap()
        except Exception as e:
            raise ValueError(f"Calculation problem with molecule # {i}: {m}."
                             f"Error: {e}") from e

        rows.append(fp.ToList())
        identifiers.append(identifier)
        if include_bit_info:
            duplicate_count = duplicate_counts.get(identifier, 0)
            duplicate_counts[identifier] = duplicate_count + 1
            if duplicate_count == 0:
                info_key = str(identifier)
            else:
                info_key = f"{identifier}__dup{duplicate_count + 1}"
            dict_bit_info[info_key] = bit_info

    fp_dataframe = pd.DataFrame(rows, index=identifiers, columns=fp_names)

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
        success, res = rdEHTTools.RunMol(
            mol_input,
            keepOverlapAndHamiltonianMatrices=True,
            confId=conf_id
            )
    except Exception as e:
        raise RuntimeError(
            f"Problem encountered with: {mol_input}. Error: {e}"
        ) from e

    if not success:
        raise RuntimeError("EHT calculation failed with the provided molecule.")

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
