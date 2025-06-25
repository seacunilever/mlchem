
from rdkit import Chem
import numpy as np
import pandas as pd
from typing import Literal, Optional, Iterable, Callable
from IPython.display import display


def mol_from_string(mol_input: str) -> Chem.rdchem.Mol:
    """
Convert a molecular string (InChI or SMILES) to an RDKit molecule object.

Attempts to interpret the input string as an InChI first, then as a SMILES.
Raises a ValueError if both conversions fail.

Parameters
----------
mol_input : str
    A string representing a molecule in InChI or SMILES format.

Returns
-------
rdkit.Chem.rdchem.Mol
    RDKit molecule object corresponding to the input string.

Raises
------
ValueError
    If the input string is not a valid InChI or SMILES.
"""

    mol = Chem.MolFromInchi(mol_input)
    if mol is not None:
        return mol
    mol = Chem.MolFromSmiles(mol_input)
    if mol is not None:
        return mol

    raise ValueError(f'{mol_input}: Molecule not recognised as '
                     'a valid SMILES or InCHI sequence.')


def smiles_to_inchi(smiles: str) -> str:
    """
Convert a SMILES string to an InChI string.

Parameters
----------
smiles : str
    A string representing a molecule in SMILES format.

Returns
-------
str
    An InChI string corresponding to the input SMILES.

Raises
------
ValueError
    If the SMILES string is invalid.
"""

    try:
        return Chem.MolToInchi(Chem.MolFromSmiles(smiles))
    except Exception:
      raise ValueError(f"Invalid SMILES: {smiles}")


def create_molecule(mol_input: str | np.str_ | Chem.rdchem.Mol,
                    add_hydrogens: bool = False,
                    show: bool = False,
                    solid_sticks: bool = False,
                    is_3d: bool = False,
                    embedding_params: Literal['ETDG',
                                              'ETKDG',
                                              'ETKDGv2',
                                              'ETKDGv3',
                                              ] = 'ETKDGv3',
                    size: tuple = [300, 300],
                    optimise: bool = True,
                    optimiser: Literal['MMFF94', 'UFF'] = 'MMFF94',
                    random_seed: int = 0xf00d) -> Chem.rdchem.Mol | None:
    """
Create and optionally display a molecule from a SMILES string or RDKit Mol object.

Supports 2D/3D visualisation, hydrogen addition, geometry optimisation,
and rendering options.

Parameters
----------
mol_input : str or numpy.str_ or rdkit.Chem.rdchem.Mol
    The molecule input as a SMILES string, numpy string, or RDKit Mol object.
add_hydrogens : bool, optional
    Whether to add hydrogens to the molecule. Default is False.
show : bool, optional
    Whether to display the molecule. Default is False.
solid_sticks : bool, optional
    Whether to render the molecule as solid sticks. Default is False.
is_3d : bool, optional
    Whether to generate a 3D representation. Default is False.
embedding_params : {'ETDG', 'ETKDG', 'ETKDGv2', 'ETKDGv3'}, optional
    Embedding parameters for 3D generation. Default is 'ETKDGv3'.
size : tuple, optional
    Size of the displayed image. Default is (300, 300).
optimise : bool, optional
    Whether to optimise the geometry. Default is True.
optimiser : {'MMFF94', 'UFF'}, optional
    Optimisation method. Default is 'MMFF94'.
random_seed : int, optional
    Random seed for embedding. Default is 0xf00d.

Returns
-------
rdkit.Chem.rdchem.Mol or None
    The processed molecule object, or None if only visualisation is requested.
"""

    from rdkit.Chem import AllChem, rdDepictor, rdDistGeom, Draw
    from rdkit.Chem.Draw import IPythonConsole as ipc

    ipc.ipython_3d = True

    if isinstance(mol_input, str) or isinstance(mol_input, np.str_):
        mol = mol_from_string(mol_input)
    elif isinstance(mol_input, Chem.rdchem.Mol):
        mol = mol_input
    else:
        raise TypeError(
            "'mol_input' argument is neither a string nor an "
            "RDKit mol object. Type: %s" % type(mol_input)
            )

    if add_hydrogens:
        mol = AllChem.AddHs(mol)

    if solid_sticks:
        params = getattr(rdDistGeom, embedding_params)()
        params.randomSeed = random_seed
        rdDistGeom.EmbedMolecule(mol, params)
        if optimise is True:
            if optimiser == 'MMFF94':
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                AllChem.UFFOptimizeMolecule(mol)

        if is_3d is True:     # solid sticks, 3d
            ipc.ipython_3d = True
            mol.GetConformer().Set3D(True)

        else:     # solid sticks, 2d
            rdDepictor.Compute2DCoords(mol)
            mol.GetConformer().Set3D(True)
    else:
        if is_3d is True:     # thin sticks, 3d
            ipc.ipython_3d = True
            params = getattr(rdDistGeom, embedding_params)()
            params.randomSeed = random_seed
            AllChem.EmbedMolecule(mol, params)
            if optimise is True:
                if optimiser == 'MMFF94':
                    AllChem.MMFFOptimizeMolecule(mol)
                else:
                    AllChem.UFFOptimizeMolecule(mol)
            mol.GetConformer().Set3D(False)
        else:
            pass     # thin sticks, 2d

    if show is True:
        img = Chem.Draw.MolToImage(mol, size)
        if solid_sticks is False:
            display(img)
        else:
            return ipc.drawMol3D(m=mol, size=size)
    else:
        return mol


def kekulise_smiles(smiles: str) -> str:
    """
Convert a SMILES string to its Kekulé form.

Parameters
----------
smiles : str
    A SMILES string.

Returns
-------
str
    The Kekulé form of the SMILES string.
"""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), kekuleSmiles=True)


def unkekulise_smiles(smiles: str) -> str:
    """
Convert a Kekulé SMILES string to its canonical form.

Parameters
----------
smiles : str
    A Kekulé SMILES string.

Returns
-------
str
    The canonical SMILES string.
"""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), kekuleSmiles=False)


def smarts_from_string(string: str) -> str:
    """
Convert a SMILES or InChI string into a SMARTS string.

Parameters
----------
string : str
    A string representing a molecule in SMILES or InChI format.

Returns
-------
str
    A SMARTS string representing the molecular pattern.
"""
    return Chem.MolToSmarts(create_molecule(string))


def mol_to_binary(mol: Chem.rdchem.Mol) -> bytes:
    """
Convert an RDKit molecule object to its binary representation.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    An RDKit molecule object.

Returns
-------
bytes
    Binary representation of the molecule.
"""
    return mol.ToBinary()


def generate_resonance(smi: str,
                       save: bool = False,
                       path_name: str = '') -> list[bytes]:
    """
Generate resonance structures for a SMILES string and optionally save images.

Parameters
----------
smi : str
    A SMILES string representing the molecule.
save : bool, optional
    Whether to save the images. Default is False.
path_name : str, optional
    Directory path to save images. Default is current directory.

Returns
-------
list of bytes
    List of binary image data for the resonance structures.
"""


    from rdkit.Chem import Draw
    import os

    def generate_images(mols: list[Chem.rdchem.Mol | np.ndarray],
                        path_name: str,
                        should_kekulise_aromatic_smiles: bool = False,
                        size: tuple = (200, 100),
                        save=save
                        ) -> list[bytes]:
        """Generate images for a list of molecules."""
        imgs = []

        for i, m in enumerate(mols):
            smi = Chem.MolToSmiles(m)
            print(f'{smi}\n')
            # Only kekulise molecule if it has aromatic atoms
            has_aromatic_atoms = any(a.GetIsAromatic() for a in m.GetAtoms())
            if not has_aromatic_atoms:
                should_kekulise_aromatic_smiles = False
            drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            drawer.drawOptions().prepareMolsBeforeDrawing = False
            m = Draw.rdMolDraw2D.PrepareMolForDrawing(
                m,
                kekulize=should_kekulise_aromatic_smiles
                )
            drawer.DrawMolecule(m)
            drawer.FinishDrawing()
            img = drawer.GetDrawingText()
            imgs.append(img)
            if save:
                with open(
                    os.path.join('./', path_name, f'{smi}_{i}.png'),
                    "wb") as hnd:
                    hnd.write(img)
        return imgs

    def unset_aromatic_flags(m: Chem.Mol) -> Chem.Mol:
        """Unset aromatic flags for bonds and atoms in the molecule."""
        for b in m.GetBonds():
            if b.GetBondType() != Chem.BondType.AROMATIC and b.GetIsAromatic():
                b.SetIsAromatic(False)
                b.GetBeginAtom().SetIsAromatic(False)
                b.GetEndAtom().SetIsAromatic(False)
        return m

    mol = Chem.MolFromSmiles(smi)
    suppl = Chem.ResonanceMolSupplier(mol, Chem.KEKULE_ALL)
    mols = [mol] + [unset_aromatic_flags(m) for m in suppl]
    imgs = generate_images(
        mols=mols, path_name=path_name, should_kekulise_aromatic_smiles=False
        )
    return imgs


def neutralise_mol(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    """
Neutralise an RDKit molecule by removing formal charges.

Based on RDKit's neutralisation recipe.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    An RDKit molecule object.

Returns
-------
rdkit.Chem.rdchem.Mol
    The neutralised molecule.
"""


    pattern = Chem.MolFromSmarts(
        "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
         )
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def remove_smarts_pattern(mol: Chem.rdchem.Mol,
                          smarts_string: str
                          ) -> Chem.rdchem.Mol:
    """
Remove substructures matching a SMARTS pattern from a molecule.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    An RDKit molecule object.
smarts_string : str
    A SMARTS pattern to remove.

Returns
-------
rdkit.Chem.rdchem.Mol
    The molecule with matching substructures removed.
"""
    pattern = Chem.MolFromSmarts(smarts_string)
    while mol.HasSubstructMatch(pattern):
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            emol = Chem.EditableMol(mol)
            for idx in sorted(match, reverse=True):
                emol.RemoveAtom(idx)
            mol = emol.GetMol()
            break
    return mol


class PropManager:
    """
Manage molecular properties using RDKit.

The `PropManager` class provides a structured interface for manipulating
molecular properties via RDKit. It is organised into five logical
subsections: `Base`, `Mol`, `Atom`, `Bond`, and `Conformation`, each
containing methods relevant to their respective domains.

Original RDKit reference:
https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#

Base Methods
------------
- assign_atom_mapnumbers(mol, atom_ids)
    Assign map numbers to atoms in a molecule.
- assign_atom_labels(mol, prop_values, atom_ids)
    Assign labels to atoms in a molecule.
- assign_atom_notes(mol, prop_values, atom_ids)
    Assign notes to atoms in a molecule.
- assign_bond_notes(mol, prop_values, bond_ids)
    Assign notes to bonds in a molecule.
- clear_all_atomprops(mol)
    Clear all properties from atoms in a molecule.
- clear_prop(rdkit_obj, prop)
    Clear a property from an RDKit object.
- get_props_dict(rdkit_obj)
    Get all properties of an RDKit object as a dictionary.
- get_prop_names(rdkit_obj)
    Get the names of all properties of an RDKit object.
- get_prop(rdkit_obj, prop)
    Get the value of a property from an RDKit object.
- set_prop(rdkit_obj, prop_name, prop_val)
    Set a property on an RDKit object.
- get_owning_mol(rdkit_obj)
    Get the molecule that owns the RDKit object.

Mol Methods
-----------
- get_atoms(mol)
    Get the atoms of a molecule.
- get_atoms_from_idx(mol, idx)
    Retrieve atom(s) from a molecule by index.
- get_bonds(mol)
    Get the bonds of a molecule.
- get_bonds_from_idx(mol, idx)
    Retrieve bond(s) from a molecule by index.
- get_bond_between_atoms(mol, idx1, idx2)
    Retrieve the bond between two atoms.
- get_coordinates(conf_or_mol, is_3d, canonOrient, bondLength)
    Get coordinates of a molecule or conformer.
- get_conformer(mol, id)
    Get a specific conformer from a molecule.
- get_conformers(mol)
    Get all conformers from a molecule.
- get_conf_ids(mol)
    Get all conformer IDs from a molecule.
- get_distance_matrix(mol, is_3d)
    Get the distance matrix of a molecule.
- get_gasteiger_charges(mol, atom_ids, nIter)
    Compute Gasteiger charges for specified atoms.
- get_stereogroups(mol)
    Get stereochemistry groups of a molecule.
- remove_conformer(mol, id)
    Remove a specific conformer.
- remove_all_conformers(mol)
    Remove all conformers from a molecule.

Atom Methods
------------
- clear_atomprops(atom)
    Clear all properties from an atom.
- get_atomic_num(atom)
    Get the atomic number.
- get_bonds(atom)
    Get bonds connected to an atom.
- get_degree(atom)
    Get the degree of an atom.
- get_total_degree(atom)
    Get total degree including hydrogens.
- get_explicit_valence(atom)
    Get explicit valence.
- get_implicit_valence(atom)
    Get implicit valence.
- get_total_valence(atom)
    Get total valence.
- has_valence_violation(atom)
    Check for valence violations.
- get_formal_charge(atom)
    Get formal charge.
- get_hybridisation(atom)
    Get hybridisation state.
- get_idx(atom)
    Get atom index.
- get_neighbours(atom, order)
    Get neighbours up to a given order.
- is_aromatic(atom)
    Check if atom is aromatic.
- is_in_ring(atom)
    Check if atom is in a ring.
- is_in_ring_size(atom, size)
    Check if atom is in a ring of a specific size.
- get_mass(atom)
    Get atomic mass.
- get_num_explicit_h(atom)
    Get number of explicit hydrogens.
- get_num_implicit_h(atom)
    Get number of implicit hydrogens.
- get_tot_h(atom)
    Get total number of hydrogens.
- get_num_radical_electrons(atom)
    Get number of radical electrons.
- set_atom_map_num(atom, num)
    Set atom map number.
- set_formal_charge(atom, charge)
    Set formal charge.
- set_is_aromatic(atom, decision)
    Set aromaticity.
- set_num_explicit_h(atom, num)
    Set number of explicit hydrogens.
- set_num_radical_electrons(atom, num)
    Set number of radical electrons.

Bond Methods
------------
- get_begin_atom(bond)
    Get the starting atom of a bond.
- get_begin_atom_idx(bond)
    Get index of the starting atom.
- get_bond_type(bond)
    Get bond type.
- get_end_atom(bond)
    Get the ending atom of a bond.
- get_end_atom_idx(bond)
    Get index of the ending atom.
- get_idx(bond)
    Get bond index.
- get_other_atom(bond, atom)
    Get the other atom in a bond.
- get_other_atom_idx(bond, idx)
    Get the index of the other atom.
- get_valence_contribution(bond, atom)
    Get valence contribution of a bond.
- is_aromatic(bond)
    Check if bond is aromatic.
- is_conjugated(bond)
    Check if bond is conjugated.
- is_in_ring(bond)
    Check if bond is in a ring.
- is_in_ring_size(bond, size)
    Check if bond is in a ring of a specific size.
- set_is_aromatic(bond, decision)
    Set aromaticity of a bond.

Conformation Methods
--------------------
- straighten_mol_2d(mol)
    Straighten the 2D depiction of a molecule.
- add_conformer(mol, conformer, assignId)
    Add a conformer and return its ID.
- generate_conformers(...)
    Generate conformers for a molecule.
- display_conformers(conf, size)
    Display conformers in 3D.
- display_3dmols_overlapped(...)
    Display multiple 3D molecules overlapped.
- canonicalise_conformer(conf, ignoreHs)
    Canonicalise a conformer.
- canonicalise_mol_conformers(mol, ignoreHs)
    Canonicalise all conformers.
- calculate_conformer_energy_from_mol(mol, conf_id, forcefield)
    Calculate energy of a conformer.
- optimise_conformers(mol, force_field, max_iter)
    Optimise all conformers.
- optimise_molecule(mol, conf_id, force_field, max_iter)
    Optimise a specific conformer.
- get_shape_descriptors(conf_or_mol, include_masses, is_3d)
    Calculate shape descriptors.
"""

    class Base:

        @staticmethod
        def assign_atom_mapnumbers(mol: Chem.rdchem.Mol,
                                   atom_ids: Iterable[int] = ()) -> None:
            """
            Assign map numbers to atoms in a molecule.

            If `atom_ids` is provided, only those atoms will be assigned 
            map numbers.
            Otherwise, all atoms in the molecule will be assigned map numbers.

            Parameters
            ----------
            mol : rdkit.Chem.rdchem.Mol
                The RDKit molecule object.
            atom_ids : Iterable[int], optional
                Atom indices to assign map numbers to. Default is all atoms.

            Returns
            -------
            None
            """

            if not atom_ids:
                atoms = PropManager.Mol.get_atoms(mol)
                atom_ids = range(len(atoms))
            else:
                atoms = PropManager.Mol.get_atoms_from_idx(mol, atom_ids)
            for i, atom in zip(atom_ids, atoms):
                PropManager.Base.set_prop(atom, 'molAtomMapNumber', str(i))

        @staticmethod
        def assign_atom_labels(
            mol: Chem.rdchem.Mol,
            prop_values: str | int | float | Iterable | None = None,
            atom_ids: Iterable[int] = ()
        ) -> None:
            """
            Assign labels to atoms in a molecule.

            If `atom_ids` is provided, only those atoms will be labelled.
            Otherwise, all atoms will be labelled. If `prop_values` is not provided,
            atom indices will be used as labels.

            Parameters
            ----------
            mol : rdkit.Chem.rdchem.Mol
                The RDKit molecule object.
            prop_values : str, int, float, Iterable, optional
                Values to assign as labels. If None, atom indices are used.
            atom_ids : Iterable[int], optional
                Atom indices to assign labels to. Default is all atoms.

            Returns
            -------
            None
            """

            if prop_values is None:
                prop_values = [a.GetIdx() for a in mol.GetAtoms()]

            if not atom_ids:
                assert len(prop_values) == len(mol.GetAtoms()), \
                 "Number of properties and molecule atoms do not match."
                atoms = PropManager.Mol.get_atoms(mol)
                atom_ids = range(len(atoms))
            else:
                assert len(prop_values) == len(atom_ids), \
                 "Number of properties and atom ids"
                "entered do not match."
                atoms = PropManager.Mol.get_atoms_from_idx(mol, atom_ids)
            for prop, i, atom in zip(prop_values, atom_ids, atoms):
                PropManager.Base.set_prop(atom, 'atomLabel', str(prop))

        @staticmethod
        def assign_atom_notes(
            mol: Chem.rdchem.Mol,
            prop_values: str | int | float | Iterable | None = None,
            atom_ids: Iterable[int] = ()
        ) -> None:
            """
Assign notes to atoms in a molecule.

If `atom_ids` is provided, only those atoms will be annotated.
Otherwise, all atoms will be annotated. If `prop_values` is not provided,
atom indices will be used as notes.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.
prop_values : str, int, float, Iterable, optional
    Values to assign as notes. If None, atom indices are used.
atom_ids : Iterable[int], optional
    Atom indices to assign notes to. Default is all atoms.

Returns
-------
None
"""

            if prop_values is None:
                prop_values = [a.GetIdx() for a in mol.GetAtoms()]

            if not atom_ids:
                assert len(prop_values) == len(mol.GetAtoms()), \
                 "Number of properties and molecule atoms do not match."
                atoms = PropManager.Mol.get_atoms(mol)
                atom_ids = range(len(atoms))
            else:
                assert len(prop_values) == len(atom_ids), \
                 "Number of properties and atom "
                "ids entered do not match."
                atoms = PropManager.Mol.get_atoms_from_idx(mol, atom_ids)
            for prop, i, atom in zip(prop_values, atom_ids, atoms):
                PropManager.Base.set_prop(atom, 'atomNote', str(prop))

        @staticmethod
        def assign_bond_notes(
            mol: Chem.rdchem.Mol,
            prop_values: str | int | float | Iterable,
            bond_ids: Iterable[int] = ()
        ) -> None:
            """
Assign notes to bonds in a molecule.

If `bond_ids` is provided, only those bonds will be annotated.
Otherwise, all bonds will be annotated.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.
prop_values : str, int, float, Iterable
    Values to assign as notes.
bond_ids : Iterable[int], optional
    Bond indices to assign notes to. Default is all bonds.

Returns
-------
None
"""

            if not bond_ids:
                assert len(prop_values) == len(mol.GetBonds()), \
                     "Number of properties and molecule"
                " bonds do not match."
                bonds = PropManager.Mol.get_bonds(mol)
                bond_ids = range(len(bonds))
            else:
                assert len(prop_values) == len(bond_ids), \
                    "Number of properties and bond"
                " ids entered do not match."
                bonds = PropManager.Mol.get_bonds_from_idx(mol, bond_ids)
            for prop, i, bond in zip(prop_values, bond_ids, bonds):
                PropManager.Base.set_prop(bond, 'bondNote', str(prop))

        @staticmethod
        def clear_all_atomprops(mol: Chem.rdchem.Mol) -> None:
            """
            Clear all properties from atoms in a molecule.

            Parameters
            ----------
            mol : rdkit.Chem.rdchem.Mol
                The RDKit molecule object.

            Returns
            -------
            None
            """

            for atom in mol.GetAtoms():
                PropManager.Atom.clear_atomprops(atom)

        @staticmethod
        def clear_prop(rdkit_obj: Chem.rdchem.Mol |
                       Chem.rdchem.Atom |
                       Chem.rdchem.Bond |
                       Chem.rdchem.Conformer,
                       prop: str) -> None:
            """
            Clear a property from an RDKit object.

            Parameters
            ----------
            rdkit_obj : rdkit.Chem.rdchem.Mol or Atom or Bond or Conformer
                The RDKit object from which the property will be removed.
            prop : str
                The name of the property to remove.

            Returns
            -------
            None
            """

            return rdkit_obj.ClearProp(prop)

        @staticmethod
        def get_props_dict(rdkit_obj: Chem.rdchem.Mol |
                           Chem.rdchem.Atom |
                           Chem.rdchem.Bond |
                           Chem.rdchem.Conformer) -> dict:
            """
Get all properties of an RDKit object as a dictionary.

Parameters
----------
rdkit_obj : rdkit.Chem.rdchem.Mol or Atom or Bond or Conformer
    The RDKit object to inspect.

Returns
-------
dict
    A dictionary of all properties associated with the object.
"""

            hidden_dict = rdkit_obj.GetPropsAsDict(includePrivate=True,
                                                   includeComputed=True)
            prop_tuple = [(key, value) for key, value in zip(
                hidden_dict.keys(),
                hidden_dict.values()
                )
                ]
            prop_dict = dict((k, v) for k, v in prop_tuple)
            return prop_dict

        @staticmethod
        def get_prop_names(rdkit_obj: Chem.rdchem.Mol |
                           Chem.rdchem.Atom |
                           Chem.rdchem.Bond |
                           Chem.rdchem.Conformer) -> list:
            """
Get the names of all properties of an RDKit object.

Parameters
----------
rdkit_obj : rdkit.Chem.rdchem.Mol or Atom or Bond or Conformer
    The RDKit object to inspect.

Returns
-------
list
    A list of property names.
"""

            return [prop for prop in rdkit_obj.GetPropNames(
                includePrivate=True, includeComputed=True)
                ]

        @staticmethod
        def get_prop(rdkit_obj: Chem.rdchem.Mol |
                     Chem.rdchem.Atom |
                     Chem.rdchem.Bond |
                     Chem.rdchem.Conformer,
                     prop: str) -> str:
            """
Get the value of a property from an RDKit object.

Shortcut for RDKit's `GetProp()` method.

Parameters
----------
rdkit_obj : rdkit.Chem.rdchem.Mol or Atom or Bond or Conformer
    The RDKit object to inspect.
prop : str
    The name of the property to retrieve.

Returns
-------
str
    The value of the specified property.
"""

            return rdkit_obj.GetProp(prop)

        @staticmethod
        def set_prop(rdkit_obj: Chem.rdchem.Mol |
                     Chem.rdchem.Atom |
                     Chem.rdchem.Bond |
                     Chem.rdchem.Conformer,
                     prop_name: str,
                     prop_val: str) -> None:
            """
Set a property on an RDKit object.

Shortcut for RDKit's `SetProp()` method.

Parameters
----------
rdkit_obj : rdkit.Chem.rdchem.Mol or Atom or Bond or Conformer
    The RDKit object to modify.
prop_name : str
    The name of the property to set.
prop_val : str
    The value to assign to the property.

Returns
-------
None
"""

            return rdkit_obj.SetProp(prop_name, prop_val)

        @staticmethod
        def get_owning_mol(rdkit_obj: Chem.rdchem.Atom |
                           Chem.rdchem.Bond |
                           Chem.rdchem.Conformer) -> Chem.rdchem.Mol:
            """
Get the molecule that owns the RDKit object.

Shortcut for RDKit's `GetOwningMol()` method.

Parameters
----------
rdkit_obj : rdkit.Chem.rdchem.Atom or Bond or Conformer
    The RDKit object whose parent molecule is to be retrieved.

Returns
-------
rdkit.Chem.rdchem.Mol
    The owning molecule.
"""

            return rdkit_obj.GetOwningMol()

    class Mol:

        @staticmethod
        def get_atoms(mol: Chem.rdchem.Mol) -> list:
            """
            Get the atoms of a molecule.

            Retrieves all atoms from the given RDKit molecule object.

            Parameters
            ----------
            mol : rdkit.Chem.rdchem.Mol
                The RDKit molecule object.

            Returns
            -------
            list of rdkit.Chem.rdchem.Atom
                A list of RDKit atom objects.
            """

            return [a for a in mol.GetAtoms()]

        @staticmethod
        def get_atoms_from_idx(
            mol: Chem.rdchem.Mol,
            idx: int |
            Iterable[int],
             ) -> Chem.rdchem.Atom | list[Chem.rdchem.Atom]:
            """
Retrieve atom(s) from a molecule based on index or indices.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.
idx : int or Iterable[int]
    The index or indices of the atom(s) to retrieve.

Returns
-------
rdkit.Chem.rdchem.Atom or list of rdkit.Chem.rdchem.Atom
    A single atom object if `idx` is an int, otherwise a list of atom objects.
"""

            if isinstance(idx, int):
                return mol.GetAtomWithIdx(idx)
            else:
                return [mol.GetAtomWithIdx(i) for i in idx]

        @staticmethod
        def get_bonds(mol: Chem.rdchem.Mol) -> list:
            """
Get the bonds of a molecule.

Retrieves all bonds from the given RDKit molecule object.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.

Returns
-------
list of rdkit.Chem.rdchem.Bond
    A list of RDKit bond objects.
"""

            return [b for b in mol.GetBonds()]

        @staticmethod
        def get_bonds_from_idx(mol: Chem.rdchem.Mol,
                               idx: int | Iterable[int]) -> list:
            """
Retrieve bond(s) from a molecule based on index or indices.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.
idx : int or Iterable[int]
    The index or indices of the bond(s) to retrieve.

Returns
-------
list of rdkit.Chem.rdchem.Bond
    A list of RDKit bond objects.
"""

            if isinstance(idx, int):
                return [mol.GetBondWithIdx(idx)]
            else:
                return [mol.GetBondWithIdx(i) for i in idx]

        @staticmethod
        def get_bond_between_atoms(mol: Chem.rdchem.Mol,
                                   idx1: int, idx2: int) -> Chem.rdchem.Bond:
            """
Retrieve the bond between two atoms in a molecule.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.
idx1 : int
    The index of the first atom.
idx2 : int
    The index of the second atom.

Returns
-------
rdkit.Chem.rdchem.Bond
    The bond object between the specified atoms.
"""

            return mol.GetBondBetweenAtoms(idx1, idx2)

        @staticmethod
        def get_coordinates(
            conf_or_mol: Chem.rdchem.Conformer | Chem.rdchem.Mol,
            is_3d: bool = False,
            canonOrient: bool = True,
            bondLength: float = -1.0
        ) -> np.ndarray:
            """
Get the coordinates of a molecule or conformer.

Parameters
----------
conf_or_mol : rdkit.Chem.rdchem.Conformer or rdkit.Chem.rdchem.Mol
    The conformer or molecule object.
is_3d : bool, optional
    Whether to retrieve 3D coordinates. Default is False.
canonOrient : bool, optional
    Whether to use canonical orientation for 2D coordinates. Default is True.
bondLength : float, optional
    Bond length for 2D coordinate generation. Default is -1.0.

Returns
-------
numpy.ndarray
    An array of atomic coordinates.
"""

            if isinstance(conf_or_mol, Chem.rdchem.Mol):
                mol = conf_or_mol
                if not is_3d:     # make molecule able to generate a conformer
                    mol.Compute2DCoords(canonOrient=canonOrient,
                                        bondLength=bondLength)
                    conf = mol.GetConformer()
                else:
                    conf = mol.GetConformer()
            elif isinstance(conf_or_mol, Chem.rdchem.Conformer):
                conf = conf_or_mol
                mol = conf.GetOwningMol()
            return conf.GetPositions()

        @staticmethod
        def get_conformer(mol: Chem.rdchem.Mol,
                          id: int = -1) -> Chem.rdchem.Conformer:
            """
Get the conformer associated with a molecule.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.
id : int, optional
    The ID of the conformer to retrieve. Default is -1.

Returns
-------
rdkit.Chem.rdchem.Conformer
    The conformer object.
"""

            return mol.GetConformer(id=id)

        @staticmethod
        def get_conformers(mol: Chem.rdchem.Mol) -> list:
            """
Get all conformers associated with a molecule.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.

Returns
-------
list of rdkit.Chem.rdchem.Conformer
    A list of conformer objects.
"""

            return [c for c in mol.GetConformers()]

        @staticmethod
        def get_conf_ids(mol: Chem.rdchem.Mol) -> list:
            """
Get all conformer IDs associated with a molecule.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.

Returns
-------
list of int
    A list of conformer IDs.
"""

            return [i for i, c in enumerate(mol.GetConformers())]

        @staticmethod
        def get_distance_matrix(mol: Chem.rdchem.Mol,
                                is_3d: bool = True) -> np.ndarray:
            """
Get the distance matrix of a molecule.

Calculates the pairwise distance matrix using atomic coordinates.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.
is_3d : bool, optional
    Whether to use 3D coordinates. Default is True.

Returns
-------
numpy.ndarray
    The distance matrix.
"""

            from mlchem.\
                chem.calculator.tools import pairwise_euclidean_distance as ped
            coords = PropManager.Mol.get_coordinates(conf_or_mol=mol,
                                                     is_3d=is_3d)
            return ped(coords)

        @staticmethod
        def get_gasteiger_charges(mol: Chem.rdchem.Mol,
                                  atom_ids: int | Iterable[int] = [],
                                  nIter: int = 12) -> list:
            """
Compute Gasteiger charges for specified atoms in a molecule.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.
atom_ids : int or Iterable[int], optional
    Atom indices to compute charges for. Default is all atoms.
nIter : int, optional
    Number of iterations for charge computation. Default is 12.

Returns
-------
list of float
    Gasteiger charges for the specified atoms.
"""

            if atom_ids is None or (isinstance(atom_ids, Iterable) and
                                    len(atom_ids) == 0):
                atom_ids = range(len(mol.GetAtoms()))
            mol.ComputeGasteigerCharges(nIter=nIter)
            return [float(a.GetProp('_GasteigerCharge')) for a in
                    PropManager.Mol.get_atoms_from_idx(mol, atom_ids)]

        @staticmethod
        def get_stereogroups(mol: Chem.rdchem.Mol) -> list:
            """
Get the stereochemistry groups of a molecule.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.

Returns
-------
list of rdkit.Chem.rdchem.StereoGroup
    A list of stereochemistry groups.
"""

            return mol.GetStereoGroups()

        @staticmethod
        def remove_conformer(mol: Chem.rdchem.Mol, id: int) -> None:
            """
Remove a conformer from a molecule.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.
id : int
    The ID of the conformer to remove.

Returns
-------
None
"""

            return mol.RemoveConformer(id)

        @staticmethod
        def remove_all_conformers(mol: Chem.rdchem.Mol) -> None:
            """
Remove all conformers from a molecule.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.

Returns
-------
None
"""

            return mol.RemoveAllConformers()

    class Atom:
        @staticmethod
        def clear_atomprops(atom: Chem.rdchem.Atom) -> None:
            """
Clear all properties from an atom.

Shortcut for the analogous RDKit method `ClearProp()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object from which to clear all properties.

Returns
-------
None
"""

            for key in atom.GetPropsAsDict():
                atom.ClearProp(key)

        @staticmethod
        def get_atomic_num(atom: Chem.rdchem.Atom) -> int:
            """
Get the atomic number of an atom.

Shortcut for the analogous RDKit method `GetAtomicNum()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
int
    The atomic number of the atom.
"""

            return atom.GetAtomicNum()

        @staticmethod
        def get_bonds(atom: Chem.rdchem.Atom) -> tuple:
            """
Get the bonds of an atom.

Shortcut for the analogous RDKit method `GetBonds()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
tuple
    A tuple of RDKit bond objects associated with the atom.
"""

            return atom.GetBonds()

        @staticmethod
        def get_degree(atom: Chem.rdchem.Atom) -> int:
            """
Get the degree of an atom.

Shortcut for the analogous RDKit method `GetDegree()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
int
    The degree of the atom.
"""

            return atom.GetDegree()

        @staticmethod
        def get_total_degree(atom: Chem.rdchem.Atom) -> int:
            """
Get the total degree of an atom, including hydrogens.

Shortcut for the analogous RDKit method `GetTotalDegree()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
int
    The total degree of the atom.
"""

            return atom.GetTotalDegree()

        @staticmethod
        def get_explicit_valence(atom: Chem.rdchem.Atom) -> int:
            """
Get the explicit valence of an atom.

Shortcut for the analogous RDKit method `GetExplicitValence()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
int
    The explicit valence of the atom.
"""

            return atom.GetExplicitValence()

        @staticmethod
        def get_implicit_valence(atom: Chem.rdchem.Atom) -> int:
            """
Get the implicit valence of an atom.

Shortcut for the analogous RDKit method `GetImplicitValence()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
int
    The implicit valence of the atom.
"""

            return atom.GetImplicitValence()

        @staticmethod
        def get_total_valence(atom: Chem.rdchem.Atom) -> int:
            """
Get the total valence of an atom.

Shortcut for the analogous RDKit method `GetTotalValence()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
int
    The total valence of the atom.
"""

            return atom.GetTotalValence()

        @staticmethod
        def has_valence_violation(atom: Chem.rdchem.Atom) -> bool:
            """
Check if an atom has a valence violation.

Shortcut for the analogous RDKit method `HasValenceViolation()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
bool
    True if the atom has a valence violation, False otherwise.
"""

            return atom.HasValenceViolation()

        @staticmethod
        def get_formal_charge(atom: Chem.rdchem.Atom) -> int:
            """
Get the formal charge of an atom.

Shortcut for the analogous RDKit method `GetFormalCharge()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
int
    The formal charge of the atom.
"""

            return atom.GetFormalCharge()

        @staticmethod
        def get_hybridisation(
            atom: Chem.rdchem.Atom
                ) -> Chem.rdchem.HybridizationType:
            """
Get the hybridisation of an atom.

Shortcut for the analogous RDKit method `GetHybridization()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
rdkit.Chem.rdchem.HybridizationType
    The hybridisation of the atom.
"""

            return atom.GetHybridization()

        @staticmethod
        def get_idx(atom: Chem.rdchem.Atom) -> int:
            """
Get the index of an atom.

Shortcut for the analogous RDKit method `GetIdx()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
int
    The index of the atom.
"""

            return atom.GetIdx()

        @staticmethod
        def get_neighbours(atom: Chem.rdchem.Atom,
                           order: int = 1) -> list:
            """
Get the neighbours of an atom up to a specified order.

This function recursively finds the neighbours of a given atom up to the 
specified order. It ensures that atoms are not revisited by checking 
their map number.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.
order : int, optional
    The order of neighbours to find. Default is 1.

Returns
-------
list
    A list of RDKit atom objects representing the neighbours.
"""

            assert order > 0, "'order' argument must be at least 1."

            def recursive_neighbours(current_atoms: set,
                                     visited_mapnums: set,
                                     current_order: int) -> set:
                """Recursively find neighbours of the given atoms."""
                if current_order == 0:
                    return current_atoms
                next_neighbours = {
                    neighbour for atom in current_atoms
                    for neighbour in atom.GetNeighbors() if
                    neighbour.GetAtomMapNum() not in visited_mapnums
                    }
                visited_mapnums.update(
                    neighbour.GetAtomMapNum()
                    for neighbour in next_neighbours
                    )
                return current_atoms | recursive_neighbours(
                    next_neighbours, visited_mapnums, current_order - 1
                    )

            initial_neighbours = set(atom.GetNeighbors())
            visited_mapnums = {atom.GetAtomMapNum()} | \
                {neighbour.GetAtomMapNum() for neighbour in initial_neighbours}
            return list(
                initial_neighbours | recursive_neighbours(
                    initial_neighbours, visited_mapnums, order - 1)
                    )

        @staticmethod
        def is_aromatic(atom: Chem.rdchem.Atom) -> bool:
            """
Check if an atom is aromatic.

Shortcut for the analogous RDKit method `GetIsAromatic()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
bool
    True if the atom is aromatic, False otherwise.
"""

            return atom.GetIsAromatic()

        @staticmethod
        def is_in_ring(atom: Chem.rdchem.Atom) -> bool:
            """
Check if an atom is in a ring.

Shortcut for the analogous RDKit method `IsInRing()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
bool
    True if the atom is in a ring, False otherwise.
"""

            return atom.IsInRing()

        @staticmethod
        def is_in_ring_size(atom: Chem.rdchem.Atom, size: int) -> bool:
            """
Check if an atom is in a ring of a specific size.

Shortcut for the analogous RDKit method `IsInRingSize()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.
size : int
    The size of the ring to check.

Returns
-------
bool
    True if the atom is in a ring of the specified size, False otherwise.
"""

            return atom.IsInRingSize(size)

        @staticmethod
        def get_mass(atom: Chem.rdchem.Atom) -> float:
            """
Get the mass of an atom.

Shortcut for the analogous RDKit method `GetMass()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
float
    The mass of the atom.
"""

            return atom.GetMass()

        @staticmethod
        def get_num_explicit_h(atom: Chem.rdchem.Atom) -> int:
            """
Get the number of explicit hydrogen atoms attached to an atom.

Shortcut for the analogous RDKit method `GetNumExplicitHs()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
int
    The number of explicit hydrogen atoms.
"""

            return atom.GetNumExplicitHs()

        @staticmethod
        def get_num_implicit_h(atom: Chem.rdchem.Atom) -> int:
            """
Get the number of implicit hydrogen atoms attached to an atom.

Shortcut for the analogous RDKit method `GetNumImplicitHs()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
int
    The number of implicit hydrogen atoms.
"""

            return atom.GetNumImplicitHs()

        @staticmethod
        def get_tot_h(atom: Chem.rdchem.Atom) -> int:
            """
Get the total number of hydrogen atoms attached to an atom.

Shortcut for the analogous RDKit method `GetTotalNumHs()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
int
    The total number of hydrogen atoms.
"""

            return atom.GetTotalNumHs()

        @staticmethod
        def get_num_radical_electrons(atom: Chem.rdchem.Atom) -> int:
            """
Get the number of radical electrons on an atom.

Shortcut for the analogous RDKit method `GetNumRadicalElectrons()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.

Returns
-------
int
    The number of radical electrons.
"""

            return atom.GetNumRadicalElectrons()

        @staticmethod
        def set_atom_map_num(atom: Chem.rdchem.Atom, num: int) -> None:
            """
Set the atom map number for an atom.

Shortcut for the analogous RDKit method `SetAtomMapNum()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.
num : int
    The atom map number to set.

Returns
-------
None
"""

            return atom.SetAtomMapNum(num)

        @staticmethod
        def set_formal_charge(atom: Chem.rdchem.Atom,
                              charge: int) -> None:
            """
Set the formal charge of an atom.

Shortcut for the analogous RDKit method `SetFormalCharge()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.
charge : int
    The formal charge to set.

Returns
-------
None
"""

            return atom.SetFormalCharge(charge)

        @staticmethod
        def set_is_aromatic(atom: Chem.rdchem.Atom,
                            decision: bool) -> None:
            """
Set the aromaticity of an atom.

Shortcut for the analogous RDKit method `SetIsAromatic()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.
decision : bool
    Whether the atom should be marked as aromatic.

Returns
-------
None
"""

            return atom.SetIsAromatic(decision)

        @staticmethod
        def set_num_explicit_h(atom: Chem.rdchem.Atom, num: int) -> None:
            """
Set the number of explicit hydrogen atoms attached to an atom.

Shortcut for the analogous RDKit method `SetNumExplicitHs()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.
num : int
    The number of explicit hydrogen atoms to set.

Returns
-------
None
"""

            return atom.SetNumExplicitHs(num)

        @staticmethod
        def set_num_radical_electrons(atom: Chem.rdchem.Atom,
                                      num: int) -> None:
            """
Set the number of radical electrons on an atom.

Shortcut for the analogous RDKit method `SetNumRadicalElectrons()`.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The RDKit atom object.
num : int
    The number of radical electrons to set.

Returns
-------
None
"""

            return atom.SetNumRadicalElectrons(num)

    class Bond:
        @staticmethod
        def get_begin_atom(bond: Chem.rdchem.Bond) -> Chem.rdchem.Atom:

          """
Get the beginning atom of a bond.

Shortcut for the analogous RDKit method `GetBeginAtom()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.

Returns
-------
rdkit.Chem.rdchem.Atom
    The atom at the beginning of the bond.
"""

          return bond.GetBeginAtom()

        @staticmethod
        def get_begin_atom_idx(bond: Chem.rdchem.Bond) -> int:
            """
Get the index of the beginning atom of a bond.

Shortcut for the analogous RDKit method `GetBeginAtomIdx()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.

Returns
-------
int
    The index of the beginning atom.
"""

            return bond.GetBeginAtomIdx()

        @staticmethod
        def get_bond_type(bond: Chem.rdchem.Bond) -> Chem.rdchem.BondType:
            """
Get the type of a bond.

Shortcut for the analogous RDKit method `GetBondType()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.

Returns
-------
rdkit.Chem.rdchem.BondType
    The type of the bond.
"""

            return bond.GetBondType()

        @staticmethod
        def get_end_atom(bond: Chem.rdchem.Bond) -> Chem.rdchem.Atom:
            """
Get the ending atom of a bond.

Shortcut for the analogous RDKit method `GetEndAtom()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.

Returns
-------
rdkit.Chem.rdchem.Atom
    The atom at the end of the bond.
"""

            return bond.GetEndAtom()

        @staticmethod
        def get_end_atom_idx(bond: Chem.rdchem.Bond) -> int:
            """
Get the index of the ending atom of a bond.

Shortcut for the analogous RDKit method `GetEndAtomIdx()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.

Returns
-------
int
    The index of the ending atom.
"""

            return bond.GetEndAtomIdx()

        @staticmethod
        def get_idx(bond: Chem.rdchem.Bond) -> int:
            """
Get the index of a bond.

Shortcut for the analogous RDKit method `GetIdx()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.

Returns
-------
int
    The index of the bond.
"""

            return bond.GetIdx()

        @staticmethod
        def get_other_atom(bond: Chem.rdchem.Bond,
                           atom: Chem.rdchem.Atom) -> Chem.rdchem.Atom:
            """
Given one atom of the bond, get the other atom.

Shortcut for the analogous RDKit method `GetOtherAtom()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.
atom : rdkit.Chem.rdchem.Atom
    One of the atoms in the bond.

Returns
-------
rdkit.Chem.rdchem.Atom
    The other atom in the bond.
"""

            return bond.GetOtherAtom(atom)

        @staticmethod
        def get_other_atom_idx(bond: Chem.rdchem.Bond,
                               idx: int) -> int:
            """
Given the index of one atom in the bond, get the index of the other atom.

Shortcut for the analogous RDKit method `GetOtherAtomIdx()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.
idx : int
    The index of one atom in the bond.

Returns
-------
int
    The index of the other atom in the bond.
"""

            return bond.GetOtherAtomIdx(idx)

        @staticmethod
        def get_valence_contribution(bond: Chem.rdchem.Bond,
                                     atom: Chem.rdchem.Atom) -> float:
            """
Get the valence contribution of a bond to an atom.

Shortcut for the analogous RDKit method `GetValenceContrib()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.
atom : rdkit.Chem.rdchem.Atom
    The atom for which to compute the valence contribution.

Returns
-------
float
    The valence contribution of the bond to the atom.
"""

            return bond.GetValenceContrib(atom)

        @staticmethod
        def is_aromatic(bond: Chem.rdchem.Bond) -> bool:
            """
Check if a bond is aromatic.

Shortcut for the analogous RDKit method `GetIsAromatic()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.

Returns
-------
bool
    True if the bond is aromatic, False otherwise.
"""

            return bond.GetIsAromatic()

        @staticmethod
        def is_conjugated(bond: Chem.rdchem.Bond) -> bool:
          """
Check if a bond is conjugated.

Shortcut for the analogous RDKit method `GetIsConjugated()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.

Returns
-------
bool
    True if the bond is conjugated, False otherwise.
"""

          return bond.GetIsConjugated()

        @staticmethod
        def is_in_ring(bond: Chem.rdchem.Bond) -> bool:
            """
Check if a bond is in a ring.

Shortcut for the analogous RDKit method `IsInRing()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.

Returns
-------
bool
    True if the bond is in a ring, False otherwise.
"""

            return bond.IsInRing()

        @staticmethod
        def is_in_ring_size(bond: Chem.rdchem.Bond, size: int) -> bool:
            """
Check if a bond is in a ring of a specific size.

Shortcut for the analogous RDKit method `IsInRingSize()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.
size : int
    The size of the ring to check.

Returns
-------
bool
    True if the bond is in a ring of the specified size, False otherwise.
"""

            return bond.IsInRingSize(size)

        @staticmethod
        def set_is_aromatic(bond: Chem.rdchem.Bond, decision: bool) -> None:
          """
Check if a bond is in a ring of a specific size.

Shortcut for the analogous RDKit method `IsInRingSize()`.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    The RDKit bond object.
size : int
    The size of the ring to check.

Returns
-------
bool
    True if the bond is in a ring of the specified size, False otherwise.
"""

          return bond.SetIsAromatic(decision)

    class Conformation:

        @staticmethod
        def straighten_mol_2d(mol: Chem.rdchem.Mol) -> None:
            """
Straighten the 2D depiction of a molecule.

This method computes 2D coordinates and straightens the depiction
of the molecule using RDKit's depiction tools.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.

Returns
-------
None
"""

            from rdkit.Chem import rdDepictor

            rdDepictor.Compute2DCoords(mol)
            rdDepictor.StraightenDepiction(mol)

        @staticmethod
        def add_conformer(mol: Chem.rdchem.Mol,
                          conformer: Chem.rdchem.Conformer,
                          assignId: bool = False) -> int:
            """
Add a conformer to a molecule and return its ID.

Shortcut for the analogous RDKit method `AddConformer()`.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object to which the conformer will be added.
conformer : rdkit.Chem.rdchem.Conformer
    The conformer to add to the molecule.
assignId : bool, optional
    Whether to assign a new ID to the conformer. Default is False.

Returns
-------
int
    The ID of the added conformer.
"""

            return mol.AddConformer(conf=conformer, assignId=assignId)

        @staticmethod
        def generate_conformers(
            mol: Chem.rdchem.Mol,
            n_conf: int,
            rms_threshold: int = 0,
            embedding_params: Literal['ETDG',
                                      'ETKDG',
                                      'ETKDGv2',
                                      'ETKDGv3'] = 'ETKDGv3',
            show: bool = False,
            size: tuple = (300, 300),
            force_field: Literal['MMFF94', 'UFF'] = 'MMFF94',
            optimise: bool = True,
            max_iter: int = 500,
            random_seed: int = 0xf00d
        ) -> tuple:
            """
Generate conformers for a molecule.

This method embeds multiple conformers using specified embedding
parameters, optionally optimises them using a force field, and
returns the conformers along with their energies and optimisation
results.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.
n_conf : int
    The number of conformers to generate.
rms_threshold : int, optional
    RMS threshold for pruning conformers. Default is 0.
embedding_params : {'ETDG', 'ETKDG', 'ETKDGv2', 'ETKDGv3'}, optional
    The embedding parameters to use. Default is 'ETKDGv3'.
show : bool, optional
    Whether to display the conformers. Default is False.
size : tuple, optional
    The size of the display window. Default is (300, 300).
force_field : {'MMFF94', 'UFF'}, optional
    The force field to use for energy calculation. Default is 'MMFF94'.
optimise : bool, optional
    Whether to optimise the conformers. Default is True.
max_iter : int, optional
    Maximum number of iterations for optimisation. Default is 500.
random_seed : int, optional
    Random seed for conformer generation. Default is 0xf00d.

Returns
-------
tuple
    A tuple containing:
    - A list of RDKit conformer objects.
    - A list of conformer energies.
    - A list of optimisation results (if `optimise` is True).
"""

            from rdkit.Chem import rdDistGeom
            from rdkit.Chem import AllChem
            from rdkit.Chem.Draw import IPythonConsole as ipc

            if embedding_params == 'ETDG':
                params = rdDistGeom.ETDG()
            elif embedding_params == 'ETKDG':
                params = rdDistGeom.ETKDG()
            elif embedding_params == 'ETKDGv2':
                params = rdDistGeom.ETKDGv2()
            elif embedding_params == 'ETKDGv3':
                params = rdDistGeom.ETKDGv3()

            params.randomSeed = random_seed
            params.pruneRmsThresh = rms_threshold
            conf_ids = rdDistGeom.EmbedMultipleConfs(mol, n_conf, params)

            energies = []
            for id in conf_ids:
                energy = PropManager.Conformation.\
                    calculate_conformer_energy_from_mol(mol, id, force_field)
                energies.append(energy)
            if optimise:
                if force_field == 'MMFF94':
                    results = AllChem.MMFFOptimizeMoleculeConfs(
                        mol, maxIters=max_iter)
                else:
                    results = AllChem.UFFOptimizeMoleculeConfs(
                        mol, maxIters=max_iter)
            if show:
                from rdkit.Chem import Draw
                ipc.ipython_3d = True
                for id in conf_ids:
                    ipc.drawMol3D(mol, confId=id, size=size)
            else:
                if optimise:
                    return [conf for conf in mol.GetConformers()], \
                        energies, results
                else:
                    return [conf for conf in mol.GetConformers()], energies

        @staticmethod
        def display_conformers(conf: Chem.rdchem.Conformer |
                               Iterable[Chem.rdchem.Conformer],
                               size: tuple = (300, 300)) -> None:
            """
Display conformers in 3D.

Shortcut for the analogous RDKit method `drawMol3D()`.

Parameters
----------
conf : rdkit.Chem.rdchem.Conformer or Iterable[rdkit.Chem.rdchem.Conformer]
    A single RDKit conformer or an iterable of conformers to display.
size : tuple, optional
    The size of the display window. Default is (300, 300).

Returns
-------
None
"""

            from rdkit.Chem.Draw import IPythonConsole as ipc

            ipc.ipython_3d = True
            if isinstance(conf, Iterable):
                for id, c in enumerate(conf):
                    ipc.drawMol3D(c.GetOwningMol(),
                                                  confId=id, size=size)
            else:
                ipc.drawMol3D(conf.GetOwningMol(),
                                              confId=conf.GetId(), size=size)

        @staticmethod
        def display_3dmols_overlapped(mols: list[Chem.rdchem.Mol],
                                      py3dmolviewer=None,
                                      size: tuple | list | int = (400, 400),
                                      confIds: list[int] | None = None,
                                      removeHs: bool = False,
                                      colours: list[str] = None) -> None:
            """
            Display multiple 3D molecules overlapped in a py3Dmol viewer.

            This function displays a list of RDKit molecules in a 3D viewer,
            with options to customise size, conformer IDs, hydrogen removal,
            and colour schemes.

            Parameters
            ----------
            mols : list of rdkit.Chem.rdchem.Mol
                List of RDKit molecule objects to display.
            py3dmolviewer : py3Dmol.view, optional
                An existing py3Dmol viewer instance. If None, a new 
                viewer is created.
            size : int or tuple or list, optional
                Size of the viewer. Can be a single integer or a 
                tuple/list of two integers. Default is (400, 400).
            confIds : list of int, optional
                List of conformer IDs to display. If None, uses default 
                conformer (-1). Default is None.
            removeHs : bool, optional
                Whether to remove hydrogen atoms before displaying. 
                Default is False.
            colours : list of str, optional
                List of colour schemes for the molecules. Default is a 
                predefined list.

            Returns
            -------
            None
            """

            import py3Dmol
            from rdkit.Chem import AllChem, Draw
            from rdkit.Chem.Draw import IPythonConsole as ipc

            assert ((isinstance(size, int) is False) or (len(size) < 3)), \
                "'size' argument must be either an integer or an interable"
            " of max length == 2"
            if colours is None:
                colours = [
                    'cyanCarbon', ' blueCarbon', 'redCarbon', 'greenCarbon',
                    'magentaCarbon', 'yellowCarbon', 'orangeCarbon',
                    'purpleCarbon', 'whiteCarbon', 'blackCarbon',
                    'salmonCarbon', 'limeCarbon', 'hotpinkCarbon',
                    'oliveCarbon', 'tealCarbon', 'wheatCarbon',
                    'lightpinkCarbon', 'aquamarineCarbon', 'limegreenCarbon',
                    'skyblueCarbon', 'darksalmonCarbon', 'brownCarbon',
                    'saddlebrownCarbon', 'peruCarbon',
                ]
            if confIds is None:
                confIds = [-1] * len(mols)
            if py3dmolviewer is None:
                if isinstance(size, int):
                    py3dmolviewer = py3Dmol.view(width=size, height=size)
                elif isinstance(size, Iterable) and len(size) == 1:
                    py3dmolviewer = py3Dmol.view(width=size[0], height=size[0])
                else:
                    py3dmolviewer = py3Dmol.view(width=size[0], height=size[1])
            py3dmolviewer.removeAllModels()
            for i, (m, confId) in enumerate(zip(mols, confIds)):
                if removeHs:
                    m = AllChem.RemoveHs(m)
                ipc.addMolToView(m,
                                                 py3dmolviewer,
                                                 confId=confId)
            for i, m in enumerate(mols):
                py3dmolviewer.setStyle({'model': i},
                                       {'stick': {'colorscheme':
                                                  colours[i % len(colours)]}})
            py3dmolviewer.zoomTo()
            return py3dmolviewer.show()

        @staticmethod
        def canonicalise_conformer(conf: Chem.rdchem.Conformer,
                                   ignoreHs: bool = False) -> None:
            """
            Canonicalise a conformer.

            Shortcut for the analogous RDKit method `CanonicalizeConformer()`.

            Parameters
            ----------
            conf : rdkit.Chem.rdchem.Conformer
                The conformer to canonicalise.
            ignoreHs : bool, optional
                Whether to ignore hydrogen atoms. Default is False.

            Returns
            -------
            None
            """

            return Chem.rdMolTransforms.CanonicalizeConformer(
                conf,
                ignoreHs=ignoreHs
                )

        @staticmethod
        def canonicalise_mol_conformers(mol: Chem.rdchem.Mol,
                                        ignoreHs: bool = False) -> None:
            """
Canonicalise all conformers of a molecule.

Shortcut for the analogous RDKit method `CanonicalizeMol()`.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The molecule whose conformers are to be canonicalised.
ignoreHs : bool, optional
    Whether to ignore hydrogen atoms. Default is False.

Returns
-------
None
"""

            return Chem.rdMolTransforms.CanonicalizeMol(mol,
                                                        ignoreHs=ignoreHs)

        @staticmethod
        def calculate_conformer_energy_from_mol(
            mol: Chem.rdchem.Mol,
            conf_id: int = -1,
            forcefield: Literal['UFF', 'MMFF94'] = 'MMFF94'
        ) -> float:
            """
Calculate the energy of a specific conformer.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The RDKit molecule object.
conf_id : int, optional
    The ID of the conformer. Default is -1.
forcefield : {'UFF', 'MMFF94'}, optional
    The force field to use. Default is 'MMFF94'.

Returns
-------
float
    The energy of the conformer in kcal/mol.
"""

            from rdkit.Chem import AllChem

            if forcefield == 'UFF':
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            elif forcefield == 'MMFF94':
                mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
                ff = AllChem.MMFFGetMoleculeForceField(mol,
                                                       mmff_props,
                                                       confId=conf_id)
            else:
                raise ValueError(
                    "Unsupported force field. Use 'UFF' or 'MMFF94'."
                    )

            energy = ff.CalcEnergy()
            return energy

        @staticmethod
        def optimise_conformers(
            mol: Chem.rdchem.Mol,
            force_field: Literal['UFF', 'MMFF94'] = 'MMFF94',
            max_iter: int = 500
        ) -> list:
            """
Optimise all conformers of a molecule.

Shortcut for the analogous RDKit methods `MMFFOptimizeMoleculeConfs()` 
and `UFFOptimizeMoleculeConfs()`.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The molecule whose conformers are to be optimised.
force_field : {'UFF', 'MMFF94'}, optional
    The force field to use. Default is 'MMFF94'.
max_iter : int, optional
    Maximum number of iterations. Default is 500.

Returns
-------
list of tuple
    Optimisation results for each conformer.
"""

            from rdkit.Chem import AllChem

            if force_field == 'MMFF94':
                results = AllChem.MMFFOptimizeMoleculeConfs(mol,
                                                            maxIters=max_iter)
            else:
                results = AllChem.UFFOptimizeMoleculeConfs(mol,
                                                           maxIters=max_iter)
            return results

        @staticmethod
        def optimise_molecule(
            mol: Chem.rdchem.Mol,
            conf_id: int = -1,
            force_field: Literal['UFF', 'MMFF94'] = 'MMFF94',
            max_iter: int = 500
        ) -> int:
            """
Optimise a specific conformer of a molecule.

Shortcut for the analogous RDKit methods `MMFFOptimizeMolecule()` and 
`UFFOptimizeMolecule()`.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The molecule whose conformer is to be optimised.
conf_id : int, optional
    The ID of the conformer. Default is -1. If value is -1, whole 
    molecule will be optimised.
force_field : {'UFF', 'MMFF94'}, optional
    The force field to use. Default is 'MMFF94'.
max_iter : int, optional
    Maximum number of iterations. Default is 500.

Returns
-------
int
    0 if converged, -1 if force field setup failed, 1 if more iterations 
    are needed.
"""

            from rdkit.Chem import AllChem

            if force_field == 'MMFF94':
                results = AllChem.MMFFOptimizeMolecule(mol,
                                                       maxIters=max_iter,
                                                       confId=conf_id)
            else:
                results = AllChem.UFFOptimizeMolecule(mol,
                                                      maxIters=max_iter,
                                                      confId=conf_id)
            return results

        @staticmethod
        def get_shape_descriptors(
            conf_or_mol: Chem.rdchem.Conformer | Chem.rdchem.Mol,
            include_masses: bool = True,
            is_3d: bool = True
        ) -> dict:
            """
Calculate shape descriptors for a conformer or molecule.

Parameters
----------
conf_or_mol : rdkit.Chem.rdchem.Conformer or rdkit.Chem.rdchem.Mol
    The conformer or molecule to analyse.
include_masses : bool, optional
    Whether to include atomic masses. Default is True.
is_3d : bool, optional
    Whether to use 3D coordinates. Default is True.

Returns
-------
dict
    A dictionary of shape descriptors.
"""

            from mlchem.chem.calculator.tools import (
                calc_gyration_tensor,
                calc_shape_descriptors_from_gyration_tensor
                )

            if isinstance(conf_or_mol, Chem.rdchem.Mol):
                conf = conf_or_mol.GetConformer()
                mol = conf_or_mol
            elif isinstance(conf_or_mol, Chem.rdchem.Conformer):
                conf = conf_or_mol
                mol = conf.GetOwningMol()
            if include_masses:
                masses = [a.GetMass() for a in mol.GetAtoms()]
            else:
                masses = None

            coordinates = PropManager.Mol.get_coordinates(conf, is_3d=is_3d)
            tensor = calc_gyration_tensor(coordinates, masses)

            return calc_shape_descriptors_from_gyration_tensor(tensor)


class MolCleaner:
    """
A class to clean and process SMILES strings.

The `MolCleaner` class provides methods to clean and process a list of SMILES strings.
The cleaning process includes steps such as initialising SMILES, removing carbon ions,
inorganics, organometallics, and mixtures, desalting, neutralising, and performing a
final quality check.

Parameters
----------
input_smiles_list : list of str
    A list of SMILES strings representing the molecules to be cleaned.
id_list : list of int, optional
    A list of IDs corresponding to the SMILES strings. If not provided, IDs will be generated automatically.

Attributes
----------
input_smiles_list : list of str
    The original list of SMILES strings.
input_id_list : list of int
    The original list of IDs.
df_input : pandas.DataFrame
    A DataFrame containing the input IDs and SMILES strings.
n_steps : int
    The number of cleaning steps performed.
smiles : list of str
    The current list of SMILES strings after cleaning steps.
ids : list of int
    The current list of IDs after cleaning steps.
IsIsomeric : bool
    A flag indicating whether the SMILES strings are isomeric.
IsCanonical : bool
    A flag indicating whether the SMILES strings are canonical.
IsKekulised : bool
    A flag indicating whether the SMILES strings are kekulised.
df_accepted : pandas.DataFrame
    A DataFrame to store accepted SMILES strings.
df_rejected : pandas.DataFrame
    A DataFrame to store rejected SMILES strings and the reason for rejection.

Examples
--------
>>> cleaner = MolCleaner([smiles_1, smiles_2], [id_1, id_2])
>>> cleaner.initialise_smiles()
>>> cleaner.remove_carbon_ions()
>>> cleaner.remove_inorganics()
>>> cleaner.remove_organometallics()
>>> cleaner.remove_mixtures()
>>> cleaner.desalt_smiles(method='largest')
>>> cleaner.neutralise_smiles()
>>> cleaner.quality_checker()

Alternatively:

>>> cleaner.full_clean()
"""


    def __init__(self,
                 input_smiles_list: list[str],
                 id_list: list[int] = None) -> None:
        """
Initialise the MolCleaner instance.

This constructor sets up the initial state of the MolCleaner object,
including the input SMILES strings, optional IDs, and internal tracking
variables for cleaning steps and SMILES processing.

Parameters
----------
input_smiles_list : list of str
    A list of SMILES strings representing the molecules to be cleaned.
id_list : list of int, optional
    A list of IDs corresponding to the SMILES strings. If not provided,
    IDs will be automatically generated as a range from 0 to the number
    of SMILES strings.

Returns
-------
None
"""



        self.n_steps = 0

        self.input_smiles_list = input_smiles_list
        self.smiles = input_smiles_list
        if id_list is None:
            id_list = np.arange(0, len(input_smiles_list))
        self.input_id_list = id_list
        self.ids = id_list

        self.IsIsomeric = False
        self.IsCanonical = False
        self.IsKekulised = False

        self.df_input = pd.DataFrame()
        self.df_input['id'] = self.input_id_list
        self.df_input['SMILES'] = self.smiles

        self.df_rejected = pd.DataFrame({'id': [],
                                         'SMILES': [],
                                         'rejection': []})

    def initialise_smiles(self,
                          isomeric: bool = False,
                          canonical: bool = True,
                          kekulise: bool = True) -> None:
                
                """
                Initialise the SMILES strings with specified options.

                This method processes the input SMILES strings according to the 
                specified options for isomeric, canonical, and kekulised representations. 
                It updates the SMILES strings and IDs, and tracks rejected SMILES 
                strings with reasons.

                Parameters
                ----------
                isomeric : bool, optional
                    Whether to generate isomeric SMILES. Default is False.
                canonical : bool, optional
                    Whether to generate canonical SMILES. Default is True.
                kekulise : bool, optional
                    Whether to kekulise the SMILES. Default is True.

                Returns
                -------
                None
                """

                smiles_accepted = []
                ids_accepted = []
                smiles_rejected = []
                ids_rejected = []
                rejections = []

                self.n_steps += 1
                if isomeric:
                    if canonical:
                        if kekulise:
                            message = 'error in stereo handling, '
                            'canonicalisation or kekulisation'
                        else:
                            message = 'error in stereo handling or canonicalisation'
                    else:
                        if kekulise:
                            message = 'error in stereo handling or kekulisation'
                        else:
                            message = 'error in stereo handling'
                else:
                    if canonical:
                        if kekulise:
                            message = 'could not canonicalise or kekulise'
                        else:
                            message = 'could not canonicalise'
                    else:
                        if kekulise:
                            message = 'could not kekulise'
                        else:
                            raise TypeError('At least one argument must be True')

                for id, smi in zip(self.ids, self.smiles):
                    try:
                        smiles_accepted.append(
                            Chem.MolToSmiles(
                                Chem.MolFromSmiles(smi),
                                kekuleSmiles=kekulise,
                                isomericSmiles=isomeric,
                                canonical=canonical
                                )
                                )
                        ids_accepted.append(id)
                    except Exception:
                        smiles_rejected.append(smi)
                        ids_rejected.append(id)
                        rejections.append(f'{self.n_steps} - {message}')

                self.IsIsomeric = isomeric
                self.IsCanonical = canonical
                self.IsKekulised = kekulise

                self.smiles = smiles_accepted
                self.ids = ids_accepted
                self.df_accepted = pd.DataFrame(data={'id': self.ids,
                                                      'SMILES': self.smiles})

                self.df_rejected = pd.concat([self.df_rejected,
                                              pd.DataFrame(
                                                  data={'id': ids_rejected,
                                                        'SMILES': smiles_rejected,
                                                        'rejection': rejections}
                                                        )])

                self.IsIsomeric = isomeric
                self.IsCanonical = canonical
                self.IsKekulised = kekulise

                self.smiles = smiles_accepted
                self.ids = ids_accepted
                self.df_accepted = pd.DataFrame(data={'id': self.ids,
                                                      'SMILES': self.smiles})
                self.df_rejected = pd.concat([self.df_rejected,
                                              pd.DataFrame(
                                                  data={'id': ids_rejected,
                                                        'SMILES': smiles_rejected,
                                                        'rejection': rejections})])

    def desalt_smiles(self,
                      method: Literal['chembl',
                                      'rdkit',
                                      'largest'] = 'largest',
                      dehydrate: bool = True,
                      isomeric: Optional[bool] = None,
                      canonical: Optional[bool] = None,
                      kekulise: Optional[bool] = None,
                      verbose: bool = False) -> None:
        """
Desalt the SMILES strings using the specified method.

This method processes the input SMILES strings to remove salts using one of the
available methods: 'chembl', 'rdkit', or 'largest'. It updates the SMILES strings
and IDs, and tracks rejected SMILES strings with reasons.

Parameters
----------
method : {'chembl', 'rdkit', 'largest'}, optional
    The method to use for desalting. Default is 'largest'.
dehydrate : bool, optional
    Whether to remove water fragments before desalting. Default is True.
isomeric : bool, optional
    Whether to generate isomeric SMILES. Default is None.
canonical : bool, optional
    Whether to generate canonical SMILES. Default is None.
kekulise : bool, optional
    Whether to kekulise the SMILES. Default is None.
verbose : bool, optional
    Whether to print verbose output. Default is False.

Returns
-------
None
"""


        smiles_accepted = []
        ids_accepted = []
        smiles_rejected = []
        ids_rejected = []
        rejections = []

        self.n_steps += 1
        message = 'could not desalt'
        if isomeric is not None:
            self.IsIsomeric = isomeric
        if canonical is not None:
            self.IsCanonical = canonical
        if kekulise is not None:
            self.IsKekulised = kekulise

        if method == 'chembl':
            from chembl_structure_pipeline import standardizer
            for id, smi in zip(self.ids, self.smiles):
                if dehydrate:
                    smi = Chem.MolToSmiles(
                      remove_smarts_pattern(create_molecule(smi), '[OH2]'),
                      isomericSmiles=self.IsIsomeric,
                      kekuleSmiles=self.IsKekulised,
                      canonical=self.IsCanonical
                      )     # dehydrate
                try:
                    desalted_mol = standardizer.get_fragment_parent_mol(
                        create_molecule(smi),
                        neutralize=False,
                        verbose=verbose
                        )
                    smiles_accepted.append(
                        Chem.MolToSmiles(
                            desalted_mol,
                            isomericSmiles=self.IsIsomeric,
                            kekuleSmiles=self.IsKekulised,
                            canonical=self.IsCanonical
                            )
                            )
                    ids_accepted.append(id)
                except Exception:
                    smiles_rejected.append(smi)
                    ids_rejected.append(id)
                    rejections.append(f'{self.n_steps} - {message}')

        if method == 'rdkit':
            from rdkit.Chem.SaltRemover import SaltRemover
            remover = SaltRemover()
            for id, smi in zip(self.ids, self.smiles):
                if dehydrate:
                    smi = Chem.MolToSmiles(
                        remove_smarts_pattern(create_molecule(smi), '[OH2]'),
                        isomericSmiles=self.IsIsomeric,
                        kekuleSmiles=self.IsKekulised,
                        canonical=self.IsCanonical
                        )     # dehydrate
                try:
                    desalted_mol = remover.StripMol(create_molecule(smi))
                    smiles_accepted.append(
                        Chem.MolToSmiles(
                            desalted_mol,
                            isomericSmiles=self.IsIsomeric,
                            kekuleSmiles=self.IsKekulised,
                            canonical=self.IsCanonical
                            )
                            )
                    ids_accepted.append(id)
                except Exception:
                    smiles_rejected.append(smi)
                    ids_rejected.append(id)
                    rejections.append(f'{self.n_steps} - {message}')

        if method == 'largest':
            from mlchem.chem.manipulation import PatternRecognition
            for id, smi in zip(self.ids, self.smiles):
                if dehydrate:
                    smi = Chem.MolToSmiles(
                        remove_smarts_pattern(create_molecule(smi), '[OH2]'),
                        isomericSmiles=self.IsIsomeric,
                        kekuleSmiles=self.IsKekulised,
                        canonical=self.IsCanonical
                        )     # dehydrate
                try:
                    components = smi.split('.')
                    lengths = [PatternRecognition.Base.count_atoms(c)
                               for c in components]
                    smiles_accepted.append(
                        components[np.argmax(lengths)]
                        )
                    ids_accepted.append(id)
                except Exception:
                    smiles_rejected.append(smi)
                    ids_rejected.append(id)
                    rejections.append(f'{self.n_steps} - {message}')

        self.smiles = smiles_accepted
        self.ids = ids_accepted
        self.df_accepted = pd.DataFrame(data={'id': self.ids,
                                              'SMILES': self.smiles})

        self.df_rejected = pd.concat([self.df_rejected,
                                      pd.DataFrame(data={
                                          'id': ids_rejected,
                                          'SMILES': smiles_rejected,
                                          'rejection': rejections}
                                                   )
                                      ])

    def neutralise_smiles(self,
                          isomeric: Optional[bool] = None,
                          canonical: Optional[bool] = None,
                          kekulise: Optional[bool] = None) -> None:
        """
Neutralise the SMILES strings.

This method processes the input SMILES strings to neutralise charged species.
It updates the SMILES strings and IDs, and tracks rejected SMILES strings with reasons.

Parameters
----------
isomeric : bool, optional
    Whether to generate isomeric SMILES. Default is None.
canonical : bool, optional
    Whether to generate canonical SMILES. Default is None.
kekulise : bool, optional
    Whether to kekulise the SMILES. Default is None.

Returns
-------
None
"""


        smiles_accepted = []
        ids_accepted = []
        smiles_rejected = []
        ids_rejected = []
        rejections = []

        self.n_steps += 1

        message = 'could not be neutralised'
        if isomeric is not None:
            self.IsIsomeric = isomeric
        if canonical is not None:
            self.IsCanonical = canonical
        if kekulise is not None:
            self.IsKekulised = kekulise

        for id, smi in zip(self.ids, self.smiles):
            try:
                mol = create_molecule(smi)
                mol_n = neutralise_mol(mol)
                smiles_accepted.append(
                    Chem.MolToSmiles(mol_n,
                                     isomericSmiles=self.IsIsomeric,
                                     kekuleSmiles=self.IsKekulised,
                                     canonical=self.IsCanonical)
                                     )
                ids_accepted.append(id)
            except Exception:
                smiles_rejected.append(smi)
                ids_rejected.append(id)
                rejections.append(f'{self.n_steps} - {message}')

        self.smiles = smiles_accepted
        self.ids = ids_accepted
        self.df_accepted = pd.DataFrame(
            data={'id': self.ids,
                  'SMILES': self.smiles}
                  )

        self.df_rejected = pd.concat(
            [self.df_rejected,
             pd.DataFrame(data={'id': ids_rejected,
                                'SMILES': smiles_rejected,
                                'rejection': rejections}
                          )
             ])

    def remove_carbon_ions(self) -> None:
        """
Remove SMILES strings containing carbon ions.

This method filters out SMILES strings that contain carbon ions.
It updates the internal SMILES list and IDs, and logs rejected entries
with the reason for rejection.

Returns
-------
None
"""


        message = 'carbon ion'
        self.n_steps += 1

        carbon_ion_mask = np.array(
            [PatternRecognition.Base.has_carbon_ion(smi)
             for smi in self.smiles]
             )
        smiles_accepted = np.array(self.smiles)[~carbon_ion_mask]
        smiles_rejected = np.array(self.smiles)[carbon_ion_mask]
        ids_rejected = [id for i, id in enumerate(self.ids)
                        if self.smiles[i] in smiles_rejected]
        rejections = [f'{self.n_steps} - {message}' for _ in ids_rejected]

        self.ids = [id for i, id in enumerate(self.ids)
                    if self.smiles[i] in smiles_accepted]
        self.smiles = smiles_accepted.tolist()

        self.df_accepted = pd.DataFrame(data={'id': self.ids,
                                              'SMILES': self.smiles})

        self.df_rejected = pd.concat(
            [self.df_rejected,
             pd.DataFrame(data={'id': ids_rejected,
                                'SMILES': smiles_rejected,
                                'rejection': rejections}
                          )])

    def remove_inorganics(self) -> None:
        """
Remove inorganic SMILES strings.

This method filters out SMILES strings that are classified as inorganic.
It updates the internal SMILES list and IDs, and logs rejected entries
with the reason for rejection.

Returns
-------
None
"""


        message = 'inorganic'
        self.n_steps += 1

        organic_mask = np.array(
            [PatternRecognition.Base.is_organic(smi)
             for smi in self.smiles])
        smiles_accepted = np.array(self.smiles)[organic_mask]
        ids_accepted = [id for i, id in enumerate(self.ids)
                        if self.smiles[i] in smiles_accepted]
        smiles_rejected = np.array(self.smiles)[~organic_mask]
        ids_rejected = [id for i, id in enumerate(self.ids)
                        if self.smiles[i] in smiles_rejected]
        rejections = [f'{self.n_steps} - {message}' for _ in ids_rejected]

        self.smiles = smiles_accepted.tolist()
        self.ids = ids_accepted

        self.df_accepted = pd.DataFrame(data={'id': self.ids,
                                              'SMILES': self.smiles})

        self.df_rejected = pd.concat(
            [self.df_rejected,
             pd.DataFrame(data={'id': ids_rejected,
                                'SMILES': smiles_rejected,
                                'rejection': rejections}
                          )])

    def remove_organometallics(self) -> None:
        """
Remove SMILES strings containing organometallic compounds.

This method filters out SMILES strings that contain organometallic
structures, excluding simple metal salts. It updates the internal
SMILES list and IDs, and logs rejected entries with the reason for
rejection.

Returns
-------
None
"""


        message = 'organometallic'
        self.n_steps += 1

        from mlchem.importables import metal_list
        self.metal_list = metal_list

        df_smiles = pd.DataFrame(data=self.smiles,
                                 columns=['S'],
                                 index=self.ids)
        df_metals = df_smiles[df_smiles.S.str.contains
                              ('|'.join(self.metal_list))
                              ]
        is_metal_salt = np.array(
            [PatternRecognition.Base.has_metal_salt(smi)
             for smi in df_metals.S.values])
        df_metal_salts = df_metals[is_metal_salt]

        try:
            df_metal_salts.S
        except AttributeError:
            df_metal_salts = pd.DataFrame(data=[],
                                          columns=['S'],
                                          index=[])

        df_organometallics = df_metals[~df_metals.S.isin(df_metal_salts.S)]

        smiles_rejected = list(df_organometallics.S.values)
        smiles_accepted = list(df_smiles[~df_smiles.S.isin
                                         (smiles_rejected)].S.values)
        ids_accepted = [id for i, id in enumerate(self.ids)
                        if self.smiles[i] in smiles_accepted]
        ids_rejected = [id for i, id in enumerate(self.ids)
                        if self.smiles[i] in smiles_rejected]
        rejections = [f'{self.n_steps} - {message}' for _ in ids_rejected]

        self.smiles = smiles_accepted
        self.ids = ids_accepted

        self.df_accepted = pd.DataFrame(data={'id': self.ids,
                                              'SMILES': self.smiles})

        self.df_rejected = pd.concat(
            [self.df_rejected, pd.DataFrame(data={'id': ids_rejected,
                                                  'SMILES': smiles_rejected,
                                                  'rejection': rejections})])

    def remove_mixtures(self) -> None:
        """
Remove SMILES strings that are mixtures.

This method filters out SMILES strings that represent mixtures of
multiple components, unless they are simple binary mixtures involving
metals or halogens. It updates the internal SMILES list and IDs, and
logs rejected entries with the reason for rejection.

Returns
-------
None
"""


        message = 'mixture'
        smiles_accepted = []
        ids_accepted = []
        smiles_rejected = []
        ids_rejected = []
        rejections = []

        self.n_steps += 1
        metals_modified = self.metal_list + ['Cl', 'H', 'I', 'Br', 'F']
        for id, smi in zip(self.ids, self.smiles):
            smi = Chem.MolToSmiles(
                remove_smarts_pattern(create_molecule(smi),
                                      '[OH2]')
                                      )     # dehydrate
            if '.' in smi:
                if len(smi.split('.')) == 2:     # A 2-component mixture
                    # Does the binary mixture contain a metal?
                    # if so, accept and go on
                    if any(metal in smi for metal in metals_modified):
                        smiles_accepted.append(smi)
                        ids_accepted.append(id)
                    else:
                        smiles_rejected.append(smi)
                        ids_rejected.append(id)
                        rejections.append(f'{self.n_steps} - {message}')
                else:     # More than 2 components
                    mixture = smi.split('.')
                    length = len(mixture)
                    count_metals = sum(
                        1 for component in mixture if any(
                            metal in component for metal in metals_modified
                            )
                            )
                    # If metals are not the only mixture components:
                    if count_metals < length - 1:
                        smiles_rejected.append(smi)
                        ids_rejected.append(id)
                        rejections.append(f'{self.n_steps} - {message}')
                    else:
                        smiles_accepted.append(smi)
                        ids_accepted.append(id)
            else:
                smiles_accepted.append(smi)
                ids_accepted.append(id)

        self.smiles = smiles_accepted
        self.ids = ids_accepted

        self.df_accepted = pd.DataFrame(data={'id': self.ids,
                                              'SMILES': self.smiles})

        self.df_rejected = pd.concat(
            [self.df_rejected,
             pd.DataFrame(data={'id': ids_rejected,
                                'SMILES': smiles_rejected,
                                'rejection': rejections})])

    def quality_checker(self) -> None:
        """
Perform quality check on SMILES strings.

This method evaluates the structural integrity of accepted SMILES
strings using the ChEMBL structure checker. It assigns a priority
score and message to each molecule, indicating the severity of any
issues found.

Returns
-------
None

Examples
--------
>>> cleaner.quality_checker()
>>> cleaner.df_accepted[['id', 'PRIORITY', 'MESSAGES']]
"""

        from chembl_structure_pipeline import checker

        priorities, messages = [], []
        for smiles in self.df_accepted.SMILES:
            priority = []
            message = []
            mol = Chem.MolFromSmiles(smiles)
            molblock = Chem.MolToMolBlock(mol)
            response = checker.check_molblock(molblock)
            if response == ():
                priorities.append(0)
                messages.append('ok')
            else:
                priority.append([r[0] for r in response])
                message.append([r[1] for r in response])
                priorities.append(np.sum(priority[0]))
                messages.append(message[0])

        self.df_accepted['PRIORITY'] = priorities
        self.df_accepted['MESSAGES'] = messages

        self.raised_messages = np.unique(np.hstack(
            self.df_accepted.MESSAGES.values))

        self.messages_priority_very_high = [
            'Error-9986 (Cannot process aromatic bonds)',
            'Illegal input',
            'InChI: Unknown element(s)',
            ]
        self.messages_priority_high = [
            'All atoms have zero coordinates',
            'InChI: Accepted unusual valence(s)',
            'InChI: Empty structure', 'molecule has 3D coordinates',
            'molecule has a radical that is not found in the known list',
            'molecule has six (or more) atoms with'
            ' exactly the same coordinates',
            'Number of atoms less than 1', 'Polymer information in mol file',
            'V3000 mol file',
            ]
        self.messages_priority_medium = [
            'InChI_RDKit/Mol stereo mismatch',
            'Mol/Inchi/RDKit stereo mismatch',
            'RDKit_Mol/InChI stereo mismatch',
            'molecule has a bond with an illegal stereo flag',
            'molecule has a bond with an illegal type',
            'molecule has a crossed bond in a ring',
            'molecule has two (or more) atoms with exactly '
            'the same coordinates',
        ]

        self.messages_priority_low = [
            message for message in self.raised_messages
            if message not in list(
                self.messages_priority_very_high +
                self.messages_priority_high+self.messages_priority_medium
                )]

    def full_clean(self, desalting_method: Literal['rdkit',
                                                   'chembl',
                                                   'largest'
                                                   ] = 'largest') -> None:
        """
Perform a full cleaning process on the SMILES strings.

This method sequentially applies all cleaning steps to the input
SMILES strings, including initialisation, filtering, desalting,
neutralisation, and quality checking.

Parameters
----------
desalting_method : {'rdkit', 'chembl', 'largest'}, optional
    The method to use for desalting. Default is 'largest'.

Returns
-------
None

Examples
--------
>>> cleaner = MolCleaner(smiles_list)
>>> cleaner.full_clean(desalting_method='chembl')
"""

        print('Initialising SMILES')
        self.initialise_smiles()
        print('Removing carbon ions')
        self.remove_carbon_ions()
        print('Removing inorganics')
        self.remove_inorganics()
        print('Removing organometallics')
        self.remove_organometallics()
        if desalting_method != 'largest':
            print('Removing mixtures')
            self.remove_mixtures()
        print('Desalting SMILES')
        self.desalt_smiles(method=desalting_method, dehydrate=True)
        print('Neutralising SMILES')
        self.neutralise_smiles()
        print('Performing quality check')
        self.quality_checker()
        print('DONE')


class MolGenerator:
    """
A class to generate SMILES strings using SELFIES fragments.

The `MolGenerator` class allows for the generation of new molecules
represented as SMILES strings by combining SELFIES fragments either
randomly or through substitution into a template molecule.

Examples
--------
>>> generator = MolGenerator()
>>> generator.generate_smiles(template_smiles='c1ccccc1',
...                           n_molecules=20,
...                           n_fragments=5,
...                           n_substitutions=1,
...                           attempt_limit=1000)
>>> cleaner = MolCleaner(generator.smiles_generated)
>>> cleaner.initialise_smiles()
>>> cleaner.neutralise_smiles()
>>> generator.smiles_generated = cleaner.smiles
"""


    def __init__(self,
                 dictionary: dict = None):
        """
Initialise the MolGenerator instance.

This constructor sets up the SELFIES fragment dictionary used for
molecule generation. If a custom dictionary is provided, it is used
to populate the internal fragment bag; otherwise, a default dictionary
is used.

Parameters
----------
dictionary : dict, optional
    A custom dictionary of SELFIES fragments and their frequencies.
    If None, a default dictionary is used.

Returns
-------
None
"""


        from mlchem.importables import chemical_dictionary as dictionary_default
        import selfies as sf

        if sorted(dictionary_default.keys()) != \
           sorted([a for a in sf.get_semantic_robust_alphabet()]):
            print('Semantic dictionary not up to date.')
            print('Please update the mlchem.importables.'
                  'chemical_dictionary if you want to use a pre-defined'
                  ' dictionary.')
            print('Otherwise build your custom dictionary '
                  'and feed it to the generator.')

        def populate_dictionary(dictionary_custom):     # inner helper function
            dictionary_to_populate = {}
            for k in dictionary_default.keys():
                dictionary_to_populate[k] = 0
            for k, v in zip(dictionary_custom.keys(),
                            dictionary_custom.values()):
                dictionary_to_populate[k] = v
            return dictionary_to_populate

        if dictionary is not None:
            self.dictionary = populate_dictionary(dictionary)
        else:
            self.dictionary = dictionary_default
        self.bag_of_fragments = [block for block in self.dictionary.keys()
                                 for _ in range(self.dictionary[block])]

    def generate_smiles(self,
                        n_molecules: int,
                        n_fragments: int,
                        template_smiles: str = None,
                        substitution_sites: list = None,
                        n_substitutions: int = None,
                        include_extremities: bool = True,
                        attempt_limit: int = 1000) -> None:
              
              """
      Generate SMILES strings using a template or random fragments.

      This method generates a specified number of SMILES strings either by
      randomly combining SELFIES fragments or by substituting fragments into
      a template molecule at specified positions.

      Parameters
      ----------
      n_molecules : int
          The number of molecules to generate.
      n_fragments : int
          The number of fragments to use for each molecule.
      template_smiles : str, optional
          A template SMILES string to use for generating molecules.
      substitution_sites : list of int, optional
          Indices in the SELFIES string where substitutions should occur.
      n_substitutions : int, optional
          The number of substitutions to make in the template.
      include_extremities : bool, optional
          Whether to include the start and end of the SELFIES string as
          possible substitution sites. Default is True.
      attempt_limit : int, optional
          The maximum number of attempts to generate the specified number
          of molecules. Default is 1000.

      Returns
      -------
      None

      Updates
      -------
      template_smiles : str
          The template SMILES string used for generation.
      smiles_generated : list of str
          The list of generated SMILES strings.
      selfies_generated : list of str
          The list of generated SELFIES strings.
      mols_generated : list of rdkit.Chem.rdchem.Mol
          The list of RDKit Mol objects corresponding to the generated SMILES.
      pattern_atoms : list
          A list of pattern atom matches for each generated molecule.
      double_legend : list of str
          A list of strings combining SMILES and SELFIES for each molecule.

      Examples
      --------
      >>> generator.generate_smiles(template_smiles='c1ccccc1',
      ...                           n_molecules=5,
      ...                           n_fragments=3,
      ...                           n_substitutions=1)
      """
      

              import random
              import selfies as sf
              from mlchem.helper import insert_string_piece, find_all_occurrences

              self.template_smiles = template_smiles
              self.smiles_generated = []
              self.selfies_generated = []
              self.mols_generated = []
              self.pattern_atoms = []

              if self.template_smiles is None:
                  for _ in range(n_molecules):
                      rnd_selfies = ''.join(
                          random.choice(list(
                              self.bag_of_fragments)) for _ in range(n_fragments))
                      rnd_smiles = sf.decoder(rnd_selfies)
                      self.smiles_generated.append(rnd_smiles)
                      self.selfies_generated.append(rnd_selfies)
                      self.mols_generated.append(Chem.MolFromSmiles(rnd_smiles))
              else:
                  if n_substitutions == 1 and substitution_sites is not None:
                      raise AssertionError(
                      'only 1 substitution site is accepted for this configuration')
                  if substitution_sites is not None and len(
                    substitution_sites) == 1:
                      raise AssertionError(
                          'only 1 substitution is accepted for this configuration')

                  self.template_selfies = sf.encoder(self.template_smiles)

                  if substitution_sites is not None:
                      self.substitution_sites = [0, len(self.template_selfies)] + \
                          substitution_sites if \
                          include_extremities else substitution_sites
                  else:
                      self.substitution_sites = [0, len(self.template_selfies)] + \
                          find_all_occurrences(self.template_selfies, '][') if \
                          include_extremities else \
                          find_all_occurrences(self.template_selfies, '][')

                  count_molecules = 0
                  count_attempts = 0
                  while count_molecules < n_molecules \
                          and count_attempts < attempt_limit:
                      modifying_selfies = self.template_selfies
                      for _ in range(n_substitutions):
                          rnd_selfies = ''.join(random.choices(
                              self.bag_of_fragments, k=n_fragments)
                              )
                          index = random.choice(self.substitution_sites)
                          modifying_selfies = insert_string_piece(
                              modifying_selfies, rnd_selfies, index)
                          try:
                              decoded_selfies = Chem.MolToSmiles(
                                  Chem.MolFromSmiles(
                                      sf.decoder(modifying_selfies)
                                      ))
                              count_attempts += 1
                              if decoded_selfies not in self.smiles_generated:
                                  self.template_smarts = smarts_from_string(
                                      self.template_smiles
                                      )
                                  pattern_result = PatternRecognition.\
                                      Base.\
                                      check_smarts_pattern(decoded_selfies,
                                                          self.template_smarts)
                                  if pattern_result[0]:
                                      self.smiles_generated.append(decoded_selfies)
                                      self.selfies_generated.append(
                                          modifying_selfies
                                          )
                                      self.mols_generated.append(
                                          create_molecule(decoded_selfies)
                                          )
                                      self.pattern_atoms.append(pattern_result)
                                      count_molecules += 1
                          except Exception:
                              pass

              self.double_legend = [f'{smi}\n\n\n{sel}' for smi,
                                    sel in zip(self.smiles_generated,
                                              [sf.encoder(s)
                                                for s in self.smiles_generated])
                              ]


class PatternRecognition:
    """
A utility class for recognising chemical patterns using SMARTS.

This class provides a reference for common SMARTS-based pattern
matching used in cheminformatics, particularly with RDKit. It includes
a vocabulary of generic chemical groupings and links to external
resources for further information on SMARTS syntax and usage.

References
----------
- https://www.rdkit.org/docs/RDKit_Book.html#smarts-reference
- https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html
- https://daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
- https://www.daylight.com/dayhtml_tutorials/languages/smarts/index.html
- https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html
- https://www.molsoft.com/man/smiles.html
- https://www.labcognition.com/onlinehelp/en/smiles_and_smarts_nomenclature.htm

SMARTS Vocabulary
-----------------
- Alkyl (ALK): alkyl side chains (not an H atom)
- AlkylH (ALH): alkyl side chains including an H atom
- Alkenyl (AEL): alkenyl side chains
- AlkenylH (AEH): alkenyl side chains or an H atom
- Alkynyl (AYL): alkynyl side chains
- AlkynylH (AYH): alkynyl side chains or an H atom
- Alkoxy (AOX): alkoxy side chains
- AlkoxyH (AOH): alkoxy side chains or an H atom
- Carbocyclic (CBC): carbocyclic side chains
- CarbocyclicH (CBH): carbocyclic side chains or an H atom
- Carbocycloalkyl (CAL): cycloalkyl side chains
- CarbocycloalkylH (CAH): cycloalkyl side chains or an H atom
- Carbocycloalkenyl (CEL): cycloalkenyl side chains
- CarbocycloalkenylH (CEH): cycloalkenyl side chains or an H atom
- Carboaryl (ARY): all-carbon aryl side chains
- CarboarylH (ARH): all-carbon aryl side chains or an H atom
- Cyclic (CYC): cyclic side chains
- CyclicH (CYH): cyclic side chains or an H atom
- Acyclic (ACY): acyclic side chains (not an H atom)
- AcyclicH (ACH): acyclic side chains or an H atom
- Carboacyclic (ABC): all-carbon acyclic side chains
- CarboacyclicH (ABH): all-carbon acyclic side chains or an H atom
- Heteroacyclic (AHC): acyclic side chains with at least one heteroatom
- HeteroacyclicH (AHH): acyclic side chains with at least one heteroatom or an H atom
- Heterocyclic (CHC): cyclic side chains with at least one heteroatom
- HeterocyclicH (CHH): cyclic side chains with at least one heteroatom or an H atom
- Heteroaryl (HAR): aryl side chains with at least one heteroatom
- HeteroarylH (HAH): aryl side chains with at least one heteroatom or an H atom
- NoCarbonRing (CXX): ring containing no carbon atoms
- NoCarbonRingH (CXH): ring containing no carbon atoms or an H atom
- Group (G): any group (not H atom)
- GroupH (GH): any group (including H atom)
- Group* (G*): any group with a ring closure
- GroupH* (GH*): any group with a ring closure or an H atom
"""

    class Base:
        @staticmethod
        def check_smiles_pattern(target: str | Chem.rdchem.Mol,
                                 smiles_pattern: str) -> tuple[bool,
                                                               list[int]]:
            """
Check if a given SMILES pattern matches a target molecule.

This function checks whether a SMILES pattern matches any substructure
within the target molecule.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or RDKit Mol object representing the target molecule.
smiles_pattern : str
    A SMILES pattern to match against the target.

Returns
-------
tuple of (bool, list of int)
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices where the pattern matches.
"""

            from mlchem.helper import flatten
            from rdkit import Chem

            target_mol = create_molecule(target)

            atoms = flatten(
                target_mol.GetSubstructMatches(
                    Chem.MolFromSmarts(smarts_from_string(smiles_pattern)),
                    maxMatches=target_mol.GetNumAtoms()
                    ))

            if len(atoms) == 0:
                return False, atoms
            else:
                return True, atoms

        @staticmethod
        def check_smarts_pattern(target: str | Chem.rdchem.Mol,
                                 smarts_pattern: str,
                                 generic_keywords: list = []
                                 ) -> tuple[bool,
                                            np.ndarray[int],
                                            str]:
            """
Check if a given SMARTS pattern matches a target molecule,
optionally using generic keywords.

This function supports both standard SMARTS syntax and generic
keywords for more intuitive pattern matching.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or RDKit Mol object representing the target molecule.
smarts_pattern : str
    A SMARTS pattern to match against the target.
generic_keywords : list of str, optional
    Generic keywords to substitute into the SMARTS pattern.

Returns
-------
tuple of (bool, numpy.ndarray of int, str)
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices where the pattern matches.
    - The final SMARTS pattern used for matching.

Examples
--------
>>> from mlchem.chem.manipulation import PatternRecognition as pr
>>> pr.Base.check_smarts_pattern('CCCCc1ccccc1', smarts_pattern='CC*', generic_keywords=['CYC'])
(True, array([3, 4, 5]), 'CC* |$;;CYC$|')

>>> pr.Base.check_smarts_pattern('CCCCc1ccccc1', smarts_pattern='[R]')
(True, array([4, 5, 6, 7, 8, 9]), '[R]')

>>> pr.Base.check_smarts_pattern('CCCCc1ccccc1', smarts_pattern='[*]', generic_keywords=['CYC'])
(True, array([4, 5, 6, 7, 8, 9]), '* |$CYC$|')
"""

            from mlchem.helper import flatten, process_custom_string
            from rdkit import Chem

            matchers = Chem.SubstructMatchParameters()
            matchers.useGenericMatchers = True

            if len(generic_keywords) > 0:
                processed_pattern = process_custom_string(
                    s=smarts_pattern, target_substring='*',
                    replacement_list=generic_keywords,
                    separator=';')
                processed_pattern_final = smarts_pattern + \
                    ' |$' + processed_pattern + \
                    '$|'
            else:
                processed_pattern_final = smarts_pattern

            target_mol = create_molecule(target)
            mol_pattern = Chem.MolFromSmarts(processed_pattern_final)
            Chem.SetGenericQueriesFromProperties(mol_pattern)
            atoms = np.unique(flatten(target_mol.GetSubstructMatches(mol_pattern,
                                                           matchers)))
            

            if len(atoms) == 0:
                return False, atoms, processed_pattern_final
            else:
                return True, atoms, processed_pattern_final

        @staticmethod
        def pattern_abs_fraction_greater_than(target: str | Chem.rdchem.Mol,
                                              func,
                                              threshold: float,
                                              hidden_pattern_function=None,
                                              ) -> bool:
            """
Determine if the fraction of atoms belonging to a pattern
exceeds a given threshold.

This function calculates the fraction of atoms in a target molecule
that match a given pattern. The pattern is defined by a function
(e.g. a SMARTS matcher). An optional hidden pattern function can be
provided to refine the numerator (e.g. to count only aromatic carbon atoms).

The denominator is always the total number of atoms in the molecule.

Use this when you want to know:

“Does this pattern make up more than X% of the molecule?”

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    The target molecule, as a SMILES string or RDKit Mol object.
func : callable
    A function that returns a tuple like (True, [atom_indices]) for the pattern.
threshold : float
    The minimum fraction of atoms that must match the pattern.
hidden_pattern_function : callable, optional
    A secondary function to refine the atom subset in the numerator.

Returns
-------
bool
    True if the fraction of matching atoms exceeds the threshold.

Examples
--------
>>> from mlchem.chem.manipulation import PatternRecognition as pr

>>> # Define a pattern function to find carbon atoms
>>> def check_carbon(target):
...     return pr.Base.check_smarts_pattern(target, smarts_pattern='[C]')

>>> # Check if more than 60% of atoms are carbon
>>> pr.Base.pattern_abs_fraction_greater_than('CCCC(=O)O', check_carbon, threshold=0.6)
True

>>> # Example with a hidden pattern function (e.g. aromatic carbon among all atoms)
>>> def check_aromatic(target, pattern_function):
...     return pr.Base.check_smarts_pattern(target, smarts_pattern='[a]')

>>> pr.Base.pattern_abs_fraction_greater_than('OCCc1ccccc1', check_aromatic,
...     threshold=0.5, hidden_pattern_function=check_carbon)
True

Use Cases
---------
- “Are more than 30% of all atoms aromatic carbon?”
- “Do heteroatoms make up more than 20% of the molecule?”

Notes
-----
This method always uses the total number of atoms in the molecule as the denominator.
To compare two patterns directly, use `pattern_rel_fraction_greater_than`.

>>> # Using abs method with hidden pattern
>>> pr.Base.pattern_abs_fraction_greater_than(
...     target,
...     func=check_pattern_aromatic,
...     threshold=0.5,
...     hidden_pattern_function=check_carbon)
"""





            target_atoms = PatternRecognition.Base.count_atoms(target=target)
            try:
                res = func(target=target)
            except Exception:
                res = func(target, hidden_pattern_function)
            pattern_atoms = len(res[1])

            return pattern_atoms / target_atoms > threshold

        @staticmethod
        def pattern_rel_fraction_greater_than(target: str | Chem.rdchem.Mol,
                                              func1,
                                              func2,
                                              threshold: float,
                                              hidden_pattern_function=None
                                              ) -> bool:
            """
Determine if the fraction of atoms belonging to one pattern
exceeds a given threshold relative to another pattern.

This function compares the number of atoms matching a primary pattern
to those matching a secondary pattern. An optional hidden pattern
function can be passed if the primary function requires two arguments.

Use this when you want to know:

“Does pattern A make up more than X% of pattern B?”

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    The target molecule, as a SMILES string or RDKit Mol object.
func1 : callable
    Function identifying the primary pattern (numerator).
func2 : callable
    Function identifying the reference pattern (denominator).
threshold : float
    The minimum relative fraction required (e.g. 0.5 means 50% of func2 atoms must match func1).
hidden_pattern_function : callable, optional
    A secondary function to pass into `func1` if it requires it.

Returns
-------
bool
    True if the relative fraction exceeds the threshold.

Examples
--------
>>> from mlchem.chem.manipulation import PatternRecognition as pr

>>> # Define pattern functions
>>> def check_carbon(target):
...     return pr.Base.check_smarts_pattern(target, smarts_pattern='[C]')

>>> def check_alkyl_carbon(target):
...     return pr.Base.check_smarts_pattern(target, smarts_pattern='[CX3]')

>>> # Check if more than 30% of carbon atoms are alkyl
>>> pr.Base.pattern_rel_fraction_greater_than('CC(C)C(=O)O', check_alkyl_carbon, check_carbon, threshold=0.3)
True

Use Cases
---------
- “Are more than 30% of carbon atoms alkyl?”
- “Are more than 50% of ring atoms aromatic?”

Notes
-----
This method uses the number of atoms matched by `func2` as the denominator.
If `func1` requires a second argument (e.g. a filtering function), it 
will be passed `hidden_pattern_function`.
"""


            try:
                target_atoms = len(func2(target=target)[1])
                pattern_atoms = len(func1(target=target)[1])
            except Exception as e:
                if hidden_pattern_function:
                    pattern_atoms = len(func1(target,
                                              hidden_pattern_function))
                else:
                    raise e

            return target_atoms > 0 and \
                (pattern_atoms / target_atoms) > threshold

        @staticmethod
        def get_atoms(target: str | Chem.rdchem.Mol) -> list:
            """
Retrieve a list of atoms from a target molecule.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    The target molecule.

Returns
-------
list of rdkit.Chem.rdchem.Atom
    A list of atom objects in the molecule.
"""

            return [a for a in create_molecule(target).GetAtoms()]

        @staticmethod
        def count_atoms(target: str | Chem.rdchem.Mol) -> int:
            """
Count the number of atoms in a target molecule.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    The target molecule.

Returns
-------
int
    The number of atoms in the molecule.
"""

            return len(PatternRecognition.Base.get_atoms(target))

        @staticmethod
        def get_bonds(target: str | Chem.rdchem.Mol) -> list:
            """
Retrieve a list of bonds from a target molecule.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    The target molecule.

Returns
-------
list of rdkit.Chem.rdchem.Bond
    A list of bond objects in the molecule.
"""

            return [b for b in create_molecule(target).GetBonds()]

        @staticmethod
        def count_bonds(target: str | Chem.rdchem.Mol) -> int:
            """
Count the number of bonds in a target molecule.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    The target molecule.

Returns
-------
int
    The number of bonds in the molecule.
"""

            return len(PatternRecognition.Base.get_bonds(target))

        @staticmethod
        def get_tautomers(target: str | Chem.rdchem.Mol) -> list[str]:
            """
Retrieve a list of tautomers for a target molecule.

This function takes a target molecule, which can be either a SMILES
string or an RDKit molecule object, and returns a list of SMILES
strings representing the tautomers of the molecule.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A molecule in SMILES format or an RDKit molecule object.

Returns
-------
list of str
    A list of SMILES strings representing the tautomers of the target
    molecule.
"""

            from rdkit import Chem
            from rdkit.Chem.MolStandardize import rdMolStandardize

            m = create_molecule(target)
            enumerator = rdMolStandardize.TautomerEnumerator()
            enumerator.Canonicalize(m)
            tauts = enumerator.Enumerate(m)
            return [Chem.MolToSmiles(t,
                                     kekuleSmiles=True,
                                     canonical=False)
                    for t in tauts]

        @staticmethod
        def get_stereoisomers(target: str | Chem.rdchem.Mol,
                              drawer=None) -> tuple[list[Chem.rdchem.Mol],
                                                    list]:
            """
Retrieve stereoisomers and their images for a target molecule.

This function takes a target molecule, which can be either a SMILES
string or an RDKit molecule object, and returns a list of stereoisomers
and their corresponding images. An optional MolDrawer instance can be
provided to customise drawing options.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A molecule in SMILES format or an RDKit molecule object.
drawer : MolDrawer, optional
    An instance of MolDrawer to preserve drawing options. Default is None.

Returns
-------
tuple of (list of rdkit.Chem.rdchem.Mol, list)
    A tuple containing a list of stereoisomer molecules and their
    corresponding images.
"""

            from mlchem.chem.visualise.drawing import MolDrawer
            from rdkit.Chem import EnumerateStereoisomers

            if not drawer:
                drawer = MolDrawer()
            mol = create_molecule(target)
            opts = EnumerateStereoisomers.StereoEnumerationOptions()
            opts.onlyStereoGroups = True
            enumerated_mols = list(
                EnumerateStereoisomers.EnumerateStereoisomers(mol)
                )
            drawer.update_drawing_options(addStereoAnnotation=True)
            enumerated_images = [drawer.draw_mol(m) for m in enumerated_mols]
            return enumerated_mols, enumerated_images

        @staticmethod
        def is_organic(target: str | Chem.rdchem.Mol) -> bool:
            """
Determine whether a target molecule is organic.

This function checks if a molecule contains carbon atoms and is not
classified as a carbonic acid. A molecule is considered inorganic if
the only carbon atoms present belong to carbonic acid groups.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A molecule in SMILES format or an RDKit molecule object.

Returns
-------
bool
    True if the molecule is organic, False otherwise.
"""

            carbon_atoms = list(PatternRecognition.MolPatterns.\
              check_carbon(target)[1])
            carbonic_acid_atoms = list(PatternRecognition.MolPatterns.\
              check_carbonic_acid(target)[1])
            if carbon_atoms:
                if len(list(set(carbon_atoms) & 
                        set(carbonic_acid_atoms))) == len(carbon_atoms):
                    return False
                return True
            return False
                    

        @staticmethod
        def has_carbon_ion(target: str | Chem.rdchem.Mol) -> bool:
            """
Detect the presence of carbon ions in a target molecule.

This function checks if a molecule contains charged carbon atoms
(carbocations or carbanions). Carbanions that are part of nitrile
groups are excluded unless carbocations are also present.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A molecule in SMILES format or an RDKit molecule object.

Returns
-------
bool
    True if the molecule contains carbon ions, False otherwise.
"""


            # Tuple with carbanions and carbocations.
            charged_carbons = (
                len(PatternRecognition.Base.check_smarts_pattern(target,
                                                                 '[C-]')[1]),
                len(PatternRecognition.Base.check_smarts_pattern(target,
                                                                 '[C+]')[1])
                                                                 )
            # If only carbanions come from nitrile
            # and no carbocations are present,
            # molecule is labelled as not having carbon ions.
            if charged_carbons[0] == len(PatternRecognition.
                                         Base.
                                         check_smarts_pattern(
                                             target, '[C-]#N'
                                             )) and charged_carbons[1] == 0:
                return False
            if np.sum(charged_carbons) == 0:
                return False
            return True

        @staticmethod
        def has_metal_salt(target: str | Chem.rdchem.Mol,
                           custom_metals: list | None = None) -> bool:
            """
Determine whether a target molecule contains a metal salt.

This function checks for the presence of metal salts in a molecule.
A custom list of metal elements can be provided, otherwise a default
list is used.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A molecule in SMILES format or an RDKit molecule object.
custom_metals : list of str, optional
    A custom list of metal elements to check for. Default is None.

Returns
-------
bool
    True if the molecule contains a metal salt, False otherwise.
"""

            if custom_metals is None:
                from mlchem.importables import metal_list
            else:
                metal_list = custom_metals

            salt_counter = 0
            mol = create_molecule(target)
            smiles = kekulise_smiles(Chem.MolToSmiles(mol))

            for metal_element in metal_list:
                element_length = len(metal_element)
                if metal_element in smiles:
                    element_index = smiles.find(metal_element)
                    if smiles[element_index - 1] == '[':
                        # Account for all metal forms
                        # (various charges and lengths)
                        for i in range(2, 5):
                            try:
                                if smiles[element_index - i] == '.' or \
                                 smiles[element_index + element_length +
                                   i - 1] == '.':
                                    salt_counter += 1
                                    break
                            except IndexError:
                                continue

            return salt_counter > 0

        @staticmethod
        def get_MCS(
            input1: str | Chem.rdchem.Mol, input2: str | Chem.rdchem.Mol,
            threshold: float = 0.0, completeAromaticRings: bool = False,
            similarity_type: Literal['tanimoto', 'johnson'] = 'tanimoto'
            ) -> tuple[bool, str, list[int], list[int], float]:
            """
Find the Maximum Common Substructure (MCS) between two molecules.

This function identifies the MCS between two molecules, which can be
provided as SMILES strings or RDKit molecule objects. It returns the
SMARTS string of the MCS, atom indices in both molecules, and a
similarity score.

More information:
https://greglandrum.github.io/rdkit-blog/posts/2023-11-08-introducingrascalmces.html

Parameters
----------
input1 : str or rdkit.Chem.rdchem.Mol
    The first molecule in SMILES format or as an RDKit object.
input2 : str or rdkit.Chem.rdchem.Mol
    The second molecule in SMILES format or as an RDKit object.
threshold : float, optional
    Similarity threshold for MCS detection. Must be in the interval [0, 1).
    Default is 0.0.
completeAromaticRings : bool, optional
    Whether to require complete aromatic ring matches. Default is False.
similarity_type : {'tanimoto', 'johnson'}, optional
    The similarity metric to use. Default is 'tanimoto'.

Returns
-------
tuple
    A tuple containing:
    - bool : Whether an MCS was found.
    - str : SMARTS string of the MCS.
    - list of int : Atom indices in the first molecule.
    - list of int : Atom indices in the second molecule.
    - float : Similarity score.
"""

            from rdkit.Chem import rdRascalMCES
          
            if not (0 <= threshold < 1):
                raise ValueError("'threshold' must be in the interval [0, 1)")

            mol_1 = create_molecule(input1)
            mol_2 = create_molecule(input2)
            opts = rdRascalMCES.RascalOptions()
            opts.similarityThreshold = threshold
            opts.completeAromaticRings = completeAromaticRings
            res = rdRascalMCES.FindMCES(mol_1, mol_2, opts)

            if len(res) == 0:
                return False, "", [], [], 0.0
            res = res.pop()

            smartstring_mcs = res.smartsString
            res_mcs_1 = PatternRecognition.Base.\
                check_smarts_pattern(input1, smartstring_mcs)
            res_mcs_2 = PatternRecognition.Base.\
                check_smarts_pattern(input2, smartstring_mcs)

            atoms_mcs_1 = res_mcs_1[1]
            atoms_mcs_2 = res_mcs_2[1]

            if similarity_type == 'tanimoto':
                try:
                    sim = len(res.bondMatches()) / (mol_1.GetNumBonds() +
                                                    mol_2.GetNumBonds() -
                                                    len(res.bondMatches())
                                                    )
                except IndexError:
                    sim = 0.0
            elif similarity_type == 'johnson':
                sim = res.similarity
            else:
                raise ValueError(
                    "similarity type must be either 'tanimoto' or 'johnson.'"
                    )

            return (True,
                    smartstring_mcs,
                    atoms_mcs_1,
                    atoms_mcs_2,
                    sim)

    class Atoms:

        @staticmethod
        def is_SP(atom: Chem.rdchem.Atom) -> int:
            """
Check if an atom is SP-hybridised.

This method evaluates whether the given RDKit atom is SP-hybridised
(i.e., has linear geometry with two electron domains).

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The atom to check.

Returns
-------
int
    1 if the atom is SP-hybridised, 0 otherwise.

Examples
--------
>>> atom = mol.GetAtomWithIdx(0)
>>> MolDrawer.is_SP(atom)
"""

            return int(atom.GetHybridization() ==
                       Chem.rdchem.HybridizationType.SP)

        @staticmethod
        def is_SP2(atom: Chem.rdchem.Atom) -> int:
            """
Check if an atom is SP2-hybridised.

This method evaluates whether the given RDKit atom is SP2-hybridised,
which typically corresponds to trigonal planar geometry with three
electron domains.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The atom to check.

Returns
-------
int
    1 if the atom is SP2-hybridised, 0 otherwise.

Examples
--------
>>> atom = mol.GetAtomWithIdx(1)
>>> MolDrawer.is_SP2(atom)
"""

            return int(atom.GetHybridization() ==
                       Chem.rdchem.HybridizationType.SP2)

        @staticmethod
        def is_SP3(atom: Chem.rdchem.Atom) -> int:
            """
Check if an atom is SP3-hybridised.

This method determines whether the given RDKit atom is SP3-hybridised,
which typically corresponds to tetrahedral geometry with four electron
domains.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The atom to check.

Returns
-------
int
    1 if the atom is SP3-hybridised, 0 otherwise.

Examples
--------
>>> atom = mol.GetAtomWithIdx(2)
>>> MolDrawer.is_SP3(atom)
"""

            return int(atom.GetHybridization() ==
                       Chem.rdchem.HybridizationType.SP3)

        @staticmethod
        def get_ring_size(atom: Chem.rdchem.Atom) -> int:
            """
Get the size of the ring an atom belongs to.

This method checks whether the given RDKit atom is part of a ring,
and if so, determines the size of the smallest ring it is in. If the
atom is not in any ring, the method returns 0.

Parameters
----------
atom : rdkit.Chem.rdchem.Atom
    The atom whose ring membership and size is to be evaluated.

Returns
-------
int
    The size of the smallest ring the atom is part of, or 0 if the atom
    is not in a ring.

Examples
--------
>>> atom = mol.GetAtomWithIdx(3)
>>> MolDrawer.get_ring_size(atom)
"""

            import numpy as np

            if atom.IsInRing():
                return np.argmax(
                    [atom.IsInRingSize(n) for n in range(1, 101)]
                    )+1
            else:
                return 0

    class Bonds:

        @staticmethod
        def is_single_bond(bond: Chem.rdchem.Bond) -> int:
          """
Check if a bond is a single bond.

This function takes an RDKit bond object and checks whether it is a
single bond.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    An RDKit bond object.

Returns
-------
int
    1 if the bond is a single bond, 0 otherwise.
"""
          return int(bond.GetBondType() == Chem.rdchem.BondType.SINGLE)

        @staticmethod
        def is_double_bond(bond: Chem.rdchem.Bond) -> int:
            """
Check if a bond is a double bond.

This function takes an RDKit bond object and checks whether it is a
double bond.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    An RDKit bond object.

Returns
-------
int
    1 if the bond is a double bond, 0 otherwise.
"""

            return int(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)

        @staticmethod
        def is_triple_bond(bond: Chem.rdchem.Bond) -> int:
            """
Check if a bond is a triple bond.

This function takes an RDKit bond object and checks whether it is a
triple bond.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    An RDKit bond object.

Returns
-------
int
    1 if the bond is a triple bond, 0 otherwise.
"""

            return int(bond.GetBondType() == Chem.rdchem.BondType.TRIPLE)

        @staticmethod
        def is_dative_bond(bond: Chem.rdchem.Bond) -> int:
            """
Check if a bond is a dative bond.

This function takes an RDKit bond object and checks whether it is a
dative bond.

Parameters
----------
bond : rdkit.Chem.rdchem.Bond
    An RDKit bond object.

Returns
-------
int
    1 if the bond is a dative bond, 0 otherwise.
"""

            return int(bond.GetBondType() == Chem.rdchem.BondType.DATIVE)

        @staticmethod
        def check_bonds(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                list[int],
                                                                str]:
            """
Check the bonds in a molecule.

This function takes a molecule and performs a generic bond pattern
check using the SMARTS pattern `*~*`.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A molecule in SMILES format or an RDKit molecule object.

Returns
-------
tuple
    A tuple containing:
    - bool : Whether the pattern was found.
    - list of int : Atom indices that match the pattern.
    - str : The SMARTS string representing the matched pattern.
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '*~*')

        @staticmethod
        def check_rotatable_bonds(target: str | Chem.rdchem.Mol) -> tuple[bool, list[int], str]:
            """
Check for rotatable bonds in a molecule.

This function takes a molecule and performs a rotatable bond pattern
check using the SMARTS pattern:
`[!$(*#*)&!D1]-!@[!$(*#*)&!D1]`.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A molecule in SMILES format or an RDKit molecule object.

Returns
-------
tuple
    A tuple containing:
    - bool : Whether the pattern was found.
    - list of int : Atom indices that match the pattern.
    - str : The SMARTS string representing the matched pattern.
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(create_molecule(target,add_hydrogens=True),
                                     '[!$(*#*)&!D1]-!@[!$(*#*)&!D1]')

        @staticmethod
        def check_single_bonds(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for single bonds in a molecule.

This function takes a molecule and performs a single bond pattern
check using the SMARTS pattern `*-*`.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A molecule in SMILES format or an RDKit molecule object.

Returns
-------
tuple
    A tuple containing:
    - bool : Whether the pattern was found.
    - list of int : Atom indices that match the pattern.
    - str : The SMARTS string representing the matched pattern.
"""
            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '*-*')

        @staticmethod
        def check_double_bonds(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for double bonds in a molecule.

This function takes a molecule and performs a double bond pattern
check using the SMARTS pattern `*=*`.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A molecule in SMILES format or an RDKit molecule object.

Returns
-------
tuple
    A tuple containing:
    - bool : Whether the pattern was found.
    - list of int : Atom indices that match the pattern.
    - str : The SMARTS string representing the matched pattern.
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '*=*')

        @staticmethod
        def check_triple_bonds(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for triple bonds in a molecule.

This function takes a molecule and performs a triple bond pattern
check using the SMARTS pattern `*#*`.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A molecule in SMILES format or an RDKit molecule object.

Returns
-------
tuple
    A tuple containing:
    - bool : Whether the pattern was found.
    - list of int : Atom indices that match the pattern.
    - str : The SMARTS string representing the matched pattern.
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '*#*')

        @staticmethod
        def check_aromatic_bonds(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for aromatic bonds in a molecule.

This function takes a molecule and performs an aromatic bond pattern
check using the SMARTS pattern `*:*`.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A molecule in SMILES format or an RDKit molecule object.

Returns
-------
tuple
    A tuple containing:
    - bool : Whether the pattern was found.
    - list of int : Atom indices that match the pattern.
    - str : The SMARTS string representing the matched pattern.
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '*:*')

        @staticmethod
        def check_cyclic_bonds(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for cyclic bonds in a molecule.

This function takes a molecule and performs a cyclic bond pattern
check using the SMARTS pattern `*@*`.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A molecule in SMILES format or an RDKit molecule object.

Returns
-------
tuple
    A tuple containing:
    - bool : Whether the pattern was found.
    - list of int : Atom indices that match the pattern.
    - str : The SMARTS string representing the matched pattern.
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '*@*')

    class MolPatterns:

        @staticmethod
        def check_pattern_aromatic(target: str | Chem.rdchem.Mol,
                                   pattern_function) -> tuple[bool,
                                                              list[int],
                                                              tuple[str, str]]:
            """
Check if a pattern is aromatic in a molecule.

This function takes a molecule and a pattern function, and checks
whether the pattern is aromatic in the molecule.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.
pattern_function : Callable
    A function that takes a molecule and returns a tuple of
    (bool, list of atom indices, SMARTS string).

Returns
-------
tuple[bool, list[int], tuple[str, str]]
    A tuple containing:
    - A boolean indicating whether the pattern is aromatic.
    - A list of atom indices matching the aromatic pattern.
    - A tuple of SMARTS strings representing the matched patterns.

Examples
--------
>>> MolPatterns.check_pattern_aromatic("c1ccccc1", some_function)
"""

            from mlchem.helper import flatten

            _, b1, c1 = pattern_function(target)
            _, b2, c2 = PatternRecognition.Base.\
                check_smarts_pattern(target, '[a]')

            intersection = flatten(list(set(b1) & set(b2)))
            boolean_response = len(intersection) > 0

            return (boolean_response,
                    intersection,
                    (c1, c2)
                    )

        @staticmethod
        def check_pattern_aromatic_substituent(
            target: str | Chem.rdchem.Mol,
            pattern_function) -> tuple[bool,
                                       list[int],
                                       tuple[str, str]]:
            """
Check if a pattern is an aromatic substituent in a molecule.

This function takes a molecule and a pattern function, and checks
whether the pattern is an aromatic substituent in the molecule.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.
pattern_function : Callable
    A function that takes a molecule and returns a tuple of
    (bool, list of atom indices, SMARTS string).

Returns
-------
tuple[bool, list[int], tuple[str, str]]
    A tuple containing:
    - A boolean indicating whether the pattern is an aromatic substituent.
    - A list of atom indices matching the substituent pattern.
    - A tuple of SMARTS strings representing the matched patterns.

Examples
--------
>>> MolPatterns.check_pattern_aromatic_substituent("c1ccccc1C", some_function)
"""

            from mlchem.helper import flatten

            a1, b1, c1 = pattern_function(target)
            a2, b2, c2 = PatternRecognition.Base.\
                check_smarts_pattern(target, '[a]!@[*]')
            b2 = list(b2)[1:]     # exclude aromatic atom 

            intersection = flatten(list(set(b1) & set(b2)))
            boolean_response = len(intersection) > 0

            return (boolean_response,
                    intersection,
                    (c1, c2)
                    )

        @staticmethod
        def check_pattern_aliphatic(
            target: str | Chem.rdchem.Mol,
            pattern_function) -> tuple[bool,
                                       list[int],
                                       tuple[str, str]]:
            """
Check if a pattern is aliphatic in a molecule.

This function takes a molecule and a pattern function, and checks
whether the pattern is aliphatic in the molecule.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.
pattern_function : Callable
    A function that takes a molecule and returns a tuple of
    (bool, list of atom indices, SMARTS string).

Returns
-------
tuple[bool, list[int], tuple[str, str]]
    A tuple containing:
    - A boolean indicating whether the pattern is aliphatic.
    - A list of atom indices matching the aliphatic pattern.
    - A tuple of SMARTS strings representing the matched patterns.

Examples
--------
>>> MolPatterns.check_pattern_aliphatic("CC", some_function)
"""

            from mlchem.helper import flatten

            a1, b1, c1 = pattern_function(target)
            a2, b2, c2 = PatternRecognition.Base.\
                check_smarts_pattern(target, '[A]')

            intersection = flatten(list(set(b1) & set(b2)))
            boolean_response = len(intersection) > 0

            return (boolean_response,
                    intersection,
                    (c1, c2))

    # # Patterns by element #  #

    # C #

        @staticmethod
        def check_carbon(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                 list[int],
                                                                 str]:
            """
Check for carbon atoms in a molecule.

The SMARTS pattern used to identify carbon atoms is ``[#6]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_carbon("CCO")
"""

            return PatternRecognition.Base.check_smarts_pattern(target, '[#6]')

        @staticmethod
        def check_carbanion(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for carbanions in a molecule.

The SMARTS pattern used to identify carbanions is ``[#6-]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_carbanion("[CH2-]C")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[#6-]')

        @staticmethod
        def check_carbocation(
            target: str | Chem.rdchem.Mol
        ) -> tuple[bool, list[int], str]:
            """
Check for carbocations in a molecule.

The SMARTS pattern used to identify carbocations is ``[#6+]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_carbocation("[CH3+]")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[#6+]')

        @staticmethod
        def check_alkyl_carbon(
            target: str | Chem.rdchem.Mol
        ) -> tuple[bool, list[int], str]:
            """
Check for alkyl carbon atoms in a molecule.

The SMARTS pattern used to identify alkyl carbon atoms is ``[CX4]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_alkyl_carbon("CC")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[CX4]')

        @staticmethod
        def check_allenic_carbon(
            target: str | Chem.rdchem.Mol
        ) -> tuple[bool, list[int], str]:
            """
Check for allenic carbon atoms in a molecule.

The SMARTS pattern used to identify allenic carbon atoms is
``$([CX2=C)]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_allenic_carbon("C=C=C")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, '[$([CX2](=C)=C)]')

        @staticmethod
        def check_vinylic_carbon(
            target: str | Chem.rdchem.Mol
        ) -> tuple[bool, list[int], str]:
            """
Check for vinylic carbon atoms in a molecule.

The SMARTS pattern used to identify vinylic carbon atoms is
``[$([CX3]=[CX3])]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_vinylic_carbon("C=CC")
"""

            return PatternRecognition.Base.check_smarts_pattern(
                target, '[$([CX3]=[CX3])]'
            )

        @staticmethod
        def check_acetylenic_carbon(
            target: str | Chem.rdchem.Mol
        ) -> tuple[bool, list[int], str]:
            """
Check for acetylenic carbon atoms in a molecule.

The SMARTS pattern used to identify acetylenic carbon atoms is
``[$([CX2]#C)]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_acetylenic_carbon("CC#C")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[$([CX2]#C)]')

        # C, O #

        @staticmethod
        def check_carbonyl(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                   list[int],
                                                                   str]:
            """
Check for carbonyl groups in a molecule.

The SMARTS pattern used to identify carbonyl groups is
``[$([CX3]=[OX1]),$([CX3+]-[OX1-])]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_carbonyl("CC(=O)O")
"""

            return PatternRecognition.Base.check_smarts_pattern(
                target, '[$([CX3]=[OX1]),$([CX3+]-[OX1-])]')

        @staticmethod
        def check_acyl_halide(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for acyl halides in a molecule.

The SMARTS pattern used to identify acyl halides is
``CX3[F,Cl,Br,I]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_acyl_halide("CC(=O)Cl")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, '[CX3](=[OX1])[F,Cl,Br,I]')

        @staticmethod
        def check_aldehyde(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                   list[int],
                                                                   str]:
            """
Check for aldehyde groups in a molecule.

The SMARTS pattern used to identify aldehyde groups is
``CX3H1[#6]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_aldehyde("CC=O")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, '[CX3H1](=O)[#6]')

        @staticmethod
        def check_anhydride(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for anhydride groups in a molecule.

The SMARTS pattern used to identify anhydride groups is
``CX3[OX2][CX3].

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_anhydride("CC(=O)OC(=O)C")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, '[CX3](=[OX1])[OX2][CX3](=[OX1])')

        @staticmethod
        def check_carboxyl(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                   list[int],
                                                                   str]:
            """
Check for carboxyl groups in a molecule.

The SMARTS pattern used to identify carboxyl groups is
``CX3[OX1H0-,OX2H1]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_carboxyl("CC(=O)O")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, '[CX3](=O)[OX1H0-,OX2H1]')

        @staticmethod
        def check_carbonic_acid(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for carbonic acid groups in a molecule.

The SMARTS pattern used to identify carbonic acid groups is
``CX3([OX2])[OX2H,OX1H0-1]``. This pattern matches both the
acid and its conjugate base, but not carbonic acid diesters.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_carbonic_acid("O=C(O)O")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target,
                                     '[CX3](=[OX1])([OX2])[OX2H,OX1H0-1]')

        @staticmethod
        def check_carbonate_ester(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], tuple[str, str]]:
            """
Check for carbonate esters in a molecule.

This function identifies mono- and diesters of carbonic acid in the
target molecule.

SMARTS patterns used:
- Monoester: ``CX3([OX2H0])[OX2H,OX1H0-1]``
- Diester: ``CX3([OX2H0])[OX2H0]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], tuple[str, str]]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A tuple of SMARTS strings representing the matched patterns.

Examples
--------
>>> MolPatterns.check_carbonate_ester("COC(=O)OC")
"""
            from mlchem.helper import flatten

            pattern_monoester = '[CX3](=[OX1])([OX2H0])[OX2H,OX1H0-1]'
            pattern_diester = '[CX3](=[OX1])([OX2H0])[OX2H0]'

            a1, b1, c1 = PatternRecognition.Base.\
                check_smarts_pattern(target, pattern_monoester)
            a2, b2, c2 = PatternRecognition.Base.\
                check_smarts_pattern(target, pattern_diester)

            return (bool(a1 + a2), flatten((b1, b2)), (c1, c2))

        @staticmethod
        def check_ester(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                list[int],
                                                                str]:
            """
Check for ester groups in a molecule.

The SMARTS pattern used to identify ester groups is
``[#6]CX3[OX2H0][#6]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_ester("CC(=O)OC")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, '[#6][CX3](=O)[OX2H0][#6]')

        @staticmethod
        def check_ketone(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                 list[int],
                                                                 str]:
            """
Check for ketone groups in a molecule.

The SMARTS pattern used to identify ketone groups is
``[#6]CX3[#6]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_ketone("CC(=O)C")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, '[#6][CX3](=O)[#6]')

        @staticmethod
        def check_ether(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                list[int],
                                                                str]:
            """
Check for ether groups in a molecule.

The SMARTS pattern used to identify ether groups is
``OD2[#6]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_ether("COC")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, '[OD2]([#6])[#6]')

        @staticmethod
        def check_alpha_diketone(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for alpha-diketone groups in a molecule.

The SMARTS pattern used to identify alpha-diketone groups is
``O=#6D3#6D3=O``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_alpha_diketone("CC(=O)CC(=O)C")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, 'O=[#6D3]([#6])[#6D3]([#6])=O')

        @staticmethod
        def check_beta_diketone(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for beta-diketone groups in a molecule.

The SMARTS pattern used to identify beta-diketone groups is
``O=#6D3[#6]#6D3=O``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_beta_diketone("CC(=O)CCC(=O)C")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target,
                                     'O=[#6D3]([#6])[#6][#6D3]([#6])=O')

        @staticmethod
        def check_gamma_diketone(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for gamma-diketone groups in a molecule.

The SMARTS pattern used to identify gamma-diketone groups is
``O=#6D3[#6][#6]#6D3=O``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_gamma_diketone("CC(=O)CCCC(=O)C")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target,
                                          'O=[#6D3]([#6])[#6][#6][#6D3]([#6])=O')

        @staticmethod
        def check_alpha_dicarbonyl(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for alpha-dicarbonyl groups in a molecule.

The SMARTS pattern used to identify alpha-dicarbonyl groups is
``O=[#6][#6]=O``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_alpha_dicarbonyl("O=CC=O")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, 'O=[#6][#6]=O')

        @staticmethod
        def check_beta_dicarbonyl(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for beta-dicarbonyl groups in a molecule.

The SMARTS pattern used to identify beta-dicarbonyl groups is
``O=[#6][#6][#6]=O``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_beta_dicarbonyl("O=CCC=O")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, 'O=[#6][#6][#6]=O')

        @staticmethod
        def check_gamma_dicarbonyl(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for gamma-dicarbonyl groups in a molecule.

The SMARTS pattern used to identify gamma-dicarbonyl groups is
``O=[#6][#6][#6][#6]=O``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_gamma_dicarbonyl("O=CCCC=O")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, 'O=[#6][#6][#6][#6]=O')

        @staticmethod
        def check_delta_dicarbonyl(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for delta-dicarbonyl groups in a molecule.

The SMARTS pattern used to identify delta-dicarbonyl groups is
``O=[#6][#6][#6][#6][#6]=O``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_delta_dicarbonyl("O=CCCCC=O")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, 'O=[#6][#6][#6][#6][#6]=O')

        # N #

        @staticmethod
        def check_nitrogen(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                   list[int],
                                                                   str]:
            """
Check for nitrogen atoms in a molecule.

The SMARTS pattern used to identify nitrogen atoms is ``[#7]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_nitrogen("CN")
"""

            return PatternRecognition.Base.check_smarts_pattern(target, '[#7]')

        @staticmethod
        def check_amine(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], tuple[str, str, str, str]]:
            """
Check for amine groups in a molecule.

This function checks for primary, secondary, tertiary, and quaternary
amine groups in the target molecule. The results are combined and
returned as a single tuple.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], tuple[str, str, str, str]]
    A tuple containing:
    - A boolean indicating whether any amine group was found.
    - A list of atom indices matching any of the amine group patterns.
    - A tuple of SMARTS strings representing the matched patterns for
      each type of amine group.

Examples
--------
>>> MolPatterns.check_amine("CN(C)C")
"""

            from mlchem.helper import flatten

            a1, b1, c1 = PatternRecognition.MolPatterns.\
                check_amine_primary(target)
            a2, b2, c2 = PatternRecognition.MolPatterns.\
                check_amine_secondary(target)
            a3, b3, c3 = PatternRecognition.MolPatterns.\
                check_amine_tertiary(target)
            a4, b4, c4 = PatternRecognition.MolPatterns.\
                check_amine_quaternary(target)

            return (bool(a1 + a2 + a3 + a4),
                    flatten((b1, b2, b3, b4)),
                    (c1, c2, c3, c4))

        @staticmethod
        def check_amine_primary(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], tuple[str, str]]:
            """
Check for primary amine groups in a molecule.

SMARTS patterns used:
- Nitrogen: ``[#7]``
- Amine: ``[#6][#7D1&!$(NC=O)]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], tuple[str, str]]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A tuple of SMARTS strings representing the matched patterns.

Examples
--------
>>> MolPatterns.check_amine_primary("CCNH2")
"""

            from mlchem.helper import flatten

            pattern_nitrogen = '[#7]'
            pattern_amine = '[#6][#7D1&!$(NC=O)]'

            a1, b1, c1 = PatternRecognition.Base.\
                check_smarts_pattern(target, pattern_nitrogen)
            a2, b2, c2 = PatternRecognition.Base.\
                check_smarts_pattern(target, pattern_amine)

            return (bool(a1 * a2),
                    flatten(set(b1) & set(b2)),
                    (c1, c2))

        @staticmethod
        def check_amine_secondary(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], tuple[str, str]]:
            """
Check for secondary amine groups in a molecule.

SMARTS patterns used:
- Nitrogen: ``[#7]``
- Amine: ``[#6][#7D2&!$(NC=O)][#6]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], tuple[str, str]]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A tuple of SMARTS strings representing the matched patterns.

Examples
--------
>>> MolPatterns.check_amine_secondary("CCNC")
"""

            from mlchem.helper import flatten

            pattern_nitrogen = '[#7]'
            pattern_amine = '[#6][#7D2&!$(NC=O)][#6]'

            a1, b1, c1 = PatternRecognition.Base.\
                check_smarts_pattern(target, pattern_nitrogen)
            a2, b2, c2 = PatternRecognition.Base.\
                check_smarts_pattern(target, pattern_amine)

            return (bool(a1 * a2),
                    flatten(set(b1) & set(b2)),
                    (c1, c2))

        @staticmethod
        def check_amine_tertiary(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], tuple[str, str]]:
            """
Check for tertiary amine groups in a molecule.

SMARTS patterns used:
- Nitrogen: ``[#7]``
- Amine: ``[#6]#7D3&!$(NC=O)[#6]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], tuple[str, str]]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A tuple of SMARTS strings representing the matched patterns.

Examples
--------
>>> MolPatterns.check_amine_tertiary("CN(C)C")
"""

            from mlchem.helper import flatten

            pattern_nitrogen = '[#7]'
            pattern_amine = '[#6][#7D3&!$(NC=O)]([#6])[#6]'

            a1, b1, c1 = PatternRecognition.Base.\
                check_smarts_pattern(target, pattern_nitrogen)
            a2, b2, c2 = PatternRecognition.Base.\
                check_smarts_pattern(target, pattern_amine)

            return (bool(a1 * a2),
                    flatten(set(b1) & set(b2)),
                    (c1, c2))

        @staticmethod
        def check_amine_quaternary(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], tuple[str, str]]:
            """
Check for quaternary amine groups in a molecule.

SMARTS patterns used:
- Nitrogen: ``[#7]``
- Amine: ``[#6][#7D4+&!$NC=O)([#6])([#6])``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], tuple[str, str]]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A tuple of SMARTS strings representing the matched patterns.

Examples
--------
>>> MolPatterns.check_amine_quaternary("CN+(C)C")
"""

            from mlchem.helper import flatten

            pattern_nitrogen = '[#7]'
            pattern_amine = '[#6][#7D4+&!$(NC=O)]([#6])([#6])([#6])'

            a1, b1, c1 = PatternRecognition.Base.\
                check_smarts_pattern(target, pattern_nitrogen)
            a2, b2, c2 = PatternRecognition.Base.\
                check_smarts_pattern(target, pattern_amine)

            return (bool(a1 * a2),
                    flatten(set(b1) & set(b2)),
                    (c1, c2))

        @staticmethod
        def check_enamine(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for enamine groups in a molecule.

The SMARTS pattern used to identify enamine groups is
``[NX3][CX3]=[CX3]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_enamine("C=CN")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, '[NX3][CX3]=[CX3]')

        @staticmethod
        def check_amide(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                list[int],
                                                                str]:
            """
Check for amide groups in a molecule.

The SMARTS pattern used to identify amide groups is
``[#7X3]#6X3[#6]``.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_amide("CC(=O)NC")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, '[#7X3][#6X3](=[OX1])[#6]')

        @staticmethod
        def check_carbamate(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for carbamate groups in a molecule.

SMARTS pattern used:
- ``[NX3,NX4+]CX3[OX2,OX1-]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_carbamate("CN(C)C(=O)OC")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target,
                                     '[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]')

        @staticmethod
        def alpha_nitroalkane(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for alpha-nitroalkane groups in a molecule.

SMARTS pattern used:
- ``[CX4H1,H2,H3]#7D3[#8]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.alpha_nitroalkane("CC(N(=O)=O)C")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, '[CX4H1,H2,H3][#7D3](=[#8])[#8]')

        @staticmethod
        def check_alanine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for alanine residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$(NX3H(C))]CX4HCX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_alanine("CC(C(=O)O)N")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                        target, '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]'
                        '([CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]'
                        )

        @staticmethod
        def check_arginine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                   list[int],
                                                                   str]:
            """
Check for arginine residues in a molecule.

SMARTS pattern used:
- ``[CX3[OX2])CH1X4[CH2X4][CH2X4][CH2X4][ND2]=CD3[NX3]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_arginine("NC(CCCNC(N)=N)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[CX3](=[OX1])([OX2])[CH1X4]([NX3])[CH2X4]'
                    '[CH2X4][CH2X4][ND2]=[CD3]([NX3])[NX3]'
                        )

        @staticmethod
        def check_asparagine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                     list[int],
                                                                     str]:
            """
Check for asparagine residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$([NXH(C))]CX4H[NX3H2])CX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_asparagine("NC(CC(=O)N)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]'
                    '([CH2X4][CX3](=[OX1])[NX3H2])[CX3](=[OX1])[OX2H,OX1-,N]'
                        )

        @staticmethod
        def check_aspartate(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for aspartate residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$(NX3H(C))][CX4H]H0-,OH])CX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_aspartate("NC(CC(=O)O)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4]'
                    '[CX3](=[OX1])[OH0-,OH])[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_cysteine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                   list[int],
                                                                   str]:
            """
Check for cysteine residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$(NX3H(C))][CX4H]([CH2X4][SX2H,SX1H0-])CX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTe matched pattern.

Examples
--------
>>> MolPatterns.check_cysteine("C(C(C(=O)O)N)S")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4]'
                    '[SX2H,SX1H0-])[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_glutamate(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for glutamate residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$(NX3H(C))]CX4H[OH0-,OH])CX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_glutamate("NC(CCC(=O)O)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4]'
                    '[CH2X4][CX3](=[OX1])[OH0-,OH])[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_glycine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for glycine residues in a molecule.

SMARTS pattern used:
- ``[$([$([NX3H2,NX4H3+]),$(NX3H(C))][CX4H2]CX3[OX2H,OX1-,N])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_glycine("C(C(=O)O)N")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3]'
                    '(=[OX1])[OX2H,OX1-,N])]'
                    )

        @staticmethod
        def check_histidine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for histidine residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$(NX3H(C))]CX4H,$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1)CX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_histidine("NC(Cc1c[nH]cn1)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4]'
                    '[#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),'
                    '$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:'
                    '[#7X3H]),$([#7X3H])]:[#6X3H]1)[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_isoleucine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                     list[int],
                                                                     str]:
            """
Check for isoleucine residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$(NX3H(C))]CX4H[CH2X4][CH3X4])CX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_isoleucine("CC(C)CC(N)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CHX4]'
                    '([CH3X4])[CH2X4][CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_leucine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for leucine residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$(NX3H(C))]CX4H[CH3X4])CX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_leucine("CC(C)C(C(=O)O)N")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4]'
                    '[CHX4]([CH3X4])[CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_lysine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                 list[int],
                                                                 str]:
            """
Check for lysine residues in a molecule.

SMARTS pattern used:
- ``CX3([OX2])CH1X4[CH2X4][CH2X4][CH2X4]CD3[CD1]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_lysine("NCCCCCC(N)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[CX3](=[OX1])([OX2])[CH1X4]([NX3])[CH2X4][CH2X4][CH2X4][CD3]([CD1])[CD1]'
                    )

        @staticmethod
        def check_methionine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                     list[int],
                                                                     str]:
            """
Check for methionine residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$([NX3H]))]CX4HCX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_methionine("CSCC(C(=O)O)N")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4]'
                    '[CH2X4][SX2][CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_phenylalanine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                        list[int],
                                                                        str]:
            """
Check for phenylalanine residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$(NX3H(C))][CX4H]([OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_phenylalanine("NC(CC1=CC=CC=C1)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]'
                    '([CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1)'
                    '[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_proline(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for proline residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H,NX4H2+]),$(NX3(C)(C))]1CX4HCX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_proline("C1CC(NC1)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]'
                    '([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_serine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                 list[int],
                                                                 str]:
            """
Check for serine residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$(NX3H(C))]CX4HCX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_serine("NC(CO)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4]'
                    '[OX2H])[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_threonine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for threonine residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$(NX3H(C))]CX4H[OX2H])CX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_threonine("CC(O)C(N)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CHX4]'
                    '([CH3X4])[OX2H])[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_tryptophan(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                     list[int],
                                                                     str]:
            """
Check for tryptophan residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$(NX3H(C))]CX4HCX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_tryptophan("NC(Cc1c[nH]c2ccccc12)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4]'
                    '[cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H]'
                    '[cX3]12)[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_tyrosine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                   list[int],
                                                                   str]:
            """
Check for tyrosine residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$(NX3H(C))]CX4H[cX3H][cX3H]1)CX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_tyrosine("NC(Cc1ccc(O)cc1)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CH2X4]'
                    '[cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H]'
                    '[cX3H]1)[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_valine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                 list[int],
                                                                 str]:
            """
Check for valine residues in a molecule.

SMARTS pattern used:
- ``[$([NX3H2,NX4H3+]),$(NX3H(C))][CX4H](3X4])CX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_valine("CC(C)C(N)C(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([CHX4]'
                    '([CH3X4])[CH3X4])[CX3](=[OX1])[OX2H,OX1-,N]'
                    )

        @staticmethod
        def check_aminoacid(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for generic amino acid residues in a molecule, including proline and glycine.

SMARTS patterns used:
- Generic amino acid: ``[$([NX3H2,NX4H3+]),$(NX3H(C))]CX4HCX3[OX2H,OX1-,N]``
- Glycine: ``[$([$([NX3H2,NX4H3+]),$(NX3H(C))][CX4H2]CX3[OX2H,OX1-,N])]``
- Proline: ``[$([NX3H,NX4H2+]),$(NX3(C)(C))]1CX4HCX3[OX2H,OX1-,N]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether any of the patterns were found.
    - A list of atom indices matching the first found pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_aminoacid("C(C(=O)O)N")
"""


            patterns = [
                '[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([*])'
                '[CX3](=[OX1])[OX2H,OX1-,N]',     # Generic
                '[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2]'
                '[CX3](=[OX1])[OX2H,OX1-,N])]',     # Glycine
                '[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2]'
                '[CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]',     # Proline
                ]

            for p in patterns:
                a, b, c = PatternRecognition.Base.\
                    check_smarts_pattern(target, p)
                if a:
                    return a, b, c
            return False, (), ()

        @staticmethod
        def check_azide(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                list[int],
                                                                str]:
            """
Check for azide groups in a molecule.

SMARTS pattern used:
- ``[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_azide("CCN=[N+]=[N-]")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]'
                    )

        @staticmethod
        def check_azo(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                              list[int],
                                                              str]:
            """
Check for azo groups in a molecule.

SMARTS pattern used:
- ``[#6][#7D2]=[#7D2][#6]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_azo("C1=CC=C(C=C1)N=NC2=CC=CC=C2")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(target,
                                     '[#6][#7D2]=[#7D2][#6]')

        @staticmethod
        def check_azoxy(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                list[int],
                                                                str]:
            """
Check for azoxy groups in a molecule.

SMARTS pattern used:
- ``[$([NX2]=NX3+[#6]),$([NX2]=NX3+0[#6])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_azoxy("CC1=NN(O)=CC=C1")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]'
                    )

        @staticmethod
        def check_diazo(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                list[int],
                                                                str]:
            """
Check for diazo groups in a molecule.

SMARTS pattern used:
- ``[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_diazo("C=N=N")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(target,
                                     '[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]'
                                     )

        @staticmethod
        def check_hydrazine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for hydrazine groups in a molecule.

SMARTS pattern used:
- ``[NX3][NX3]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_hydrazine("NN")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[NX3][NX3]')

        @staticmethod
        def check_hydrazone(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for hydrazone groups in a molecule.

Hydrazones are compounds with the structure R₂C=NNR₂, derived from 
aldehydes or ketones by replacing =O with =NNH₂ or analogues.
Reference: https://doi.org/10.1351/goldbook.H02884

SMARTS pattern used:
- ``[#7X3][#7D2]=#6D3[#6]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_hydrazone("C=NN(C)C")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target,
                                          '[#7X3][#7D2]=[#6D3]([#6])[#6]')

        @staticmethod
        def check_imine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                list[int],
                                                                str]:
            """
Check for imine groups in a molecule.

SMARTS pattern used:
- ``$([CX3[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_imine("C=NC")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),'
                    '$([NX2H])]'
                    )

        @staticmethod
        def check_iminium(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for iminium groups in a molecule.

SMARTS pattern used:
- ``[NX3+]=[CX3]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_iminium("C=N+C")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[NX3+]=[CX3]')

        @staticmethod
        def check_imide(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                list[int],
                                                                str]:
            """
Check for imide groups in a molecule.

SMARTS pattern used:
- ``[#6][#6D3#7X3]#6D3[#6]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_imide("O=C1NC(=O)CC1")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[#6][#6D3](=[#8D1])[#7X3][#6D3](=[#8D1])[#6]'
                    )

        @staticmethod
        def check_nitrate(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for nitrate groups in a molecule.

SMARTS pattern used:
- ``$([NX3(=[OX1])O),$(NX3+(=[OX1])O)]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_nitrate("C(C(=O)O)N+[O-]")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]'
                    )

        @staticmethod
        def check_nitro(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                list[int],
                                                                str]:
            """
Check for nitro groups in a molecule.

SMARTS pattern used:
- ``$(NX3=O),$([NX3+[O-])][!#8]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_nitro("CC(=O)N(=O)=O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'
                    )

        @staticmethod
        def check_nitrile(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for nitrile groups in a molecule.

SMARTS pattern used:
- ``[NX1]#[CX2]``

Compounds with the structure RC≡N, i.e., C-substituted derivatives of hydrocyanic acid.

Reference
---------
https://doi.org/10.1351/goldbook.N04151

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_nitrile("CC#N")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[NX1]#[CX2]')

        @staticmethod
        def check_isonitrile(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                     list[int],
                                                                     str]:
            """
Check for isonitrile groups in a molecule.

SMARTS pattern used:
- ``[CX1-]#[NX2+]``

Isomeric forms of hydrocyanic acid and its derivatives (RN≡C).

Reference
---------
https://doi.org/10.1351/goldbook.I03270

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_isonitrile("CN#C")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[CX1-]#[NX2+]')

        @staticmethod
        def check_nitroso(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for nitroso groups in a molecule.

SMARTS pattern used:
- ``[NX2]=[OX1]``

Nitroso groups (-NO) attached to carbon or other elements.

Reference
---------
https://doi.org/10.1351/goldbook.N04169

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_nitroso("C1=CC=CC=C1N=O")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[NX2]=[OX1]')

        @staticmethod
        def check_n_oxide(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for N-oxide groups in a molecule.

SMARTS pattern used:
- ``[$([#7X3H1,#7X3&!#7X3H2,#7X3H0,#7X4+][#8]); !$(#7~[O]); !$([#7]=[#7])]``

Derived from tertiary amines by attachment of an oxygen atom to nitrogen.

Reference
---------
https://doi.org/10.1351/goldbook.A00273

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_n_oxide("CN+(C)[O-]")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([#7X3H1,#7X3&!#7X3H2,#7X3H0,#7X4+][#8]);'
                    '!$([#7](~[O])~[O]);!$([#7]=[#7])]'
                    )

        @staticmethod
        def check_cyanamide(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for cyanamide groups in a molecule.

SMARTS pattern used:
- ``[NX3][CX2]#[NX1]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_cyanamide("NC#N")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[NX3][CX2]#[NX1]')

        @staticmethod
        def check_cyanate(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for cyanate groups in a molecule.

SMARTS pattern used:
- ``[#8D2][#6D2]#[#7D1]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_cyanate("OC#N")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target,
                                          '[#8D2][#6D2]#[#7D1]')

        @staticmethod
        def check_isocyanate(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                     list[int],
                                                                     str]:
            """
Check for isocyanate groups in a molecule.

SMARTS pattern used:
- ``[#7D2]=[#6D2]=[#8D1]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_isocyanate("N=C=O")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target,
                                          '[#7D2]=[#6D2]=[#8D1]')

        # O #

        @staticmethod
        def check_oxygen(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                 list[int],
                                                                 str]:
            """
Check for oxygen atoms in a molecule.

SMARTS pattern used:
- ``[#8]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_oxygen("CCO")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[#8]')

        @staticmethod
        def check_alcohol(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for alcohol groups in a molecule.

SMARTS pattern used:
- ``[#6][OX2H]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_alcohol("CCO")
"""

            return PatternRecognition.Base.\
                check_smarts_pattern(target, '[#6][OX2H]')

        @staticmethod
        def check_enol(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                               list[int],
                                                               str]:
            """
Check for enol groups in a molecule. Matches both enol and enolate forms.

SMARTS pattern used:
- ``[$([OX2H][#6X3]=[#6]),$([OX1-][#6X3]=[#6])]``

Enols are vinylic alcohols with the structure HOCR'=CR₂, tautomeric with aldehydes or ketones.

Reference
---------
https://doi.org/10.1351/goldbook.E02124

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_enol("C=C(O)C")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(target,
                                     '[$([OX2H][#6X3]=[#6]),'
                                     '$([OX1-][#6X3]=[#6])]'
                                     )

        # P #

        @staticmethod
        def check_phosphorus(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                     list[int],
                                                                     str]:
            """
Check for phosphorus atoms in a molecule.

SMARTS pattern used:
- ``[P]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_phosphorus("CP(=O)(O)O")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[P]')

        @staticmethod
        def check_phosphoric_acid(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                          list[int],
                                                                          str]:
            """
Check for phosphoric acid groups in a molecule.

SMARTS pattern used:
- ``[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]), $(P+([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]``

This pattern matches orthophosphoric acid and polyphosphoric acid anhydrides, but not mono- or di-esters of monophosphoric acid.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_phosphoric_acid("OP(=O)(O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])'
                    '([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),'
                    '$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),'
                    '$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),'
                    '$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]'
                    )

        @staticmethod
        def check_phosphoric_ester(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                           list[int],
                                                                           str]:
            """
Check for phosphoric ester groups in a molecule.

SMARTS pattern used:
- ``[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)]), $(P+([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]``

This pattern matches both neutral and charged forms of phosphoric esters, but not non-ester phosphoric acids.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_phosphoric_ester("COP(=O)(OC)OC")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),'
                    '$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),'
                    '$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),'
                    '$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),'
                    '$([OX2][#6]),$([OX2]P)])]'
                    )

        # S #

        @staticmethod
        def check_sulphur(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for sulphur atoms in a molecule.

SMARTS pattern used:
- ``[#16]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphur("CCS")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[#16]')

        @staticmethod
        def check_thiol(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                list[int],
                                                                str]:
            """
Check for thiol groups in a molecule.

SMARTS pattern used:
- ``[#16X2H]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_thiol("CCSH")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[#16X2H]')

        @staticmethod
        def check_thiocarbonyl(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                       list[int],
                                                                       str]:
            """
Check for thiocarbonyl groups in a molecule.

SMARTS pattern used:
- ``[#6X3]=[#16X1]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_thiocarbonyl("CC(=S)C")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#6X3]=[#16X1]')

        @staticmethod
        def check_thioketone(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                     list[int],
                                                                     str]:
            """
Check for thioketone groups in a molecule.

SMARTS pattern used:
- ``[#6]#6D3=[#16X1]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_thioketone("CC(=S)C")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#6][#6D3]([#6])=[#16X1]')

        @staticmethod
        def check_thioaldehyde(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for thioaldehyde groups in a molecule.

SMARTS pattern used:
- ``[#6][#6X3H1]=[#16X1]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_thioaldehyde("CC(=S)H")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target,
                                          '[#6][#6X3H1]=[#16X1]')

        @staticmethod
        def check_thioanhydride(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for thioanhydride groups in a molecule.

SMARTS pattern used:
- ``CX3[SX2]CX3``

Thioanhydrides are compounds with the structure acyl-S-acyl, also called diacylsulfanes.

Reference
---------
https://doi.org/10.1351/goldbook.T06351

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_thioanhydride("CC(=O)SC(=O)C")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(target, '[CX3](=[OX1])[SX2][CX3](=[OX1])')

        @staticmethod
        def check_thiocarboxylic(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for thiocarboxylic groups in a molecule.

SMARTS pattern used:
- ``$([$([CX3[OX2H1]),$(CX3[OX1-])]), $($([CX3[SX2H1]),$(CX3[SX1-])]), $($([CX3[SX2H1]),$(CX3[SX1-])])]``

Thiocarboxylic acids are compounds where one or both oxygens of a carboxy group are replaced by sulphur.

Reference
---------
https://doi.org/10.1351/goldbook.T06352

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_thiocarboxylic("CC(=S)SH")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([$([CX3](=[SX1])[OX2H1]),$([CX3](=[SX1])'
                    '[OX1-])]),$([$([CX3](=[SX1])[SX2H1]),$([CX3](=[SX1])'
                    '[SX1-])]),$([$([CX3](=[OX1])[SX2H1]),$([CX3]'
                    '(=[OX1])[SX1-])])]'
                    )

        @staticmethod
        def check_thioester(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for thioester groups in a molecule.

SMARTS pattern used:
- ``[$(S([#6])CX3),$(O([#6])CX3),$(#16CX3)]``

Matches mono- and di-thioesters.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_thioester("CC(=O)SC")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$(S([#6])[CX3](=O)),$(O([#6])[CX3](=S)),'
                    '$([#16]([#6])[CX3](=S))]'
                    )

        @staticmethod
        def check_sulphide(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                   list[int],
                                                                   str]:
            """
Check for sulphide groups in a molecule.

SMARTS pattern used:
- ``[#6][#16D2][#6]``

Sulphides are compounds with the structure R-S-R' (R ≠ H), also known as thioethers.

Reference
---------
https://doi.org/10.1351/goldbook.S06102

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphide("CCSC")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#6][#16D2][#6]')

        @staticmethod
        def check_disulphide(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                     list[int],
                                                                     str]:
            """
Check for disulphide groups in a molecule.

SMARTS pattern used:
- ``[#16X2H0][#16X2H0]``

Disulphides contain an S-S bond, commonly found in biological systems such as cystine.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_disulphide("CSSC")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#16X2H0][#16X2H0]')

        @staticmethod
        def check_thiocarbamate(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for thiocarbamate groups in a molecule.

SMARTS pattern used:
- ``[$([#6][#8D2]CD3[#7X3,#7X4+]), $([#6][#16D2]CD3[#7X3,#7X4+])]``

Matches both O- and S-organyl thiocarbamates and their conjugated bases.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_thiocarbamate("COC(=S)NC")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target, '[$([#6][#8D2][CD3](=[S])[#7X3,#7X4+]),'
                    '$([#6][#16D2][CD3](=[O])[#7X3,#7X4+])]'
                    )

        @staticmethod
        def check_thiocyanate(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for thiocyanate groups in a molecule.

SMARTS pattern used:
- ``[#16D2]#[#7]``

Thiocyanates are salts and esters of thiocyanic acid (HSC≡N), e.g. methyl thiocyanate (CH₃SC≡N).

Reference
---------
https://doi.org/10.1351/goldbook.T06353

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_thiocyanate("CSC#N")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[#16D2]#[#7]')

        @staticmethod
        def check_isothiocyanate(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for isothiocyanate groups in a molecule.

SMARTS pattern used:
- ``[#7D2]=[#6]=[#16D1]``

Isothiocyanates are sulphur analogues of isocyanates (RN=C=S).

Reference
---------
https://doi.org/10.1351/goldbook.I03320

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_isothiocyanate("NC(=S)N")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#7D2]=[#6]=[#16D1]')

        @staticmethod
        def check_sulphinic_acid(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for sulphinic acid groups in a molecule.

SMARTS pattern used:
- ``[$([#6]#16X3[OX2H,OX1H0-]), $([#6]#16X3+[OX2H,OX1H0-])]``

Sulphinic acids (RS(=O)OH) and their conjugate bases (sulphinates) are included.

Reference
---------
https://doi.org/10.1351/goldbook.S06109

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphinic_acid("CS(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([#6][#16X3](=[OX1])[OX2H,OX1H0-]),'
                    '$([#6][#16X3+]([OX1-])[OX2H,OX1H0-])]')

        @staticmethod
        def check_sulphinic_ester(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for sulphinic ester groups in a molecule.

SMARTS pattern used:
- ``[$([#6]#16X3[OX2][#6]), $([#6]#16X3+[OX2][#6])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphinic_ester("CS(=O)OC")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([#6][#16X3](=[OX1])[OX2][#6]),'
                    '$([#6][#16X3+]([OX1-])[OX2][#6])]')

        @staticmethod
        def check_sulphone(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                   list[int],
                                                                   str]:
            """
Check for sulphone groups in a molecule.

SMARTS pattern used:
- ``$([#16X4=[OX1]), $(#16X4+2[OX1-])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphone("CS(=O)(=O)C")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]')

        @staticmethod
        def check_carbosulphone(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for carbosulphone groups in a molecule.

SMARTS pattern used:
- ``$([#16X4(=[OX1])([#6])[#6]), $(#16X4+2([OX1-])([#6])[#6])]``

Carbosulphones are sulphones with two carbon substituents.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_carbosulphone("CCS(=O)(=O)C")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),'
                    '$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]')

        @staticmethod
        def check_sulphonic_acid(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for sulphonic acid groups in a molecule.

SMARTS pattern used:
- ``$([#16X4(=[OX1])[OX2H,OX1H0-]), $([#16X42([OX1-])[OX2H,OX1H0-])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphonic_acid("CS(=O)(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([#16X4](=[OX1])(=[OX1])[OX2H,OX1H0-]),'
                    '$([#16X4+2]([OX1-])([OX1-])[OX2H,OX1H0-])]')

        @staticmethod
        def check_sulphonic_ester(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for sulphonic ester groups in a molecule.

SMARTS pattern used:
- ``$([#16X4(=[OX1])[OX2H0]), $(#16X4+2([OX1-])[OX2H0])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphonic_ester("CS(=O)(=O)OC")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([#16X4](=[OX1])(=[OX1])[OX2H0]),'
                    '$([#16X4+2]([OX1-])([OX1-])[OX2H0])]')

        @staticmethod
        def check_sulphonamide(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for sulphonamide groups in a molecule.

SMARTS pattern used:
- ``$([SX4(=[OX1])([!O])[NX3]), $(SX4+2([OX1-])([!O])[NX3])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphonamide("CS(=O)(=O)N")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),'
                    '$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]')

        @staticmethod
        def check_sulphoxide(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                     list[int],
                                                                     str]:
            """
Check for sulphoxide groups in a molecule.

SMARTS pattern used:
- ``[$([#16X3]=[OX1]), $([#16X3+][OX1-])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphoxide("CS(=O)C")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(target,
                                     '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]')

        @staticmethod
        def check_carbosulphoxide(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for carbosulphoxide groups in a molecule.

SMARTS pattern used:
- ``$([#16X3([#6])[#6]), $(#16X3+([#6])[#6])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_carbosulphoxide("CCS(=O)C")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([#16X3](=[OX1])([#6])[#6]),'
                    '$([#16X3+]([OX1-])([#6])[#6])]')

        @staticmethod
        def check_sulphuric_acid(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for sulphuric acid groups in a molecule.

SMARTS pattern used:
- ``$([SX4(=[OX1])([OX2H1,OX1-])[OX2H1,OX1-])]``

Matches both acid and conjugate base forms.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphuric_acid("OS(=O)(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([SX4](=[OX1])(=[OX1])([OX2H1,OX1-])[OX2H1,OX1-])]')

        @staticmethod
        def check_sulphuric_ester(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for sulphuric ester groups in a molecule.

SMARTS pattern used:
- ``$([SX4(=[OX1])([OX2H1])[OX2H0][#6]), $(SX4(=[OX1])([OX2H0][#6])[OX2H0][#6])]``

Matches both mono- and di-esters of sulphuric acid.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphuric_ester("COS(=O)(=O)OC")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([SX4](=[OX1])(=[OX1])([OX2H1])[OX2H0][#6]),'
                    '$([SX4](=[OX1])(=[OX1])([OX2H0][#6])[OX2H0][#6])]')

        @staticmethod
        def check_sulphamic_acid(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for sulphamic acid groups in a molecule.

SMARTS pattern used:
- ``$([#16X4(=[OX1])(=[OX1])[OX2H,OX1H0-]), $(#16X4+2([OX1-])([OX1-])[OX2H,OX1H0-])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphamic_acid("NS(=O)(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([#16X4]([NX3,NX4+])(=[OX1])(=[OX1])'
                    '[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])'
                    '([OX1-])[OX2H,OX1H0-])]')

        @staticmethod
        def check_sulphamic_ester(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for sulphamic ester groups in a molecule.

SMARTS pattern used:
- ``$([#16X4(=[OX1])(=[OX1])[OX2][#6]), $(#16X4+2([OX1-])([OX1-])[OX2][#6])]``

Parameters
----------
kit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphamic_ester("NS(=O)(=O)OC")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),'
                    '$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]')

        @staticmethod
        def check_sulphenic_acid(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for sulphenic acid groups in a molecule.

SMARTS pattern used:
- ``[#16X2][OX2H,OX1H0-]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphenic_acid("CSO")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#16X2][OX2H,OX1H0-]')

        @staticmethod
        def check_sulphenic_ester(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for sulphenic ester groups in a molecule.

SMARTS pattern used:
- ``[#16X2][OX2H0][#6]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_sulphenic_ester("CSOC")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#16X2][OX2H0][#6]')

        # X #

        @staticmethod
        def check_halogen(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for halogen atoms in a molecule.

SMARTS pattern used:
- ``[F,Cl,Br,I]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether any halogen atom was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_halogen("CCCl")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[F,Cl,Br,I]')

        @staticmethod
        def check_fluorine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                   list[int],
                                                                   str]:
            """
Check for fluorine atoms in a molecule.

SMARTS pattern used:
- ``[F]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether fluorine was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_fluorine("CCF")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[F]')

        @staticmethod
        def check_chlorine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                   list[int],
                                                                   str]:
            """
Check for chlorine atoms in a molecule.

SMARTS pattern used:
- ``[Cl]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether chlorine was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_chlorine("CCCl")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[Cl]')

        @staticmethod
        def check_bromine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                  list[int],
                                                                  str]:
            """
Check for bromine atoms in a molecule.

SMARTS pattern used:
- ``[Br]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether bromine was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_bromine("CCBr")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[Br]')

        @staticmethod
        def check_iodine(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                 list[int],
                                                                 str]:
            """
Check for iodine atoms in a molecule.

SMARTS pattern used:
- ``[I]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether iodine was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_iodine("CCI")
"""

            return PatternRecognition.Base.check_smarts_pattern(target, '[I]')

        @staticmethod
        def check_halogen_carbon(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for carbon atoms connected to halogens in a molecule.

SMARTS pattern used:
- ``[#6]~[F,Cl,Br,I]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_halogen_carbon("CCCl")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#6]~[F,Cl,Br,I]')

        @staticmethod
        def check_halogen_nitrogen(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for nitrogen atoms connected to halogens in a molecule.

SMARTS pattern used:
- ``[#7]~[F,Cl,Br,I]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_halogen_nitrogen("NCl")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#7]~[F,Cl,Br,I]')

        @staticmethod
        def check_halogen_oxygen(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for oxygen atoms connected to halogens in a molecule.

SMARTS pattern used:
- ``[#8]~[F,Cl,Br,I]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_halogen_oxygen("OCl")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#8]~[F,Cl,Br,I]')

        @staticmethod
        def check_haloalkane(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                     list[int],
                                                                     str]:
            """
Check for haloalkane groups in a molecule.

SMARTS pattern used:
- ``[CX4]-[F,Cl,Br,I]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_haloalkane("CCl")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[CX4]-[F,Cl,Br,I]')

        @staticmethod
        def check_haloalkane_primary(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for primary haloalkane groups in a molecule.

SMARTS pattern used:
- ``[CX4H3,CX4H2]-[F,Cl,Br,I]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_haloalkane_primary("CCl")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[CX4H3,CX4H2]-[F,Cl,Br,I]')

        @staticmethod
        def check_haloalkane_secondary(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for secondary haloalkane groups in a molecule.

SMARTS pattern used:
- ``[CX4H1]-[F,Cl,Br,I]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_haloalkane_secondary("CCClCC")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[CX4H1]-[F,Cl,Br,I]')

        @staticmethod
        def check_haloalkane_tertiary(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for tertiary haloalkane groups in a molecule.

SMARTS pattern used:
- ``[CX4H0]-[F,Cl,Br,I]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_haloalkane_tertiary("C(C)(C)(Cl)C")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[CX4H0]-[F,Cl,Br,I]')

        @staticmethod
        def check_haloalkene(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                     list[int],
                                                                     str]:
            """
Check for haloalkene groups in a molecule.

SMARTS pattern used:
- ``[C&!c]=[C&!c][F,Cl,Br,I]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_haloalkene("C=CCl")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[C&!c]=[C&!c][F,Cl,Br,I]')

        @staticmethod
        def check_oxohalide(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for oxohalide groups in a molecule.

SMARTS pattern used:
- ``[#8]=[*H0]~[F,Cl,Br,I]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_oxohalide("O=CCl")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target,
                                          '[#8]=[*H0]~[F,Cl,Br,I]')

        # # Other groups # #

        @staticmethod
        def check_alkali_metals(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for alkali metals in a molecule.

SMARTS pattern used:
- ``[Li,Na,K,Rb,Cs,Fr]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether any alkali metal was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_alkali_metals("[Na+]")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[Li,Na,K,Rb,Cs,Fr]')

        @staticmethod
        def check_alkaline_earth_metals(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for alkaline earth metals in a molecule.

SMARTS pattern used:
- ``[Be,Mg,Ca,Sr,Ba,Ra]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether any alkaline earth metal was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_alkaline_earth_metals("[Mg++]")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[Be,Mg,Ca,Sr,Ba,Ra]')

        @staticmethod
        def check_transition_metals(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for transition metals in a molecule.

SMARTS pattern used:
- ``[Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr,Rf,Db,Sg,Bh,Hs,Mt,Ds,Rg,Cn]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether any transition metal was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_transition_metals("[Fe++]")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(target,
                                     '[Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Y,'
                                     'Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,La,Ce,'
                                     'Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,'
                                     'Tm,Yb,Lu,Ac,Th,Pa,U,Np,Pu,Am,Cm,'
                                     'Bk,Cf,Es,Fm,Md,No,Lr,Rf,Db,Sg,Bh,'
                                     'Hs,Mt,Ds,Rg,Cn]')

        @staticmethod
        def check_boron_group_elements(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for boron group elements in a molecule.

SMARTS pattern used:
- ``[B,Al,Ga,In,Ti,Nh]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether any boron group element was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_boron_group_elements("B(Cl)(Cl)Cl")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[B,Al,Ga,In,Ti,Nh]')

        @staticmethod
        def check_carbon_group_elements(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for carbon group elements in a molecule (excluding carbon).

SMARTS pattern used:
- ``[Si,Ge,Sn,Pb,Fl]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether any carbon group element was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_carbon_group_elements("Si(Cl)(Cl)(Cl)Cl")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[Si,Ge,Sn,Pb,Fl]')

        @staticmethod
        def check_nitrogen_group_elements(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for nitrogen group elements in a molecule (excluding nitrogen and phosphorus).

SMARTS pattern used:
- ``[As,Sb,Bi]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether any nitrogen group element was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_nitrogen_group_elements("As")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[As,Sb,Bi]')

        @staticmethod
        def check_chalcogens(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for chalcogens in a molecule (excluding oxygen and sulphur).

SMARTS pattern used:
- ``[Se,Te,Po,Lv]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether any chalcogen was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_chalcogens("Se=C")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[Se,Te,Po,Lv]')

        @staticmethod
        def check_noble_gases(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for noble gases in a molecule.

SMARTS pattern used:
- ``[He,Ne,Ar,Kr,Xe,Rn]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether any noble gas was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_noble_gases("[Ar]")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[He,Ne,Ar,Kr,Xe,Rn]')

        # # Other Structures # #

        @staticmethod
        def check_pos_charge_1(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for positively charged atoms in a molecule.

SMARTS pattern used:
- ``[+]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether any positive charge was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_pos_charge_1("C[N+](C)(C)C")
"""

            return PatternRecognition.Base.check_smarts_pattern(target, '[+]')

        @staticmethod
        def check_pos_charge_2(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for two positively charged atoms in a molecule.

SMARTS pattern used:
- ``[+].[+]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether two positive charges were found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_pos_charge_2("[Na+].[Na+]")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[+].[+]')

        @staticmethod
        def check_pos_charge_3(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for three positively charged atoms in a molecule.

SMARTS pattern used:
- ``[+].[+].[+]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether three positive charges were found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_pos_charge_3("[Na+].[Na+].[K+]")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[+].[+].[+]')

        @staticmethod
        def check_neg_charge_1(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for negatively charged atoms in a molecule.

SMARTS pattern used:
- Negative charge: ``[-]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_neg_charge_1("[O-]C(=O)C")
"""

            return PatternRecognition.Base.check_smarts_pattern(target, '[-]')

        @staticmethod
        def check_neg_charge_2(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for two negatively charged atoms in a molecule.

SMARTS pattern used:
- Two negative charges: ``[-].[-]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_neg_charge_2("[O-].[O-]C(=O)C")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[-].[-]')

        @staticmethod
        def check_neg_charge_3(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for three negatively charged atoms in a molecule.

SMARTS pattern used:
- Three negative charges: ``[-].[-].[-]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_neg_charge_3("[O-].[O-].[O-]C(=O)C")
"""

            return PatternRecognition.Base.check_smarts_pattern(target,
                                                                '[-].[-].[-]')

        @staticmethod
        def check_zwitterion(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                                     list[int],
                                                                     str]:
            """
Check for zwitterions in a molecule.

SMARTS pattern used:
- Zwitterion: multiple patterns with oppositely charged atoms separated 
by up to ten bonds.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_zwitterion("CN+(C)CC(=O)[O-]")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[$([!-0!-1!-2!-3!-4]~*~[!+0!+1!+2!+3!+4]),'
                    '$([!-0!-1!-2!-3!-4]~*~*~[!+0!+1!+2!+3!+4]),'
                    '$([!-0!-1!-2!-3!-4]~*~*~*~[!+0!+1!+2!+3!+4]),'
                    '$([!-0!-1!-2!-3!-4]~*~*~*~*~[!+0!+1!+2!+3!+4]),'
                    '$([!-0!-1!-2!-3!-4]~*~*~*~*~*~[!+0!+1!+2!+3!+4]),'
                    '$([!-0!-1!-2!-3!-4]~*~*~*~*~*~*~[!+0!+1!+2!+3!+4]'
                    '),$([!-0!-1!-2!-3!-4]~*~*~*~*~*~*~*~[!+0!+1!+2!+3!'
                    '+4]),$([!-0!-1!-2!-3!-4]~*~*~*~*~*~*~*~*~[!+0!+1!'
                    '+2!+3!+4]),$([!-0!-1!-2!-3!-4]~*~*~*~*~*~*~*~*~*~'
                    '[!+0!+1!+2!+3!+4])]')

        @staticmethod
        def check_hbond_acceptors(
          target: str | Chem.rdchem.Mol
           ) -> tuple[bool, list[int], str]:
            """
Check for hydrogen bond acceptors in a molecule.

SMARTS pattern used:
- H-bond acceptor: ``[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_hbond_acceptors("CC(=O)O")
"""

            return PatternRecognition.\
                Base.\
                check_smarts_pattern(
                    target,
                    '[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,'
                    '*+1,*+2,*+3])]')

        @staticmethod
        def check_hbond_acceptors_higher_than(target: str | Chem.rdchem.Mol,
                                              n: int) -> tuple[bool,
                                                               list[int],
                                                               str]:
            """
Check for a number of hydrogen bond acceptors strictly higher than a 
threshold.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.
n : int
    Minimum number of hydrogen bond acceptors (exclusive).

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the number of acceptors is greater 
    than `n`.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_hbond_acceptors_higher_than("CC(=O)O", 1)
"""

            _, atoms, pattern = PatternRecognition.MolPatterns.\
                check_hbond_acceptors(target)
            if len(atoms) > n:
                return True, atoms, pattern
            else:
                return False, atoms, pattern

        @staticmethod
        def check_hbond_acceptors_lower_than(target: str | Chem.rdchem.Mol,
                                             n: int) -> tuple[bool,
                                                              list[int],
                                                              str]:
            """
Check for a number of hydrogen bond acceptors strictly lower than a 
threshold.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.
n : int
    Maximum number of hydrogen bond acceptors (exclusive).

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the number of acceptors is less 
    than `n`.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_hbond_acceptors_lower_than("CC(=O)O", 3)
"""

            _, atoms, pattern = PatternRecognition.MolPatterns.\
                check_hbond_acceptors(target)
            if len(atoms) < n:
                return True, atoms, pattern
            else:
                return False, atoms, pattern

        @staticmethod
        def check_hbond_donors(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for hydrogen bond donors in a molecule.

SMARTS pattern used:
- H-bond donor: ``[!$([#6,H0,-,-2,-3])]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_hbond_donors("CC(O)N")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[!$([#6,H0,-,-2,-3])]')

        @staticmethod
        def check_hbond_donors_higher_than(target: str | Chem.rdchem.Mol,
                                           n: int) -> tuple[bool,
                                                            list[int],
                                                            str]:
            """
Check for a number of hydrogen bond donors strictly higher than a threshold.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.
n : int
    Minimum number of hydrogen bond donors (exclusive).

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the number of donors is greater than `n`.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_hbond_donors_higher_than("CC(O)N", 1)
"""

            _, atoms, pattern = PatternRecognition.MolPatterns.\
                check_hbond_donors(target)
            if len(atoms) > n:
                return True, atoms, pattern
            else:
                return False, atoms, pattern

        @staticmethod
        def check_hbond_donors_lower_than(target: str | Chem.rdchem.Mol,
                                          n: int) -> tuple[bool,
                                                           list[int],
                                                           str]:
            """
Check for a number of hydrogen bond donors strictly lower than a threshold.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.
n : int
    Maximum number of hydrogen bond donors (exclusive).

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the number of donors is less than `n`.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_hbond_donors_lower_than("CC(O)N", 3)
"""

            _, atoms, pattern = PatternRecognition.MolPatterns.\
                check_hbond_donors(target)
            if len(atoms) < n:
                return True, atoms, pattern
            else:
                return False, atoms, pattern

        @staticmethod
        def check_unbranched_rotatable_chain(target: str | Chem.rdchem.Mol,
                                             n_units: int) -> tuple[bool,
                                                                    list[int],
                                                                    str]:
            """
Check for unbranched rotatable chains in a molecule.

SMARTS pattern used:
- Unbranched rotatable chain: ``[R0;D2]-`` repeated `n_units` times.

Matches: Any non-cyclic, aliphatic atom with two connections 
(degree 2), regardless of element type.

Use case: More general — it detects unbranched chains of any atoms 
(not just carbon) that are rotatable and not part of a ring

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.
n_units : int
    Number of repeated rotatable units.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_unbranched_rotatable_chain("CCCC", 3)
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, n_units * '[R0;D2]-'[:-1])

        @staticmethod
        def check_unbranched_rotatable_carbons(
            target: str | Chem.rdchem.Mol,
            n_units: int
             ) -> tuple[bool, list[int], str]:
            """
Check for unbranched rotatable carbon chains in a molecule.

SMARTS pattern used:
- Unbranched rotatable carbon: ``[R0;CD2]-`` repeated `n_units` times.

Matches: Specifically carbon atoms (C) that are non-cyclic (R0) and have 
two connections (D2).

Use case: More specific — it only detects unbranched chains made of 
aliphatic carbon atoms.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.
n_units : int
    Number of repeated carbon rotatable units.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_unbranched_rotatable_carbons("CCCC", 3)
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, n_units * '[R0;CD2]-'[:-1])

        @staticmethod
        def check_unbranched_structure(target: str | Chem.rdchem.Mol,
                                       n_units: int) -> tuple[bool,
                                                              list[int],
                                                              str]:
            """
Check for unbranched chains in a molecule.

SMARTS pattern used:
- Unbranched chain: ``[R0;D2]~`` repeated `n_units` times

Matches: Any non-cyclic atom with two connections, connected by any bond 
type (~ = single, double, or triple).

Use case: More general — detects unbranched chains regardless of bond 
type (e.g. C=C-C≡C), and not limited to rotatable bonds.

This pattern matches non-cyclic atoms with two connections (degree 2),
connected by any bond type (single, double, or triple), forming a linear,
unbranched chain.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.
n_units : int
    Number of repeated unbranched units.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> MolPatterns.check_unbranched_structure("CC=CCCC", 4)
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, n_units * '[R0;D2]~'[:-1])

        # Ring systems #

    class Rings:

        @staticmethod
        def check_ring(target: str | Chem.rdchem.Mol) -> tuple[bool,
                                                               list[int],
                                                               str]:
            """
Check for ring atoms in a molecule.

SMARTS pattern used:
- Ring atom: ``[R]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> Rings.check_ring("c1ccccc1")
"""

            return PatternRecognition.Base.check_smarts_pattern(target, '[R]')

        @staticmethod
        def check_pattern_cyclic(
            target: str | Chem.rdchem.Mol,
            pattern_function: Callable
             ) -> tuple[bool, list[int], tuple[str, str]]:
            """
Check whether a given pattern overlaps with ring atoms in a molecule.

This method checks if the atoms matched by a custom pattern function
intersect with atoms that are part of a ring.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.
pattern_function : Callable
    A function that returns a SMARTS match result for a specific pattern.

Returns
-------
tuple[bool, list[int], tuple[str, str]]
    A tuple containing:
    - A boolean indicating whether the pattern overlaps with ring atoms.
    - A list of atom indices in the intersection.
    - A tuple of SMARTS strings representing the matched patterns.

Examples
--------
>>> Rings.check_pattern_cyclic("C1CCCOC1", MolPatterns.check_oxygen)
"""

            from mlchem.helper import flatten

            _, b1, c1 = pattern_function(target)
            _, b2, c2 = PatternRecognition.Rings.check_ring(target)

            intersection = flatten(list(set(b1) & set(b2)))
            boolean_response = len(intersection) > 0

            return boolean_response, intersection, (c1, c2)

        @staticmethod
        def check_pattern_cyclic_substituent(
            target: str | Chem.rdchem.Mol,
            pattern_function: Callable
        ) -> tuple[bool, list[int], tuple[str, str]]:
            """
Check whether a given pattern overlaps with ring atoms that are
connected to non-ring atoms.

This identifies ring atoms that are part of a substituent group
(i.e. connected to atoms outside the ring via non-ring bonds).

SMARTS pattern used:
- Ring atom with non-ring bond: ``[R]!@[*]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.
pattern_function : Callable
    A function that returns a SMARTS match result for a specific pattern.

Returns
-------
tuple[bool, list[int], tuple[str, str]]
    A tuple containing:
    - A boolean indicating whether the pattern overlaps with ring substituents.
    - A list of atom indices in the intersection.
    - A tuple of SMARTS strings representing the matched patterns.

Examples
--------
>>> Rings.check_pattern_cyclic_substituent("c1ccccc1C(=O)O", MolPatterns.check_carboxylic_acid)
"""

            from mlchem.helper import flatten

            _, b1, c1 = pattern_function(target)

            # Ring connected to something NOT via ring bond
            _, b2, c2 = PatternRecognition.Base.\
                check_smarts_pattern(target, '[R]!@[*]')
            intersection = flatten(
                list(set(b1) & set(b2))
                )
            boolean_response = len(intersection) > 0

            return boolean_response, intersection, (c1, c2)

        @staticmethod
        def check_ortho_substituted_aromatic_r6(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for ortho-substituted aromatic 6-membered rings.

SMARTS pattern used:
- Ortho substitution: ``a1(-[*&!#1&!a&!R])a(-[*&!#1&!a&!R])aaaa1``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> Rings.check_ortho_substituted_aromatic_r6("c1(C)c(C)cccc1")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target,
                                          'a1(-[*&!#1&!a&!R])a(-[*&!#1&!a&!R])aaaa1')

        @staticmethod
        def check_meta_substituted_aromatic_r6(
            target: str | Chem.rdchem.Mol) -> tuple[bool, list[int], str]:
            """
Check for meta-substituted aromatic 6-membered rings.

SMARTS pattern used:
- Meta substitution: ``a1(-[*&!#1&!a&!R])aa(-[*&!#1&!a&!R])aaa1``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> Rings.check_meta_substituted_aromatic_r6("c1(C)cc(C)ccc1")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target,
                                          'a1(-[*&!#1&!a&!R])aa(-[*&!#1&!a&!R])aaa1')
        
        @staticmethod
        def check_para_substituted_aromatic_r6(
            target: str | Chem.rdchem.Mol
             ) -> tuple[bool, list[int], str]:
            """
Check for para-substituted aromatic 6-membered rings.

SMARTS pattern used:
- Para substitution: ``a1(-[*&!#1&!a&!R])aaa(-[*&!#1&!a&!R])aa1``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> Rings.check_para_substituted_aromatic_r6("c1(C)ccc(C)cc1")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target,
                                          'a1(-[*&!#1&!a&!R])aaa(-[*&!#1&!a&!R])aa1')

        @staticmethod
        def check_macrocycle(
            target: str | Chem.rdchem.Mol
        ) -> tuple[bool, list[int], str]:
            """
Check for macrocycles in a molecule.

SMARTS pattern used:
- Macrocycle: ``[r;!r3;!r4;!r5;!r6;!r7]``

This pattern matches atoms in rings larger than 7 members, excluding
common small rings (3-7 atoms).

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether the pattern was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> Rings.check_macrocycle("C1CCCCCCCCCCCC1")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target,
                                          '[r;!r3;!r4;!r5;!r6;!r7]')

        @staticmethod
        def check_ring_size(
            target: str | Chem.rdchem.Mol, size: int
        ) -> tuple[bool, list[int], str]:
            """
Check for rings of a specific size in a molecule.

SMARTS pattern used:
- Ring of size N: ``[rN]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.
size : int
    The size of the ring to detect.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether a ring of the specified size was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> Rings.check_ring_size("C1CCCCC1", 6)
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, f"[r{size}]")

        @staticmethod
        def check_ring_fusion(
            target: str | Chem.rdchem.Mol
        ) -> tuple[bool, list[int], str]:
            """
Check for fused ring systems in a molecule.

SMARTS pattern used:
- Fused rings: ``[#6R2,#6R3,#6R4]``

This pattern matches carbon atoms that are part of two or more rings,
indicating ring fusion.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether fused rings were found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> Rings.check_ring_fusion("c1ccc2ccccc2c1")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#6R2,#6R3,#6R4]')

        @staticmethod
        def check_heterocycle(
            target: str | Chem.rdchem.Mol
        ) -> tuple[bool, list[int], str]:
            """
Check for heterocycles in a molecule.

This method uses RDKit's generic matcher shortcut ``CHC`` to identify
any heterocyclic ring system.

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether a heterocycle was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> Rings.check_heterocycle("c1ccncc1")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '*', ['CHC'])

        @staticmethod
        def check_heterocycle_N(
            target: str | Chem.rdchem.Mol
        ) -> tuple[bool, list[int], str]:
            """
Check for nitrogen-containing heterocycles in a molecule.

SMARTS pattern used:
- Nitrogen in ring: ``[#7R]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether a nitrogen heterocycle was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> Rings.check_heterocycle_N("c1ccncc1")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#7R]')

        @staticmethod
        def check_heterocycle_O(
            target: str | Chem.rdchem.Mol
        ) -> tuple[bool, list[int], str]:
            """
Check for oxygen-containing heterocycles in a molecule.

SMARTS pattern used:
- Oxygen in ring: ``[#8R]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether an oxygen heterocycle was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> Rings.check_heterocycle_O("C1COC1")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#8R]')

        @staticmethod
        def check_heterocycle_S(
            target: str | Chem.rdchem.Mol
        ) -> tuple[bool, list[int], str]:
            """
Check for sulphur-containing heterocycles in a molecule.

SMARTS pattern used:
- Sulphur in ring: ``[#16R]``

Parameters
----------
target : str or rdkit.Chem.rdchem.Mol
    A SMILES string or an RDKit molecule object.

Returns
-------
tuple[bool, list[int], str]
    A tuple containing:
    - A boolean indicating whether a sulphur heterocycle was found.
    - A list of atom indices matching the pattern.
    - A SMARTS string representing the matched pattern.

Examples
--------
>>> Rings.check_heterocycle_S("C1CSC1")
"""

            return PatternRecognition.\
                Base.check_smarts_pattern(target, '[#16R]')
