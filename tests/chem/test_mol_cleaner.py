import pytest
from unittest.mock import patch

from mlchem.chem.manipulation import MolCleaner


def test_initialise_smiles_accepts_valid_and_rejects_invalid():
    cleaner = MolCleaner(["CCO", "invalid"], id_list=[10, 11])

    cleaner.initialise_smiles(isomeric=False, canonical=True, kekulise=True)

    assert cleaner.ids == [10]
    assert cleaner.smiles == ["CCO"]
    assert cleaner.df_rejected.shape[0] >= 1
    assert 11 in cleaner.df_rejected["id"].tolist()


def test_initialise_smiles_requires_at_least_one_mode_enabled():
    cleaner = MolCleaner(["CCO"])

    with pytest.raises(TypeError, match="At least one argument must be True"):
        cleaner.initialise_smiles(isomeric=False, canonical=False, kekulise=False)


def test_desalt_smiles_largest_keeps_largest_fragment():
    cleaner = MolCleaner(["CCO.[Na+]", "CC.Cl"])
    cleaner.initialise_smiles(canonical=True, kekulise=False)

    cleaner.desalt_smiles(method="largest", dehydrate=False)

    assert cleaner.smiles == ["CCO", "CC"]
    assert cleaner.df_rejected.empty


def test_desalt_smiles_rdkit_runs_with_state_overrides():
    cleaner = MolCleaner(["CCO.[Na+]"])
    cleaner.initialise_smiles(canonical=True, kekulise=False)

    cleaner.desalt_smiles(
        method="rdkit",
        dehydrate=False,
        isomeric=False,
        canonical=True,
        kekulise=False,
    )

    assert cleaner.ids == [0]
    assert cleaner.smiles == ["CCO"]


def test_neutralise_smiles_accepts_and_rejects():
    cleaner = MolCleaner(["CC(O)[O-]"])
    cleaner.initialise_smiles(canonical=True, kekulise=False)
    cleaner.smiles = ["CC(O)[O-]", "invalid"]
    cleaner.ids = [0, 1]

    cleaner.neutralise_smiles()

    assert cleaner.smiles == ["CC(O)O"]
    assert cleaner.ids == [0]
    assert cleaner.df_rejected.shape[0] >= 1


def test_remove_carbon_ions_filters_charged_carbon():
    cleaner = MolCleaner(["CCO", "[CH3-]"])
    cleaner.initialise_smiles(canonical=True, kekulise=False)

    cleaner.remove_carbon_ions()

    assert cleaner.smiles == ["CCO"]
    assert cleaner.ids == [0]


def test_remove_inorganics_filters_non_organic_entries():
    cleaner = MolCleaner(["CCO", "[Na+]"])
    cleaner.initialise_smiles(canonical=True, kekulise=False)

    cleaner.remove_inorganics()

    assert cleaner.smiles == ["CCO"]
    assert cleaner.ids == [0]


def test_remove_organometallics_rejects_non_salt_metal_entries():
    cleaner = MolCleaner(["CC[Na]", "CCO", "[Na+].[Cl-]"])
    cleaner.initialise_smiles(canonical=True, kekulise=False)

    with patch(
        "mlchem.chem.manipulation.PatternRecognition.Base.has_metal_salt",
        side_effect=[False, True],
    ):
        cleaner.remove_organometallics()

    assert "CC[Na]" not in cleaner.smiles
    assert "[Cl-].[Na+]" in cleaner.smiles
    assert "CCO" in cleaner.smiles


def test_remove_mixtures_rejects_non_metal_binary_mixtures():
    cleaner = MolCleaner(["CC.O", "CC.[Na+]", "CCO"])
    cleaner.initialise_smiles(canonical=True, kekulise=False)
    cleaner.metal_list = ["Na"]

    cleaner.remove_mixtures()

    assert "CC.O" not in cleaner.smiles
    assert "CC.[Na+]" in cleaner.smiles
    assert "CCO" in cleaner.smiles