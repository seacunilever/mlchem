import pytest
from mlchem.importables import (
    metal_list, chemical_dictionary, colour_dictionary,
    chemotype_dictionary, bokeh_dictionary, bokeh_tooltips,
    interpretable_descriptors_rdkit, interpretable_descriptors_mordred,
    similarity_metric_dictionary
)

def test_metal_list():
    assert isinstance(metal_list, list)
    assert 'Li' in metal_list
    assert 'U' in metal_list
    assert 'Cn' in metal_list
    assert 'Fe' in metal_list
    assert 'Au' in metal_list

def test_chemical_dictionary():
    assert isinstance(chemical_dictionary, dict)
    assert '[#B-1]' in chemical_dictionary
    assert '[#C]' in chemical_dictionary
    assert '[S]' in chemical_dictionary
    assert '[Br]' in chemical_dictionary
    assert '[Cl]' in chemical_dictionary

def test_colour_dictionary():
    assert isinstance(colour_dictionary, dict)
    assert 'red' in colour_dictionary
    assert 'green' in colour_dictionary
    assert 'blue' in colour_dictionary
    assert 'yellow' in colour_dictionary
    assert 'purple' in colour_dictionary

def test_chemotype_dictionary():
    assert isinstance(chemotype_dictionary, dict)
    assert 'Carbon' in chemotype_dictionary
    assert 'Nitrogen' in chemotype_dictionary
    assert 'Sulphur' in chemotype_dictionary
    assert 'Oxygen' in chemotype_dictionary
    assert 'Fluorine' in chemotype_dictionary
    assert callable(chemotype_dictionary['Carbon'][0])
    assert isinstance(chemotype_dictionary['Carbon'][1], dict)

def test_bokeh_dictionary():
    assert isinstance(bokeh_dictionary, dict)
    assert 'title_location' in bokeh_dictionary
    assert 'legend_location' in bokeh_dictionary
    assert 'xaxis_label' in bokeh_dictionary
    assert 'yaxis_label' in bokeh_dictionary
    assert 'axis_minor_tick_in' in bokeh_dictionary

def test_bokeh_tooltips():
    assert isinstance(bokeh_tooltips, str)
    assert '@MOLFILE' in bokeh_tooltips
    assert '@NAME_SHORT' in bokeh_tooltips
    assert '@CLASS' in bokeh_tooltips
    assert '@index' in bokeh_tooltips
    assert '@METADATA' in bokeh_tooltips

def test_interpretable_descriptors_rdkit():
    assert isinstance(interpretable_descriptors_rdkit, list)
    assert 'MolWt' in interpretable_descriptors_rdkit
    assert 'NumHAcceptors' in interpretable_descriptors_rdkit
    assert 'MolLogP' in interpretable_descriptors_rdkit
    assert 'HeavyAtomCount' in interpretable_descriptors_rdkit
    assert 'RingCount' in interpretable_descriptors_rdkit

def test_interpretable_descriptors_mordred():
    assert isinstance(interpretable_descriptors_mordred, list)
    assert 'nAcid' in interpretable_descriptors_mordred
    assert 'nAtom' in interpretable_descriptors_mordred
    assert 'SLogP' in interpretable_descriptors_mordred
    assert 'nRing' in interpretable_descriptors_mordred
    assert 'TPSA' in interpretable_descriptors_mordred

def test_similarity_metric_dictionary():
    assert isinstance(similarity_metric_dictionary, dict)
    assert 'Tanimoto' in similarity_metric_dictionary
    assert 'Dice' in similarity_metric_dictionary
    assert 'Cosine' in similarity_metric_dictionary
    assert 'Sokal' in similarity_metric_dictionary
    assert 'Russel' in similarity_metric_dictionary
    assert callable(similarity_metric_dictionary['Tanimoto'])
    assert callable(similarity_metric_dictionary['Dice'])
    assert callable(similarity_metric_dictionary['Cosine'])

if __name__ == "__main__":
    pytest.main()