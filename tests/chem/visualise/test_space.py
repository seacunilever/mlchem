import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from mlchem.chem.visualise.space import ChemicalSpace
from typing import Iterable

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'SMILES': ['CCO', 'CCN', 'CCC'],
        'NAME': ['ethanol', 'ethylamine', 'propane'],
        'CLASS': ['CLASS_A', 'CLASS_A', 'CLASS_B']
    })
    return data

@pytest.fixture
def sample_descriptors():
    descriptors = pd.DataFrame({
        'SMILES': ['CCO', 'CCN', 'CCC'],
        'desc1': [1.0, 2.0, 3.0],
        'desc2': [4.0, 5.0, 6.0]
    }).set_index('SMILES')
    return descriptors

@pytest.fixture
def chemical_space(sample_data, sample_descriptors):
    return ChemicalSpace(data=sample_data, df_descriptors=sample_descriptors)


def test_process(chemical_space):
    chemical_space.process(diversity_filter=0, collinearity_filter=0.9, standardise=True)
    assert hasattr(chemical_space, 'df_processed')
    assert hasattr(chemical_space, 'scaler')

@pytest.fixture
def df_processed(chemical_space):
    chemical_space.process(diversity_filter=0, collinearity_filter=0.9, standardise=True)
    return chemical_space.df_processed


@pytest.fixture
def df_compressed(chemical_space):
    chemical_space.process(diversity_filter=0, collinearity_filter=0.9, standardise=True)
    df_compressed = pd.DataFrame({
        'DIM_1': [0.1, 0.2, 0.3],
        'DIM_2': [0.4, 0.5, 0.6]
    }, index=['CCO', 'CCN', 'CCC'])
    chemical_space.prepare(df_compressed)
    return df_compressed



def test_prepare(chemical_space):
    chemical_space.process(diversity_filter=0, collinearity_filter=0.9, standardise=True)
    df_compressed = pd.DataFrame({
        'DIM_1': [0.1, 0.2, 0.3],
        'DIM_2': [0.4, 0.5, 0.6]
    }, index=['CCO', 'CCN', 'CCC'])
    print("Before prepare:")
    print(df_compressed)
    chemical_space.prepare(df_compressed)
    print("After prepare:")
    print(chemical_space.df_compressed)
    assert 'MOLFILE' in chemical_space.df_compressed.columns
    assert 'NAME' in chemical_space.df_compressed.columns
    assert 'CLASS' in chemical_space.df_compressed.columns

def test_update_bokeh_options(chemical_space):
    new_options = {'title_location': 'below', 'title_fontsize': '20px'}
    chemical_space.update_bokeh_options(**new_options)
    assert chemical_space.bokeh_dictionary['title_location'] == 'below'
    assert chemical_space.bokeh_dictionary['title_fontsize'] == '20px'

def test_reset_bokeh_options(chemical_space):
    chemical_space.update_bokeh_options(title_location='below')
    chemical_space.reset_bokeh_options()
    assert chemical_space.bokeh_dictionary['title_location'] == 'above'

def test_reset_bokeh_tooltips(chemical_space):
    chemical_space.bokeh_tooltips = {'tooltip': 'value'}
    chemical_space.reset_bokeh_tooltips()
    assert chemical_space.bokeh_tooltips != {'tooltip': 'value'}

def test_plot(chemical_space, monkeypatch):
    show_mock = MagicMock()
    monkeypatch.setattr("bokeh.plotting.show", show_mock)

    chemical_space.process(diversity_filter=0, collinearity_filter=0.9, standardise=True)
    df_compressed = pd.DataFrame({
        'DIM_1': [0.1, 0.2, 0.3],
        'DIM_2': [0.4, 0.5, 0.6]
    }, index=['CCO', 'CCN', 'CCC'])
    chemical_space.prepare(df_compressed)
    chemical_space.plot(filename='test_plot',save_html=True)
    show_mock.assert_called_once()
    assert chemical_space.filename == 'test_plot'

if __name__ == "__main__":
    pytest.main()