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
import logging
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
    chemical_space.process(diversity_filter=0.5, collinearity_filter=0.9, standardise=True)
    assert hasattr(chemical_space, 'df_processed')
    assert hasattr(chemical_space, 'scaler')


@pytest.mark.parametrize(
    ('process_kwargs', 'expect_binary_unchanged'),
    [
        ({}, True),
        ({'skip_binary_columns': True}, True),
        ({'skip_binary_columns': False}, False),
    ],
    ids=['default', 'skip-binary', 'scale-binary'],
)
def test_process_skip_binary_columns(
    sample_data,
    process_kwargs,
    expect_binary_unchanged,
):
    descriptors = pd.DataFrame(
        {
            'continuous': [1.0, 2.0, 4.0],
            'binary': [0, 1, 0],
        },
        index=['CCO', 'CCN', 'CCC'],
    )
    chemical_space = ChemicalSpace(
        data=sample_data,
        df_descriptors=descriptors,
    )

    chemical_space.process(
        diversity_filter=None,
        collinearity_filter=None,
        **process_kwargs,
    )

    if expect_binary_unchanged:
        pd.testing.assert_series_equal(
            chemical_space.df_processed['binary'],
            descriptors['binary'],
        )
    else:
        assert not np.allclose(
            chemical_space.df_processed['binary'],
            descriptors['binary'],
        )


def test_missing_descriptors_logs_warning(sample_data, caplog):
    with caplog.at_level(
        logging.WARNING,
        logger='mlchem.chem.visualise.space',
    ):
        ChemicalSpace(data=sample_data)

    assert 'no descriptors available' in caplog.text


def test_process_logs_filter_progress(chemical_space, caplog):
    with caplog.at_level(
        logging.INFO,
        logger='mlchem.chem.visualise.space',
    ):
        chemical_space.process(
            diversity_filter=0.5,
            collinearity_filter=0.9,
            standardise=True,
        )

    assert 'Before filtering:' in caplog.text
    assert 'After diversity filter:' in caplog.text
    assert 'After collinearity filter:' in caplog.text


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

def test_plot(chemical_space, monkeypatch, tmp_path):
    show_mock = MagicMock()
    monkeypatch.setattr("bokeh.plotting.show", show_mock)
    monkeypatch.chdir(tmp_path)

    chemical_space.process(diversity_filter=0, collinearity_filter=0.9, standardise=True)
    df_compressed = pd.DataFrame({
        'DIM_1': [0.1, 0.2, 0.3],
        'DIM_2': [0.4, 0.5, 0.6]
    }, index=['CCO', 'CCN', 'CCC'])
    chemical_space.prepare(df_compressed)
    chemical_space.plot(filename='test_plot',save_html=True)
    show_mock.assert_called_once()
    assert chemical_space.filename == 'test_plot'


@pytest.fixture(params=[False, True], ids=['classes', 'categorical-metadata'])
def plotting_space(request, chemical_space, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("bokeh.io.output_notebook", lambda: None)
    show_mock = MagicMock()
    monkeypatch.setattr("bokeh.plotting.show", show_mock)

    frame = chemical_space.data.iloc[[2, 0, 1]].copy()
    frame['DIM_1'] = [0.3, 0.1, 0.2]
    frame['DIM_2'] = [0.6, 0.4, 0.5]
    frame['MOLFILE'] = ['b.png', 'a1.png', 'a2.png']
    frame['NAME_SHORT'] = frame.NAME
    chemical_space.metadata_exists = request.param
    chemical_space.metadata_categorical = request.param
    if request.param:
        frame['METADATA'] = ['z', 'y', 'x']
        chemical_space.metadata_names = np.unique(frame.METADATA)
    chemical_space.df_compressed = frame
    chemical_space.classnames = np.unique(frame.CLASS)
    return chemical_space, show_mock


def assert_plot_groups(space, show_mock, opacities, sizes):
    from bokeh.models import GlyphRenderer

    show_mock.assert_called_once()
    plot = show_mock.call_args.args[0].children[0]
    renderers = [r for r in plot.renderers if isinstance(r, GlyphRenderer)]
    metadata_groups = space.metadata_names if space.metadata_categorical else [None]
    assert len(renderers) == len(space.classnames) * len(metadata_groups)
    for i, classname in enumerate(space.classnames):
        for j, metadata in enumerate(metadata_groups):
            renderer = renderers[i * len(metadata_groups) + j]
            expected = space.df_compressed[space.df_compressed.CLASS == classname]
            if metadata is not None:
                expected = expected[expected.METADATA == metadata]
                assert renderer.glyph.marker == space.shape_list[j]
            assert list(renderer.data_source.data['SMILES']) == list(expected.SMILES)
            assert list(renderer.data_source.data['DIM_1']) == list(expected.DIM_1)
            assert list(renderer.data_source.data['DIM_2']) == list(expected.DIM_2)
            assert renderer.glyph.fill_alpha == pytest.approx(opacities[i])
            assert renderer.glyph.line_alpha == pytest.approx(opacities[i])
            assert renderer.glyph.size == sizes[i]
            assert renderer.glyph.fill_color == space.colour_list[i]


def test_plot_filtered_dataframe_ignores_directory_counts(plotting_space, tmp_path):
    space, show_mock = plotting_space
    space.df_compressed = space.df_compressed.iloc[:2].copy()
    before = space.df_compressed.copy(deep=True)
    for classname in space.classnames:
        folder = tmp_path / classname
        folder.mkdir()
        for index in range(5):
            (folder / f'{index}.png').touch()

    space.plot()

    assert_plot_groups(space, show_mock, [0.7125, 0.7125], [10, 10])
    pd.testing.assert_frame_equal(space.df_compressed, before)


@pytest.mark.parametrize('marker_size', [None, 17])
def test_plot_default_styles_and_scalar_marker_size(plotting_space, marker_size):
    space, show_mock = plotting_space
    if marker_size is None:
        space.plot()
    else:
        # All existing parameters remain usable positionally.
        space.plot(None, None, '', '', 25, 650, 866, marker_size, False)

    expected_size = 10 if marker_size is None else marker_size
    assert_plot_groups(
        space, show_mock, [0.95 * (1 - 1 / 3), 0.95 * (1 - 1 / 6)],
        [expected_size, expected_size],
    )


@pytest.mark.parametrize(
    'kwargs, opacities, sizes',
    [
        ({'opacity_list': [0.0, 1.0]}, [0.0, 1.0], [10, 10]),
        (
            {'marker_size_list': [4.5, 20], 'marker_size': 17},
            [0.95 * (1 - 1 / 3), 0.95 * (1 - 1 / 6)],
            [4.5, 20],
        ),
        (
            {'opacity_list': [0.2, 0.8], 'marker_size_list': [4.5, 20]},
            [0.2, 0.8], [4.5, 20],
        ),
    ],
)
def test_plot_explicit_class_styles(plotting_space, kwargs, opacities, sizes):
    space, show_mock = plotting_space
    space.plot(**kwargs)
    assert_plot_groups(space, show_mock, opacities, sizes)


@pytest.mark.parametrize('explicit', [False, True])
def test_plot_single_class_zero_remainder(plotting_space, explicit):
    space, show_mock = plotting_space
    space.df_compressed = space.df_compressed[
        space.df_compressed.CLASS == 'CLASS_A'
    ].copy()
    space.classnames = np.array(['CLASS_A'])
    kwargs = {'opacity_list': [0.8], 'marker_size_list': [6]} if explicit else {}

    space.plot(**kwargs)

    assert_plot_groups(
        space, show_mock, [0.8 if explicit else 0.475], [6 if explicit else 10],
    )


@pytest.mark.parametrize('name', ['opacity_list', 'marker_size_list'])
@pytest.mark.parametrize('values', [[], [0.5], [0.5, 0.5, 0.5]])
def test_plot_rejects_style_list_length(plotting_space, name, values):
    space, show_mock = plotting_space
    with pytest.raises(ValueError, match=name):
        space.plot(**{name: values})
    show_mock.assert_not_called()


@pytest.mark.parametrize(
    'name, value',
    [
        ('opacity_list', -0.1), ('opacity_list', 1.1),
        ('opacity_list', float('nan')), ('opacity_list', float('inf')),
        ('opacity_list', '0.5'), ('opacity_list', None), ('opacity_list', True),
        ('marker_size_list', 0), ('marker_size_list', -1),
        ('marker_size_list', float('nan')), ('marker_size_list', float('inf')),
        ('marker_size_list', '10'), ('marker_size_list', None),
        ('marker_size_list', True),
    ],
)
def test_plot_rejects_style_values(plotting_space, name, value):
    space, show_mock = plotting_space
    with pytest.raises(ValueError, match=name):
        space.plot(**{name: [0.5, value]})
    show_mock.assert_not_called()


def test_process_invalid_diversity_filter_raises(chemical_space):
    with pytest.raises(ValueError):
        chemical_space.process(diversity_filter=-0.1, collinearity_filter=0.9)

    with pytest.raises(ValueError):
        chemical_space.process(diversity_filter=1.0, collinearity_filter=0.9)


def test_process_invalid_collinearity_filter_raises(chemical_space):
    with pytest.raises(ValueError):
        chemical_space.process(diversity_filter=0.0, collinearity_filter=-0.1)

    with pytest.raises(ValueError):
        chemical_space.process(diversity_filter=0.0, collinearity_filter=1.1)


def test_prepare_rejects_wrong_columns(chemical_space):
    chemical_space.process(diversity_filter=0, collinearity_filter=0.9, standardise=True)
    df_compressed = pd.DataFrame(
        {
            'DIM_1': [0.1, 0.2, 0.3],
            'WRONG': [0.4, 0.5, 0.6],
        },
        index=['CCO', 'CCN', 'CCC'],
    )
    with pytest.raises(ValueError):
        chemical_space.prepare(df_compressed)


def test_prepare_rejects_non_smiles_index(chemical_space):
    chemical_space.process(diversity_filter=0, collinearity_filter=0.9, standardise=True)
    df_compressed = pd.DataFrame(
        {
            'DIM_1': [0.1, 0.2, 0.3],
            'DIM_2': [0.4, 0.5, 0.6],
        },
        index=[10, 20, 30],
    )
    with pytest.raises(ValueError):
        chemical_space.prepare(df_compressed)

if __name__ == "__main__":
    pytest.main()