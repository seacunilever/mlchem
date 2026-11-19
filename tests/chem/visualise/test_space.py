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
import os
from pathlib import Path
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


@pytest.mark.parametrize('label_type', ['string', 'integer'])
@pytest.mark.parametrize('already_sorted', [False, True])
@pytest.mark.parametrize('categorical_metadata', [False, True])
def test_prepare_and_plot_keep_structure_rows_aligned(
    chemical_space,
    monkeypatch,
    tmp_path,
    label_type,
    already_sorted,
    categorical_metadata,
):
    import shutil
    from bokeh.models import GlyphRenderer, HoverTool

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("bokeh.io.output_notebook", lambda: None)
    show_mock = MagicMock()
    monkeypatch.setattr("bokeh.plotting.show", show_mock)

    smiles = ['NCCO', 'CCO', 'CCN', 'CCOC']
    names = ['name_ncco', 'name_cco', 'name_ccn', 'name_ccoc']
    classes = ['B', 'A', 'B', 'A'] if label_type == 'string' else [2, 1, 2, 1]
    metadata = ['y', 'x', 'x', 'y']
    order = [1, 3, 2, 0] if already_sorted else [0, 1, 2, 3]
    data = {
        'SMILES': [smiles[i] for i in order],
        'NAME': [names[i] for i in order],
        'CLASS': [classes[i] for i in order],
    }
    if categorical_metadata:
        data['METADATA'] = [metadata[i] for i in order]
    chemical_space.data = pd.DataFrame(data)
    chemical_space.metadata_exists = categorical_metadata
    chemical_space.metadata_categorical = categorical_metadata

    processed_smiles = list(reversed(chemical_space.data.SMILES.tolist()))
    chemical_space.df_processed = pd.DataFrame(
        {'descriptor': range(len(processed_smiles))},
        index=pd.Index(processed_smiles, name='SMILES'),
    )
    compressed_smiles = chemical_space.data.SMILES.tolist()
    coordinates = {
        value: (i + 0.1, i + 0.2)
        for i, value in enumerate(smiles)
    }
    compressed = pd.DataFrame(
        {
            'DIM_1': [coordinates[value][0] for value in compressed_smiles],
            'DIM_2': [coordinates[value][1] for value in compressed_smiles],
        },
        index=pd.Index(compressed_smiles, name='SMILES'),
    )
    original_rows = compressed.index.tolist()
    original_dim_1 = compressed.DIM_1.tolist()
    original_dim_2 = compressed.DIM_2.tolist()

    def create_test_structures(df, structure_column_name, folder_name):
        folder = Path(folder_name)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        for index, value in enumerate(df[structure_column_name]):
            image_path = folder / f'{index}.png'
            image_path.write_text(value, encoding='utf-8')
            timestamp = 1_000_000_000 + len(df) - index
            os.utime(image_path, (timestamp, timestamp))

    monkeypatch.setattr(
        'mlchem.helper.create_structure_files', create_test_structures
    )
    chemical_space.prepare(compressed)
    assert chemical_space.df_compressed is compressed
    assert compressed.SMILES.tolist() == original_rows
    assert compressed.DIM_1.tolist() == original_dim_1
    assert compressed.DIM_2.tolist() == original_dim_2

    expected = {
        smiles[i]: {
            'NAME': names[i],
            'CLASS': classes[i],
            'METADATA': metadata[i],
            'DIM_1': coordinates[smiles[i]][0],
            'DIM_2': coordinates[smiles[i]][1],
        }
        for i in range(len(smiles))
    }
    prepared = chemical_space.df_compressed.set_index('SMILES')
    for value, row in expected.items():
        actual = prepared.loc[value]
        assert actual.NAME == row['NAME']
        assert actual.CLASS == row['CLASS']
        assert actual.DIM_1 == row['DIM_1']
        assert actual.DIM_2 == row['DIM_2']
        assert Path(actual.MOLFILE).read_text(encoding='utf-8') == value
        if categorical_metadata:
            assert actual.METADATA == row['METADATA']

    chemical_space.plot()

    layout = show_mock.call_args.args[0]
    plot = layout.children[0]
    renderers = [
        renderer for renderer in plot.renderers
        if isinstance(renderer, GlyphRenderer)
    ]
    plotted_smiles = set()
    for renderer in renderers:
        source = renderer.data_source.data
        for i, value in enumerate(source['SMILES']):
            assert source['NAME'][i] == expected[value]['NAME']
            assert source['CLASS'][i] == expected[value]['CLASS']
            assert source['DIM_1'][i] == expected[value]['DIM_1']
            assert source['DIM_2'][i] == expected[value]['DIM_2']
            assert Path(source['MOLFILE'][i]).read_text(
                encoding='utf-8'
            ) == value
            plotted_smiles.add(value)
    assert plotted_smiles == set(smiles)

    hover = next(tool for tool in plot.tools if isinstance(tool, HoverTool))
    assert '@MOLFILE' in hover.tooltips
    assert '@NAME_SHORT' in hover.tooltips


def test_prepare_rejects_ambiguous_smiles_before_generating_structures(
    chemical_space,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    chemical_space.data = pd.DataFrame({
        'SMILES': ['CCO', 'CCO'],
        'NAME': ['first', 'second'],
        'CLASS': ['A', 'A'],
    })
    chemical_space.df_processed = pd.DataFrame(
        {'descriptor': [1]}, index=pd.Index(['CCO'], name='SMILES')
    )
    compressed = pd.DataFrame(
        {'DIM_1': [0.1], 'DIM_2': [0.2]},
        index=pd.Index(['CCO'], name='SMILES'),
    )
    create_mock = MagicMock()
    monkeypatch.setattr('mlchem.helper.create_structure_files', create_mock)

    with pytest.raises(ValueError, match='unique SMILES keys'):
        chemical_space.prepare(compressed)
    create_mock.assert_not_called()


def test_prepare_rejects_compressed_smiles_without_processed_key(
    chemical_space,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    chemical_space.df_processed = pd.DataFrame(
        {'descriptor': [1]}, index=pd.Index(['CCO'], name='SMILES')
    )
    compressed = pd.DataFrame(
        {'DIM_1': [0.1], 'DIM_2': [0.2]},
        index=pd.Index(['CCC'], name='SMILES'),
    )
    create_mock = MagicMock()
    monkeypatch.setattr('mlchem.helper.create_structure_files', create_mock)

    with pytest.raises(ValueError, match='df_processed'):
        chemical_space.prepare(compressed)
    create_mock.assert_not_called()


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