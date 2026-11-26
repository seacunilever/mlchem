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
        folder.mkdir(parents=True, exist_ok=True)
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
        assert Path(actual.MOLFILE).parent == Path(
            chemical_space.class_structure_folders[row['CLASS']]
        )
        if label_type == 'string':
            assert Path(actual.MOLFILE).parent.name == str(row['CLASS'])
        else:
            assert not Path(actual.MOLFILE).parent.name.startswith('class_')
        if categorical_metadata:
            assert actual.METADATA == row['METADATA']

    chemical_space.plot()

    layout = show_mock.call_args.args[0]
    plot = layout.children[0]
    renderers = [
        renderer for renderer in plot.renderers
        if isinstance(renderer, GlyphRenderer) and renderer.visible
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


def test_prepare_uses_safe_label_derived_structure_folders(
    chemical_space,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    labels = ['EVAL', 'eval', '../TRAIN']
    chemical_space.data = pd.DataFrame({
        'SMILES': ['CCO', 'CCN', 'CCC'],
        'NAME': ['ethanol', 'ethylamine', 'propane'],
        'CLASS': labels,
    })
    chemical_space.df_processed = pd.DataFrame(
        {'descriptor': [1, 2, 3]},
        index=pd.Index(['CCO', 'CCN', 'CCC'], name='SMILES'),
    )
    compressed = pd.DataFrame(
        {'DIM_1': [0.1, 0.2, 0.3], 'DIM_2': [0.4, 0.5, 0.6]},
        index=pd.Index(['CCO', 'CCN', 'CCC'], name='SMILES'),
    )

    def create_test_structures(df, structure_column_name, folder_name):
        folder = Path(folder_name)
        folder.mkdir(parents=True, exist_ok=True)
        for index, value in enumerate(df[structure_column_name]):
            (folder / f'{index}.png').write_text(value, encoding='utf-8')

    monkeypatch.setattr(
        'mlchem.helper.create_structure_files', create_test_structures
    )
    chemical_space.prepare(compressed)

    folders = list(chemical_space.class_structure_folders.values())
    assert len(set(folders)) == len(labels)
    structure_root = (tmp_path / '.mlchem_structures').resolve()
    assert all(Path(folder).resolve().is_relative_to(structure_root)
               for folder in folders)
    folder_names = [Path(folder).name for folder in folders]
    assert all(name.isascii() and len(name) <= 255 for name in folder_names)
    assert not any(
        any(character in '<>:"/\\|?*' for character in name)
        or name.endswith(('.', ' '))
        for name in folder_names
    )
    assert len({name.casefold() for name in folder_names}) == len(folder_names)
    assert 'EVAL' not in [Path(folder).name for folder in folders]
    assert all(not Path(folder).name.startswith('class_') for folder in folders)
    assert not (tmp_path / 'TRAIN').exists()

    prepared = chemical_space.df_compressed.set_index('SMILES')
    for smiles, class_label in zip(['CCO', 'CCN', 'CCC'], labels):
        molfile = prepared.loc[smiles, 'MOLFILE']
        assert Path(molfile).parent == Path(
            chemical_space.class_structure_folders[class_label]
        )
        assert Path(molfile).read_text(encoding='utf-8') == smiles


def test_prepare_legacy_metadata_categories_use_only_compressed_rows(
    monkeypatch, tmp_path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('bokeh.io.output_notebook', lambda: None)
    show = MagicMock()
    monkeypatch.setattr('bokeh.plotting.show', show)
    monkeypatch.setattr('mlchem.helper.create_structure_files', MagicMock())
    smiles = ['C', 'CC'] + ['C' * length for length in range(3, 12)]
    data = pd.DataFrame({
        'SMILES': smiles,
        'NAME': smiles,
        'CLASS': ['A'] * len(smiles),
        'METADATA': [f'set_{i}' for i in range(len(smiles) - 1)] + [np.nan],
    })
    descriptors = pd.DataFrame({'descriptor': range(len(smiles))}, index=smiles)
    space = ChemicalSpace(
        data, df_descriptors=descriptors,
        metadata_exists=True, metadata_categorical=True,
    )
    space.df_processed = descriptors
    compressed = pd.DataFrame(
        {'DIM_1': [0.0, 1.0], 'DIM_2': [1.0, 0.0]}, index=smiles[:2],
    )

    space.prepare(compressed)
    assert space.metadata_names.tolist() == ['set_0', 'set_1']
    space.plot()
    assert show.call_count == 1


def test_prepare_legacy_metadata_keeps_unsortable_categories_in_input_order(
    monkeypatch, tmp_path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('mlchem.helper.create_structure_files', MagicMock())
    monkeypatch.setattr('bokeh.io.output_notebook', lambda: None)
    show = MagicMock()
    monkeypatch.setattr('bokeh.plotting.show', show)
    data = pd.DataFrame({
        'SMILES': ['C', 'CC'],
        'NAME': ['methane', 'ethane'],
        'CLASS': ['A', 'A'],
        'METADATA': ['train', 1],
    })
    descriptors = pd.DataFrame({'descriptor': [1, 2]}, index=data.SMILES)
    space = ChemicalSpace(
        data, df_descriptors=descriptors,
        metadata_exists=True, metadata_categorical=True,
    )
    space.df_processed = descriptors

    space.prepare(pd.DataFrame(
        {'DIM_1': [0.0, 1.0], 'DIM_2': [1.0, 0.0]}, index=data.SMILES,
    ))
    assert space.metadata_names.tolist() == ['train', 1]
    space.plot()
    renderers = [
        renderer for renderer in show.call_args.args[0].children[0].renderers
        if renderer.visible
        and hasattr(renderer, 'glyph')
        and hasattr(renderer.glyph, 'marker')
    ]
    assert [renderer.glyph.marker for renderer in renderers] == ['circle', 'triangle']


def test_legacy_shape_encoding_rejects_missing_metadata_but_explicit_shape_works(
    encoding_space,
):
    space, show = encoding_space
    space.df_compressed.loc[space.df_compressed.index[0], 'METADATA'] = None
    with pytest.raises(ValueError, match='METADATA.*missing values'):
        space.plot()
    show.assert_not_called()

    space.plot(
        shape_by='SCORE',
        shape_map={0.1: 'circle', 0.9: 'triangle'},
    )
    assert show.call_count == 1


def test_plot_cached_molfiles_does_not_generate_structure_root(
    chemical_space,
    monkeypatch,
    tmp_path,
):
    from bokeh.models import GlyphRenderer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('bokeh.io.output_notebook', lambda: None)
    show_mock = MagicMock()
    monkeypatch.setattr('bokeh.plotting.show', show_mock)

    rows = []
    for index, (smiles, label) in enumerate(
        [('CCO', 'EVAL'), ('CCN', 'TEST'), ('CCC', 'TRAIN')]
    ):
        folder = tmp_path / label
        folder.mkdir()
        molfile = folder / f'{smiles}.png'
        molfile.write_text(smiles, encoding='utf-8')
        rows.append({
            'DIM_1': index + 0.1,
            'DIM_2': index + 0.2,
            'SMILES': smiles,
            'NAME': f'name_{smiles}',
            'NAME_SHORT': f'name_{smiles}',
            'CLASS': label,
            'MOLFILE': str(molfile),
        })

    cached = pd.DataFrame(rows)
    chemical_space.df_compressed = cached
    chemical_space.classnames = np.array(['EVAL', 'TEST', 'TRAIN'])
    chemical_space.plot()

    assert not (tmp_path / '.mlchem_structures').exists()
    layout = show_mock.call_args.args[0]
    plot = layout.children[0]
    rendered_paths = []
    for renderer in plot.renderers:
        if isinstance(renderer, GlyphRenderer) and renderer.visible:
            rendered_paths.extend(renderer.data_source.data['MOLFILE'])
    assert rendered_paths == cached.MOLFILE.tolist()
    assert all(Path(path).is_file() for path in rendered_paths)


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


def test_update_bokeh_options_rejects_unknown_key_without_mutation(chemical_space):
    before = chemical_space.bokeh_dictionary.copy()
    with pytest.raises(ValueError, match='Unknown Bokeh option'):
        chemical_space.update_bokeh_options(legend_lcation='top_right')
    assert chemical_space.bokeh_dictionary == before


@pytest.mark.parametrize('bad_value', [None, '', 20])
def test_update_bokeh_options_validates_title_fontsize(chemical_space, bad_value):
    before = chemical_space.bokeh_dictionary.copy()
    with pytest.raises(ValueError, match='title_fontsize'):
        chemical_space.update_bokeh_options(title_fontsize=bad_value)
    assert chemical_space.bokeh_dictionary == before


def test_plot_uses_updated_title_fontsize_and_explicit_override(plotting_space):
    space, show = plotting_space
    space.update_bokeh_options(title_fontsize='20px')
    space.plot()
    plot = show.call_args.args[0].children[0]
    assert plot.title.text_font_size == '20px'

    space.plot(title_fontsize=18)
    plot = show.call_args.args[0].children[0]
    assert plot.title.text_font_size == '18px'

    space.plot(title_fontsize=np.int64(19))
    plot = show.call_args.args[0].children[0]
    assert plot.title.text_font_size == '19px'

    space.plot(title_fontsize='1.5em')
    plot = show.call_args.args[0].children[0]
    assert plot.title.text_font_size == '1.5em'

    space.reset_bokeh_options()
    space.plot()
    plot = show.call_args.args[0].children[0]
    assert plot.title.text_font_size == '25px'


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
    renderers = [
        r for r in plot.renderers
        if isinstance(r, GlyphRenderer) and r.visible
    ]
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


@pytest.fixture
def encoding_space(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('bokeh.io.output_notebook', lambda: None)
    show = MagicMock()
    monkeypatch.setattr('bokeh.plotting.show', show)
    monkeypatch.setattr('mlchem.helper.create_structure_files', MagicMock())
    data = pd.DataFrame({
        'SMILES': ['CCO', 'CCN', 'CCC', 'CCOC'],
        'NAME': ['ethanol', 'ethylamine', 'propane', 'ether'],
        'CLASS': ['score[0]', 'score[0]', 'score[1]', 'score[1]'],
        'METADATA': ['train', 'test', 'train', 'test'],
        'SCORE': [0.1, 0.9, 0.9, 0.1],
    })
    space = ChemicalSpace(
        data=data, df_descriptors=data.set_index('SMILES')[['SCORE']],
        metadata_exists=True, metadata_categorical=True,
    )
    space.df_processed = space.df_descriptors
    compressed = pd.DataFrame({
        'DIM_1': [1., 2., 3., 4.], 'DIM_2': [2., 3., 4., 5.],
    }, index=['CCOC', 'CCN', 'CCO', 'CCC'])
    space.prepare(compressed)
    expected = data.set_index('SMILES').loc[compressed.SMILES]
    np.testing.assert_array_equal(compressed.SCORE, expected.SCORE)
    assert set(space.metadata_names) == {'train', 'test'}
    return space, show


@pytest.mark.parametrize('rotation', range(3))
def test_plot_independent_encodings_use_isolated_and_text_only_legend_keys(
    encoding_space, rotation,
):
    from bokeh.embed import file_html
    from bokeh.models import GlyphRenderer, HoverTool
    from bokeh.resources import INLINE

    space, show = encoding_space
    columns = ['CLASS', 'METADATA', 'SCORE']
    styles = {
        'colour': ['#1565c0', '#ef6c00'],
        'shape': ['circle', 'triangle'],
        'opacity': [0.35, 0.9],
        'marker_size': [7, 16],
    }
    kwargs = {}
    for i, (encoding, values) in enumerate(styles.items()):
        column = columns[(i + rotation) % len(columns)]
        kwargs[f'{encoding}_by'] = column
        kwargs[f'{encoding}_map'] = dict(zip(sorted(space.df_compressed[column].unique()), values))
    before = space.df_compressed.copy(deep=True)
    space.plot(title='Independent encodings', **kwargs)
    plot = show.call_args.args[0].children[0]
    assert (plot.width, plot.height) == (866, 650)
    renderers = [
        r for r in plot.renderers
        if isinstance(r, GlyphRenderer) and r.visible
    ]
    plotted = []
    for renderer in renderers:
        source = renderer.data_source.data
        plotted.extend(source['SMILES'])
        for encoding, property_name in [
            ('colour', 'fill_color'), ('shape', 'marker'),
            ('opacity', 'fill_alpha'), ('marker_size', 'size'),
        ]:
            column = kwargs[f'{encoding}_by']
            mapping = kwargs[f'{encoding}_map']
            assert all(getattr(renderer.glyph, property_name) == mapping[value] for value in source[column])
        assert renderer.glyph.line_alpha == renderer.glyph.fill_alpha
    assert sorted(plotted) == sorted(before.SMILES)
    keys = {item.label.value: item for item in plot.legend[0].items}
    assert plot.legend[0].ncols == 2
    for encoding, label in [
        ('colour', 'Colour'), ('shape', 'Shape'), ('opacity', 'Opacity'), ('marker_size', 'Size'),
    ]:
        column = kwargs[f'{encoding}_by']
        for value in space.df_compressed[column].unique():
            key = f'{label} ({column}): {value}'
            if encoding == 'opacity':
                key += f' = {kwargs["opacity_map"][value]}'
            elif encoding == 'marker_size':
                key += f' = {kwargs["marker_size_map"][value]} px'
            item = keys[key]
            if encoding in ('opacity', 'marker_size'):
                assert item.renderers == []
                continue
            assert len(item.renderers) == 1
            renderer = item.renderers[0]
            assert renderer not in renderers
            assert renderer.visible is False
            assert list(renderer.data_source.data) == ['x', 'y']
            assert item.index == 0
            assert renderer.glyph.size == space.bokeh_dictionary['legend_proxy_size']
            assert renderer.glyph.fill_alpha == space.bokeh_dictionary['legend_proxy_alpha']
            if encoding == 'colour':
                assert renderer.glyph.marker == 'circle'
                assert renderer.glyph.fill_color == kwargs['colour_map'][value]
                assert renderer.glyph.line_color == 'grey'
            else:
                assert renderer.glyph.fill_color == 'grey'
                assert renderer.glyph.marker == kwargs['shape_map'][value]
    hover = next(tool for tool in plot.tools if isinstance(tool, HoverTool))
    assert set(hover.renderers) == set(renderers)
    assert '@MOLFILE' in hover.tooltips and '@NAME_SHORT' in hover.tooltips
    html = file_html(show.call_args.args[0], INLINE, 'Independent encodings')
    assert 'Independent encodings' in html and 'Scatter' in html
    assert 'score[0]' in html and 'MOLFILE' in html
    if rotation == 0:
        assert 'Opacity (SCORE): 0.1 = 0.35' in html
        assert 'Size (CLASS): score[0] = 7 px' in html
    pd.testing.assert_frame_equal(space.df_compressed, before)


def test_plot_continuous_colour_and_html_export(encoding_space, monkeypatch, tmp_path):
    from bokeh.io import save
    from bokeh.models import ColorBar, GlyphRenderer
    from bokeh.resources import INLINE

    space, _ = encoding_space
    saved = []

    def save_plot(layout):
        saved.append(layout.children[0])
        save(layout, resources=INLINE)

    monkeypatch.setattr('bokeh.plotting.show', save_plot)
    space.plot(
        colour_by='SCORE', colour_continuous=True, colour_palette=['navy', 'gold'],
        shape_by='METADATA', shape_map={'train': 'square', 'test': 'triangle'},
        opacity_by='CLASS', opacity_map={'score[0]': 0.5, 'score[1]': 1.0},
        marker_size_by='METADATA', marker_size_map={'train': 8, 'test': 14},
        filename='continuous', save_html=True,
    )
    plot = saved[0]
    bar = next(item for item in plot.right if isinstance(item, ColorBar))
    assert bar.title == 'SCORE'
    assert (bar.color_mapper.low, bar.color_mapper.high) == (0.1, 0.9)
    assert bar.color_mapper.palette == ['navy', 'gold']
    for renderer in plot.renderers:
        if isinstance(renderer, GlyphRenderer) and renderer.visible:
            assert renderer.glyph.fill_color.field == 'SCORE'
            assert renderer.glyph.fill_color.transform is bar.color_mapper
    assert not any(item.label.value.startswith('Colour') for item in plot.legend[0].items)
    html = (tmp_path / 'continuous.html').read_text(encoding='utf-8')
    assert 'ColorBar' in html and 'LinearColorMapper' in html and 'MOLFILE' in html


def test_legend_proxy_options_are_customizable_outside_plot(encoding_space):
    from bokeh.models import GlyphRenderer

    space, show = encoding_space
    space.update_bokeh_options(
        legend_proxy_size=13,
        legend_proxy_alpha=0.8,
        legend_colour_proxy_marker='square',
        legend_shape_proxy_colour='black',
        legend_ncols=1,
    )
    space.plot(
        colour_by='CLASS', colour_map={'score[0]': 'red', 'score[1]': 'blue'},
        shape_by='METADATA', shape_map={'train': 'square', 'test': 'triangle'},
    )
    plot = show.call_args.args[0].children[0]
    assert plot.legend[0].ncols == 1
    proxies = [
        renderer for renderer in plot.renderers
        if isinstance(renderer, GlyphRenderer) and not renderer.visible
    ]
    colour_proxy = next(renderer for renderer in proxies if 'colour' in renderer.name)
    shape_proxy = next(renderer for renderer in proxies if 'shape' in renderer.name)
    assert colour_proxy.glyph.marker == 'square'
    assert colour_proxy.glyph.size == shape_proxy.glyph.size == 13
    assert colour_proxy.glyph.fill_alpha == shape_proxy.glyph.fill_alpha == 0.8
    assert shape_proxy.glyph.fill_color == 'black'


def test_continuous_colour_only_does_not_configure_missing_legend(
    encoding_space,
):
    import warnings
    from bokeh.models import ColorBar

    space, show = encoding_space
    space.metadata_exists = space.metadata_categorical = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        space.plot(colour_by='SCORE', colour_continuous=True)

    plot = show.call_args.args[0].children[0]
    assert len(plot.legend) == 0
    assert any(isinstance(item, ColorBar) for item in plot.right)
    assert not any('plot.legend.' in str(item.message) for item in caught)


@pytest.mark.parametrize('encoding', ['colour', 'shape', 'opacity', 'marker_size'])
def test_plot_mapping_validation(encoding_space, encoding):
    space, show = encoding_space
    column = 'METADATA'
    good = {'colour': 'red', 'shape': 'circle', 'opacity': 0.5, 'marker_size': 10}[encoding]
    by, mapping = f'{encoding}_by', f'{encoding}_map'
    for kwargs, message in [
        ({by: 'missing', mapping: {}}, 'Unknown'),
        ({mapping: {'train': good}}, 'requires'),
        ({by: column}, 'requires'),
        ({by: column, mapping: {'train': good}}, 'missing categories'),
        ({by: column, mapping: {'train': good, 'test': good}, f'{encoding}_list': [good, good]}, 'either'),
    ]:
        with pytest.raises(ValueError, match=message):
            space.plot(**kwargs)
    space.df_compressed.loc[space.df_compressed.index[0], column] = None
    with pytest.raises(ValueError, match='missing values'):
        space.plot(**{by: column, mapping: {'train': good, 'test': good}})
    show.assert_not_called()


@pytest.mark.parametrize('encoding,value', [
    ('colour', 'not-a-colour'), ('colour', None), ('shape', 'not-a-marker'),
    ('opacity', -0.01), ('opacity', 1.01), ('opacity', float('nan')),
    ('opacity', True), ('marker_size', 0), ('marker_size', -1),
    ('marker_size', float('inf')), ('marker_size', 'large'),
])
def test_plot_mapping_rejects_invalid_styles(encoding_space, encoding, value):
    space, show = encoding_space
    with pytest.raises(ValueError, match=f'{encoding}_map'):
        space.plot(**{
            f'{encoding}_by': 'CLASS',
            f'{encoding}_map': {'score[0]': value, 'score[1]': value},
        })
    show.assert_not_called()


@pytest.mark.parametrize('kwargs,message', [
    ({'colour_continuous': True}, 'requires'),
    ({'colour_palette': ['red', 'blue']}, 'requires'),
    ({'colour_by': 'CLASS', 'colour_continuous': True}, 'numeric'),
    ({'colour_by': 'SCORE', 'colour_continuous': True, 'colour_map': {}}, 'conflicts'),
    ({'colour_by': 'SCORE', 'colour_continuous': True, 'colour_list': ['red']}, 'either'),
    ({'colour_by': 'SCORE', 'colour_continuous': True, 'colour_palette': ['red']}, 'at least two'),
])
def test_plot_continuous_colour_validation(encoding_space, kwargs, message):
    space, show = encoding_space
    with pytest.raises(ValueError, match=message):
        space.plot(**kwargs)
    show.assert_not_called()


@pytest.mark.parametrize('bad_score', [float('nan'), float('inf')])
def test_plot_continuous_colour_rejects_nonfinite_values(encoding_space, bad_score):
    space, show = encoding_space
    space.df_compressed['SCORE'] = bad_score
    with pytest.raises(ValueError, match='missing|finite'):
        space.plot(colour_by='SCORE', colour_continuous=True)
    show.assert_not_called()


def test_legacy_legends_do_not_overlay_shape_renderers(plotting_space):
    from bokeh.models import GlyphRenderer

    space, show = plotting_space
    space.plot()
    plot = show.call_args.args[0].children[0]
    for item in plot.legend[0].items:
        assert len(item.renderers) == 1
        assert item.index == 0
        assert item.renderers[0].visible is False
        assert list(item.renderers[0].data_source.data) == ['x', 'y']
    if space.metadata_categorical:
        shape_keys = [item for item in plot.legend[0].items if item.label.value.startswith('Shape')]
        assert len(shape_keys) == len(space.metadata_names)
        assert all(item.renderers[0].glyph.fill_color == 'grey' for item in shape_keys)
    colour_keys = [item for item in plot.legend[0].items if item.label.value.startswith('Colour')]
    assert all(item.renderers[0].glyph.marker == 'circle' for item in colour_keys)
    assert all(
        item.renderers[0].glyph.size == space.bokeh_dictionary['legend_proxy_size']
        for item in colour_keys
    )


def test_plot_legacy_lists_and_notebook_output_order(plotting_space, monkeypatch):
    space, show = plotting_space
    calls = []
    monkeypatch.setattr('bokeh.io.reset_output', lambda: calls.append('reset'))
    monkeypatch.setattr('bokeh.io.output_notebook', lambda: calls.append('notebook'))
    space.plot(
        colour_list=['purple', 'gold'], shape_list=['square', 'circle', 'triangle'],
        opacity_list=[0.25, 0.75], marker_size_list=[8, 18],
    )
    assert calls == ['reset', 'notebook']
    assert_plot_groups(space, show, [0.25, 0.75], [8, 18])
    text_items = [
        item for item in show.call_args.args[0].children[0].legend[0].items
        if item.label.value.startswith(('Opacity', 'Size'))
    ]
    assert {
        item.label.value for item in text_items
    } == {
        'Opacity (CLASS): CLASS_A = 0.25',
        'Opacity (CLASS): CLASS_B = 0.75',
        'Size (CLASS): CLASS_A = 8 px',
        'Size (CLASS): CLASS_B = 18 px',
    }
    assert all(item.renderers == [] for item in text_items)


def test_plot_explicit_mapping_without_metadata_flags(encoding_space):
    space, show = encoding_space
    space.metadata_exists = space.metadata_categorical = False
    space.df_compressed['PAIR'] = [('a', 1), ('b', 2), ('a', 1), ('b', 2)]
    space.plot(
        shape_by='PAIR',
        shape_map={('a', 1): 'square', ('b', 2): 'triangle', ('extra', 3): 'circle'},
        colour_list=['navy', 'gold'],
    )
    plot = show.call_args.args[0].children[0]
    shape_items = [item for item in plot.legend[0].items if item.label.value.startswith('Shape')]
    assert len(shape_items) == 2
    assert {item.renderers[0].glyph.marker for item in shape_items} == {'square', 'triangle'}
    assert len({item.renderers[0].glyph.fill_color for item in plot.legend[0].items
                if item.label.value.startswith('Colour')}) == 2


def test_opacity_and_size_legend_entries_are_text_only(encoding_space):
    space, show = encoding_space
    space.plot(
        opacity_by='METADATA', opacity_map={'train': 0.0, 'test': 1.0},
        marker_size_by='METADATA', marker_size_map={'train': 8, 'test': 14},
    )
    items = show.call_args.args[0].children[0].legend[0].items
    text_items = [item for item in items if item.label.value.startswith(('Opacity', 'Size'))]
    assert {item.label.value for item in text_items} == {
        'Opacity (METADATA): train = 0.0',
        'Opacity (METADATA): test = 1.0',
        'Size (METADATA): train = 8 px',
        'Size (METADATA): test = 14 px',
    }
    assert all(item.renderers == [] for item in text_items)


@pytest.mark.parametrize('options', [
    {'legend_proxy_size': 0},
    {'legend_proxy_alpha': 0},
    {'legend_proxy_alpha': 1.1},
    {'legend_colour_proxy_marker': 'not-a-marker'},
    {'legend_shape_proxy_colour': 'not-a-colour'},
    {'legend_ncols': 0},
])
def test_plot_rejects_invalid_legend_proxy_options(encoding_space, options):
    space, show = encoding_space
    space.update_bokeh_options(**options)
    with pytest.raises(ValueError, match='legend_'):
        space.plot()
    show.assert_not_called()


@pytest.mark.parametrize('bad_columns,metadata_exists', [
    (['SMILES', 'NAME'], False),
    (['SMILES', 'NAME', 'CLASS'], True),
    (['SMILES', 'NAME', 'CLASS', 'CLASS'], False),
    (['SMILES', 'NAME', 'CLASS', 'MOLFILE'], False),
])
def test_constructor_rejects_missing_duplicate_or_reserved_columns(bad_columns, metadata_exists):
    with pytest.raises(ValueError, match='columns'):
        ChemicalSpace(pd.DataFrame(columns=bad_columns), metadata_exists=metadata_exists)


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