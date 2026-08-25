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
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from PIL import Image
from bokeh.models import DataTable
import matplotlib.pyplot as plt
from mlchem.helper import (
    convert_size, generate_random_rgb, convert_rgb, make_rgb_transparent,
    visualise_colour, visualise_colour_grid, show_png, create_smooth_gradient_circle,
    suppress_warnings, standardise_path, generate_combination_cascade,
    count_features, loadingbar, create_progressive_column_names, try_except,
    find_all_occurrences, reset_string, insert_string_piece, flatten,
    process_custom_string, sort_list_by_other_list, merge_dicts_with_duplicates,
    add_inchi_to_dataframe, identify_df_duplicates, create_structure_files,
    prepare_dataframe, prepare_datatable, compute_alpha, size_ratio, bokeh_plot,
    create_mask, assign_sign, normalise_iterable, dfs_to_excel, generate_cartesian_product
)

def test_convert_size():
    assert convert_size(size=(2, 3), dpi=100) == (200, 300)
    assert convert_size(pixel_size=(200, 300), dpi=100) == (2.0, 3.0)
    with pytest.raises(ValueError):
        convert_size(size=(2, 3), pixel_size=(200, 300), dpi=100)

def test_generate_random_rgb():
    rgb = generate_random_rgb()
    assert all(0 <= value <= 1 for value in rgb)

def test_convert_rgb():
    assert convert_rgb((255, 128, 64), 'normalise') == (1.0, 0.5019607843137255, 0.25098039215686274)
    assert convert_rgb((1.0, 0.5, 0.25), 'denormalise') == (255, 127, 63)
    with pytest.raises(ValueError):
        convert_rgb((255, 128, 64), 'invalid_mode')

def test_make_rgb_transparent():
    assert make_rgb_transparent((1.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.5) == (1.0, 0.5, 0.5)

def test_visualise_colour():
    with patch('matplotlib.pyplot.show') as mock_show:
        visualise_colour((1.0, 0.0, 0.0))
        assert mock_show.called

def test_visualise_colour_grid():
    from mlchem.importables import colour_dictionary

    plt.close('all')
    plt.switch_backend('Agg')

    with patch('matplotlib.pyplot.show') as mock_show:
        visualise_colour_grid(colour_dictionary)
        assert mock_show.called


def test_visualise_colour_grid_save_calls_savefig(tmp_path):
    colour_dictionary = {
        'red': (1.0, 0.0, 0.0),
        'green': (0.0, 1.0, 0.0),
        'blue': (0.0, 0.0, 1.0),
    }
    output_file = tmp_path / 'palette.png'

    plt.close('all')
    plt.switch_backend('Agg')

    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        with patch('matplotlib.pyplot.show'):
            visualise_colour_grid(
                colour_dictionary,
                save=True,
                filename=str(output_file),
                figsize=(4, 4),
            )

    mock_savefig.assert_called_once_with(str(output_file))

def test_show_png():
    # Create a simple PNG image in memory
    img = Image.new('RGB', (10, 10), color='red')
    
    # Convert the image to bytes in PNG format
    import io
    bio = io.BytesIO()
    img.save(bio, format='PNG')
    img_bytes = bio.getvalue()
    
    img_loaded = show_png(img_bytes)
    assert isinstance(img_loaded, Image.Image)

def test_create_smooth_gradient_circle():
    img = create_smooth_gradient_circle(10, (255, 0, 0), 0.5)
    assert isinstance(img, Image.Image)

def test_suppress_warnings():
    import warnings
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn("another warning", RuntimeWarning)

    assert len(record) == 1

    # Suppress warnings
    suppress_warnings()

    # Ensure warning emission is suppressed after calling helper
    with warnings.catch_warnings(record=True) as suppressed:
        warnings.warn("new warning", RuntimeWarning)
        assert len(suppressed) == 0

def test_standardise_path():
    assert standardise_path("C:\\Users\\User\\Documents") == "C:/Users/User/Documents"

def test_generate_combination_cascade():
    result = generate_combination_cascade(['a', 'b', 'c'], 2)
    expected = [['a'], ['b'], ['c'], ['a', 'b'], ['a', 'c'], ['b', 'c']]
    assert result == expected

def test_count_features():
    assert count_features(['a', 'b', 'a b']) == 4
    assert count_features(['a', 'a b', 'c^2']) == 5

def test_loadingbar(capsys):
    loadingbar(1, 3, 10)
    captured = capsys.readouterr()
    assert "001/003 [===       ]" in captured.out

def test_create_progressive_column_names():
    assert create_progressive_column_names("col", 3) == ["col1", "col2", "col3"]

def test_try_except():
    assert try_except(lambda: 1 / 0, exc="error") == "error"
    assert try_except(lambda: 1 / 1) == 1.0


def test_try_except_exception_scope_and_validation():
    assert try_except(lambda: 1 / 0, exc="zero", exceptions=ZeroDivisionError) == "zero"

    with pytest.raises(KeyError):
        try_except(lambda: {}['missing'], exc="fallback", exceptions=ValueError)

    with pytest.raises(TypeError):
        try_except(lambda: 1, exceptions=(ValueError, 'invalid'))

def test_find_all_occurrences():
    assert find_all_occurrences("test test test", "test") == [0, 5, 10]

def test_reset_string():
    assert reset_string("Hello, World!") == "helloworld"

def test_insert_string_piece():
    assert insert_string_piece("Hello World", "Beautiful ", 5) == "Hello Beautiful World"
    with pytest.raises(ValueError):
        insert_string_piece("Hello World", "Beautiful ", -1)

def test_flatten():
    assert flatten([1, [2, 3], [[4, 5], 6]]) == (1, 2, 3, 4, 5, 6)


def test_flatten_treats_strings_as_atomic_values():
    assert flatten('abc') == ('abc',)
    assert flatten(['ab', ['cd']]) == ('ab', 'cd')


def test_process_custom_string():
    result = process_custom_string(s='Nc1ccccc1N',
                                   target_substring='c1ccccc1',
                                   replacement_list=['CYC'])
    assert result == ";CYC;"

def test_sort_list_by_other_list():
    strings = ["apple", "banana", "cherry", "date"]
    values = [3, -7, 2, -5]
    result = sort_list_by_other_list(strings,values)
    assert result == (['banana', 'date', 'apple', 'cherry'], [-7, -5, 3, 2])

def test_merge_dicts_with_duplicates():
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 3, "c": 4}
    result = merge_dicts_with_duplicates(dict1, dict2)
    expected = {"a": 1, "b": 2, "b_duplicate": 3, "c": 4}
    assert result == expected

def test_add_inchi_to_dataframe():
    df = pd.DataFrame({"SMILES": ["CCO", "CCC"]})
    result = add_inchi_to_dataframe(df, 1, "SMILES")
    assert "INCHI" in result.columns

def test_identify_df_duplicates():
    df = pd.DataFrame({"A": [1, 2, 2, 3], "B": [4, 5, 5, 6]})
    cleaned_df, duplicates_df = identify_df_duplicates(df, "A")
    assert len(cleaned_df) == 3
    assert len(duplicates_df) == 1


def test_identify_df_duplicates_validation_errors():
    df = pd.DataFrame({"A": [1, 2, 2], "B": [3, 4, 4]})

    with pytest.raises(KeyError):
        identify_df_duplicates(df, "missing")

    with pytest.raises(ValueError):
        identify_df_duplicates(df, "A", keep='invalid')

def test_create_structure_files(tmpdir):
    df = pd.DataFrame({"SMILES": ["CCO", "CCC"]})
    folder = tmpdir.mkdir("structures")
    create_structure_files(df, "SMILES", str(folder))
    assert len(folder.listdir()) == 2

def test_prepare_dataframe(tmpdir):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    folder = tmpdir.mkdir("files")
    file1 = folder.join("file1.txt")
    file2 = folder.join("file2.txt")
    file1.write("content1")
    file2.write("content2")
    result = prepare_dataframe(df, str(folder))
    assert "MOLFILE" in result.columns

def test_prepare_datatable():
    df = pd.DataFrame({"DIM_1": [1, 2], "DIM_2": [3, 4],
                       "SMILES": ["CCO", "CCC"],
                       "MOLFILE": ["file1", "file2"],
                       "NAME": ["name1", "name2"],
                       "NAME_SHORT": ["n1", "n2"],
                       "METADATA": ["meta1", "meta2"]})
    datatable = prepare_datatable(df)
    assert isinstance(datatable, DataTable)


def test_prepare_datatable_missing_columns():
    df = pd.DataFrame({"DIM_1": [1], "DIM_2": [2]})
    with pytest.raises(KeyError):
        prepare_datatable(df)

def test_compute_alpha():
    assert compute_alpha(50) == 0.95
    assert compute_alpha(100) == 0.9
    assert compute_alpha(150) == 0.9
    assert compute_alpha(300) == 0.8
    assert compute_alpha(500) == 0.8
    assert compute_alpha(800) == 0.5
    assert compute_alpha(1000) == 0.5
    assert compute_alpha(2000) == 0.2
    assert compute_alpha(3000) == 0.2

    with pytest.raises(ValueError):
        compute_alpha(-1)

def test_size_ratio():
    assert size_ratio(1, 1) == 0.75
    assert size_ratio(1, 3) == 0.875

    with pytest.raises(ValueError):
        size_ratio(0, 0)

    with pytest.raises(ValueError):
        size_ratio(-1, 1)

def test_bokeh_plot(monkeypatch):
    from bokeh.plotting import figure

    show_mock = MagicMock()
    monkeypatch.setattr("bokeh.plotting.show", show_mock)

    p = figure()
    p.line([1, 2, 3], [4, 5, 6])  # Add a renderer to the plot
    classnames = ["class1", "class2"]
    dict_datatables = {"class1": DataTable(), "class2": DataTable()}
    bokeh_plot(p, classnames, dict_datatables)
    
    # Check that bokeh.io.show was called once
    show_mock.assert_called_once()

def test_create_mask():
    array = np.array([1, 2, 3, 4, 5])
    mask = create_mask(array, 1.9, 4.1)
    assert np.array_equal(mask, np.array([False, True, True, True, False]))

def test_assign_sign():
    assert assign_sign(5) == "+"
    assert assign_sign(-3) == "-"

def test_normalise_iterable():
    assert normalise_iterable([1, 2, 3, 4]) == [1/4, 2/4, 3/4, 4/4]
    assert normalise_iterable([-1, -2, -3, -4]) == [-1/4, -2/4, -3/4, -4/4]
    assert normalise_iterable([0, 0, 0]) == [0, 0, 0]
    assert normalise_iterable([]) == []
    assert normalise_iterable((x for x in [1, 2, 3])) == [1/3, 2/3, 1]

def test_dfs_to_excel(tmpdir):
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"C": [5, 6], "D": [7, 8]})
    file_path = tmpdir.join("test.xlsx")
    dfs_to_excel(file_path, [df1, df2], ["Sheet1", "Sheet2"])
    assert file_path.exists()

def test_generate_cartesian_product_two_lists():
    """Test Cartesian product of two lists."""
    result = generate_cartesian_product([1, 2], ['a', 'b'])
    expected = [[1, 'a'], [1, 'b'], [2, 'a'], [2, 'b']]
    assert result == expected
    # Verify output is list of lists, not tuples
    assert all(isinstance(combo, list) for combo in result)

def test_generate_cartesian_product_three_lists():
    """Test Cartesian product of three lists."""
    result = generate_cartesian_product([True, False], ['x', 'y'], [1])
    expected = [[True, 'x', 1], [True, 'y', 1], [False, 'x', 1], [False, 'y', 1]]
    assert result == expected

def test_generate_cartesian_product_single_element_lists():
    """Test Cartesian product when lists have single elements."""
    result = generate_cartesian_product([1], ['a'], [True])
    expected = [[1, 'a', True]]
    assert result == expected

def test_generate_cartesian_product_many_lists():
    """Test Cartesian product with many lists."""
    result = generate_cartesian_product([1, 2], ['a'], [True, False], [10])
    expected = [
        [1, 'a', True, 10],
        [1, 'a', False, 10],
        [2, 'a', True, 10],
        [2, 'a', False, 10]
    ]
    assert result == expected

def test_generate_cartesian_product_with_tuples():
    """Test that tuples are accepted as input."""
    result = generate_cartesian_product((1, 2), ('a', 'b'))
    expected = [[1, 'a'], [1, 'b'], [2, 'a'], [2, 'b']]
    assert result == expected

def test_generate_cartesian_product_mixed_types():
    """Test Cartesian product with mixed data types."""
    result = generate_cartesian_product([1, 2.5], ['a', None], [True])
    assert len(result) == 4
    assert [1, 'a', True] in result
    assert [2.5, None, True] in result

def test_generate_cartesian_product_fewer_than_two_lists():
    """Test that ValueError is raised with fewer than two lists."""
    with pytest.raises(ValueError, match="At least two lists are required"):
        generate_cartesian_product([1, 2])
    
    with pytest.raises(ValueError, match="At least two lists are required"):
        generate_cartesian_product()

def test_generate_cartesian_product_empty_list():
    """Test that ValueError is raised when any list is empty."""
    with pytest.raises(ValueError, match="is empty"):
        generate_cartesian_product([1, 2], [])
    
    with pytest.raises(ValueError, match="is empty"):
        generate_cartesian_product([], [1, 2])

def test_generate_cartesian_product_non_list_argument():
    """Test that TypeError is raised for non-list/tuple arguments."""
    with pytest.raises(TypeError, match="must be a list or tuple"):
        generate_cartesian_product([1, 2], 'abc')
    
    with pytest.raises(TypeError, match="must be a list or tuple"):
        generate_cartesian_product([1, 2], {'a': 1})

def test_generate_cartesian_product_large_output():
    """Test that function correctly computes large Cartesian products."""
    # 3 x 3 x 3 = 27 combinations
    result = generate_cartesian_product([1, 2, 3], ['a', 'b', 'c'], [10, 20, 30])
    assert len(result) == 27
    # Check first and last combinations
    assert result[0] == [1, 'a', 10]
    assert result[-1] == [3, 'c', 30]

if __name__ == "__main__":
    pytest.main()