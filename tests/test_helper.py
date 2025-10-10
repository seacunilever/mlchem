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
    create_mask, assign_sign, normalise_iterable, dfs_to_excel
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

    # Ensure no warnings are raised after suppressing

    assert not warnings.warn("new warning")

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

def test_compute_alpha():
    assert compute_alpha(50) == 0.95
    assert compute_alpha(150) == 0.9
    assert compute_alpha(500) == 0.8
    assert compute_alpha(1000) == 0.5
    assert compute_alpha(3000) == 0.2

def test_size_ratio():
    assert size_ratio(1, 1) == 0.75
    assert size_ratio(1, 3) == 0.875

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

def test_dfs_to_excel(tmpdir):
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"C": [5, 6], "D": [7, 8]})
    file_path = tmpdir.join("test.xlsx")
    dfs_to_excel(file_path, [df1, df2], ["Sheet1", "Sheet2"])
    assert file_path.exists()

if __name__ == "__main__":
    pytest.main()