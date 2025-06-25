from typing import Literal, Iterable, Callable, Any
import pandas as pd
import numpy as np
from PIL import Image
from bokeh.models import DataTable
from bokeh.plotting import figure


# DRAWING #


def convert_size(
    size: tuple[float, float] | None = None,
    pixel_size: tuple[int, int] | None = None,
    dpi: int = 100
) -> tuple[int, int] | tuple[float, float]:
    """
    Convert between size in inches and size in pixels.

    Parameters
    ----------
    size : tuple of float, optional
        Size in inches as (width, height).

    pixel_size : tuple of int, optional
        Size in pixels as (width, height).

    dpi : int, default=100
        Dots per inch used for conversion.

    Returns
    -------
    tuple of int or tuple of float
        Converted size in pixels if `size` is provided, or in inches 
        if `pixel_size` is provided.
    """

    if size and pixel_size:
        raise ValueError("Provide either size or pixel_size, not both.")
    if not size and not pixel_size:
        raise ValueError("Either size or pixel_size must be provided.")

    if size:
        width_px = size[0] * dpi
        height_px = size[1] * dpi
        return int(width_px), int(height_px)

    if pixel_size:
        width_in = pixel_size[0] / dpi
        height_in = pixel_size[1] / dpi
        return width_in, height_in


def generate_random_rgb() -> tuple[float, float, float]:
    """
    Generate a random RGB color.

    Returns
    -------
    tuple of float
        A tuple of three floats representing an RGB color, with each 
        component in the range [0, 1].
    """

    import random
    return (random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1))


def convert_rgb(
    rgb_tuple: tuple[int, int, int],
    mode: Literal['normalise', 'denormalise']
) -> tuple[float, float, float] | tuple[int, int, int]:
    """
    Convert RGB values between 0-255 and 0-1 ranges.

    Parameters
    ----------
    rgb_tuple : tuple of int
        RGB values in the form (R, G, B).

    mode : {'normalise', 'denormalise'}
        Conversion mode. 'normalise' converts from 0-255 to 0-1, 
        'denormalise' converts from 0-1 to 0-255.

    Returns
    -------
    tuple of float or tuple of int
        Converted RGB values.
    """

    if mode == 'normalise':
        return tuple(value / 255.0 for value in rgb_tuple)    # (0,1)
    elif mode == 'denormalise':
        return tuple(int(value * 255) for value in rgb_tuple)     # (0,255)
    else:
        raise ValueError("Mode must be either 'normalise' or 'denormalise'.")


def make_rgb_transparent(
    fg_rgb: tuple[float, float, float],
    bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    alpha: float = 0.5
) -> tuple[float, float, float]:
    """
Apply transparency to a foreground RGB color over a background color.

Parameters
----------
fg_rgb : tuple of float
    Foreground RGB color in the range [0, 1].

bg_rgb : tuple of float, optional
    Background RGB color in the range [0, 1]. Default is white 
    (1.0, 1.0, 1.0).

alpha : float, default=0.5
    Transparency level, where 0 is fully transparent and 1 is fully opaque.

Returns
-------
tuple of float
    RGB color after applying transparency.
"""

    result = [alpha * c1 + (1 - alpha) * c2 for
              c1, c2 in
              zip(fg_rgb, bg_rgb)]
    return result[0], result[1], result[2]


def visualise_colour(
    rgb_tuple: tuple[float, float, float]
) -> None:
    """
Display a single RGB color.

Parameters
----------
rgb_tuple : tuple of float
    RGB values in the range [0, 1].

Returns
-------
None
"""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_facecolor(rgb_tuple)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def visualise_colour_grid(
    colour_dictionary: dict[str, tuple[float, float, float]],
    save: bool = False,
    filename: str = '',
    figsize: tuple[int, int] = (20, 20)
) -> None:
    """
Display a grid of RGB colors.

Parameters
----------
colour_dictionary : dict of str to tuple of float
    Dictionary mapping color names to RGB tuples.

save : bool, default=False
    Whether to save the figure.

filename : str, optional
    Filename to save the figure if `save` is True.

figsize : tuple of int, default=(20, 20)
    Size of the figure.

Returns
-------
None
"""

    import numpy as np
    import matplotlib.pyplot as plt

    # Calculate the number of rows and columns to display the colour grid
    num_colors = len(colour_dictionary)
    num_columns = int(np.ceil(np.sqrt(num_colors)))
    num_rows = int(np.ceil(num_colors / num_columns))

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=figsize)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through the colour dictionary and create a subplot for each colour
    for ax, (colour_name, rgb_tuple) in zip(axes, colour_dictionary.items()):
        ax.set_facecolor(rgb_tuple)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(colour_name, fontsize=12)

    # Hide any unused subplots
    for ax in axes[len(colour_dictionary):]:
        ax.axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    plt.show()


def show_png(data: bytes) -> Image.Image:
    """
Display a PNG image from binary data.

Parameters
----------
data : bytes
    Binary data of the PNG image.

Returns
-------
Image.Image
    PIL Image object.
"""

    import io

    bio = io.BytesIO(data)
    img = Image.open(bio)
    return img


def create_smooth_gradient_circle(
    radius: int,
    color: tuple[int, int, int],
    alpha: float
) -> Image.Image:
    """
Create a smooth gradient circle with transparency.

Parameters
----------
radius : int
    Radius of the circle.

color : tuple of int
    Base RGB color in the range [0, 255].

alpha : float
    Transparency level in the range [0, 1].

Returns
-------
Image.Image
    PIL Image object containing the gradient circle.
"""

    from PIL import ImageDraw

    diameter = radius * 2
    gradient = Image.new('RGBA', (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)

    for i in range(radius):
        current_alpha = int(255 * alpha * (1 - i / radius))
        draw.ellipse(
            (i, i, diameter - i, diameter - i),
            fill=color + (current_alpha,)
        )

    return gradient


# MISCELLANEOUS #


def suppress_warnings() -> None:
    """
Suppress all warnings in the current Python session.

Returns
-------
None
"""

    import warnings
    warnings.filterwarnings("ignore")


def standardise_path(path: str) -> str:
    """
Convert a Windows-style path to a standardised format.

Parameters
----------
path : str
    Path string with backslashes.

Returns
-------
str
    Path string with forward slashes.
"""

    return path.replace("\\", "/")


def generate_combination_cascade(
    elements: Iterable,
    n: int
) -> Iterable[Iterable]:
    """
Generate all combinations of elements from size 1 to n.

Parameters
----------
elements : iterable
    Input elements to combine.

n : int
    Maximum size of combinations.

Returns
-------
list of list
    List of combinations of elements.
"""

    from itertools import combinations

    cols = [list(a) for a in combinations(elements, 1)]
    for i in range(2, n + 1):
        cols += [list(a) for a in combinations(elements, i)]
    return cols


def count_features(list_features: Iterable[str]) -> int:
    """
Count the total number of features, including interaction terms.

Parameters
----------
list_features : iterable of str
    List of feature names, possibly including interaction terms.

Returns
-------
int
    Total number of features.

Example
-------
>>> count_features(['a', 'b', 'a b'])
4
>>> count_features(['a', 'a b', 'c^2'])
5
>>> count_features(['a', 'b', 'c', 'c a', 'c b^2', 'a^3'])
11
"""
    import numpy as np

    flattened_list = np.hstack([a.split(' ') for a in list_features])
    count_degree_1 = len(flattened_list)
    count_degree_2 = sum('^2' in s for s in flattened_list)
    count_degree_3 = sum('^3' in s for s in flattened_list)
    final_count = count_degree_1 + count_degree_2 + (2 * count_degree_3)
    return final_count


def loadingbar(
    count: int,
    total: int,
    size: int
) -> None:
    """
Display a loading bar to indicate progress.

Parameters
----------
count : int
    Current iteration.

total : int
    Total number of iterations.

size : int
    Length of the loading bar.

Returns
-------
None
"""
    import sys

    percent = float(count) / float(total)
    filled_length = int(size * percent)
    bar = '=' * filled_length + ' ' * (size - filled_length)
    sys.stdout.\
        write(
            f"\r{str(count).rjust(3, '0')}/{str(total).rjust(3, '0')} [{bar}]"
            )
    sys.stdout.flush()


def create_progressive_column_names(
    serial_name: str,
    n: int
) -> list[str]:
    """
Generate a list of sequential column names.

Parameters
----------
serial_name : str
    Base name for the columns.

n : int
    Number of columns to generate.

Returns
-------
list of str
    List of column names.
"""

    return [f"{serial_name}{i + 1}" for i in range(n)]


def try_except(
    func: Callable[[], Any],
    exc: Any = None
) -> Any:
    """
Execute a function and return its result or a fallback value on exception.

Parameters
----------
func : callable
    Function to execute.

exc : any, optional
    Value to return if an exception occurs. Default is None.

Returns
-------
any
    Result of the function or fallback value.
"""

    try:
        return func()
    except Exception:
        return exc


def find_all_occurrences(
    text: str,
    substring: str
) -> list[int]:
    """
Find all starting indices of a substring in a text.

Parameters
----------
text : str
    Text to search.

substring : str
    Substring to find.

Returns
-------
list of int
    List of starting indices where the substring occurs.
"""

    indices = []
    start = 0
    while True:
        index = text.find(substring, start)
        if index == -1:
            break
        indices.append(index)
        start = index + 1
    return indices


def reset_string(input_string: str) -> str:
    """
Remove punctuation and convert a string to lowercase.

Parameters
----------
input_string : str
    Input string.

Returns
-------
str
    Processed string.
"""

    punctuation = '''"!()-[]{};:'\\, <>./?@#$%^&*_~'''
    for ch in input_string:
        if ch in punctuation:
            input_string = input_string.replace(ch, "")
    return input_string.lower()


def insert_string_piece(
    text: str,
    substring: str,
    index: int
) -> str:
    """
Insert a substring into a string at a specified index.

Parameters
----------
text : str
    Original string.

substring : str
    Substring to insert.

index : int
    Index at which to insert the substring.

Returns
-------
str
    Modified string.

Raises
------
ValueError
    If index is less than 0.
"""

    if index > 0:
        return text[:index + 1] + substring + text[index + 1:]
    elif index < 0:
        raise ValueError('index must be at least 0')
    elif index == 0:
        return substring + text


def flatten(args: Any) -> tuple[Any, ...]:
    """
Flatten a nested structure into a single tuple.

Parameters
----------
args : Any
    The nested structure to flatten.

Returns
-------
tuple
    A flattened tuple containing all elements.
"""

    try:
        iter(args)
        final = []
        for arg in args:
            final += flatten(arg)
        return tuple(final)
    except TypeError:
        return (args,)


def process_custom_string(
    s: str,
    target_substring: str,
    replacement_list: list[str],
    separator: str = ';'
) -> str:
    """
Process a string by replacing a target substring with elements from a 
list and formatting the result.

Parameters
----------
s : str
    The original string.

target_substring : str
    The substring to replace.

replacement_list : list of str
    List of strings to replace the target substring.

separator : str, default=';'
    Separator used in formatting the result.

Returns
-------
str
    The processed and formatted string.
"""

    import string

    # Replace target_substring with respective strings from the list
    for replacement in replacement_list:
        s = s.replace(target_substring, replacement, 1)

    # Remove punctuation characters
    s = ''.join(char for char in s if char not in string.punctuation)

    # Replace all digits with the separator
    s = ''.join(separator if char.isdigit() else char for char in s)

    # Replace all remaining non-punctuation characters with the separator,
    # except for the replacements
    result = []
    replacement_index = 0
    i = 0
    while i < len(s):
        if (replacement_index < len(replacement_list) and
                s[i:i+len(replacement_list[replacement_index])] ==
                replacement_list[replacement_index]):
            if result and result[-1] != separator:
                result.append(separator)
            result.append(replacement_list[replacement_index])
            i += len(replacement_list[replacement_index])
            replacement_index += 1
        else:
            result.append(separator if s[i].isalnum() else s[i])
            i += 1

    return ''.join(result)


def sort_list_by_other_list(list_1: list,
                            list_2: list[int | float]
                            ) -> tuple[list[str], list[int | float]]:
    """
Sort one list based on the absolute values of another list.

Parameters
----------
list_1 : list
    List of elements to sort.

list_2 : list of int or float
    List of values used to determine sort order.

Returns
-------
tuple of list
    Sorted list of elements and corresponding sorted values.
"""


    combined = list(zip(list_1, list_2))
    sorted_combined = sorted(combined, key=lambda x: abs(x[1]), reverse=True)
    
    sorted_elements = [x[0] for x in sorted_combined]
    sorted_values = [x[1] for x in sorted_combined]
    
    return (sorted_elements, sorted_values)


def merge_dicts_with_duplicates(
    dict1: dict[str, Any],
    dict2: dict[str, Any]
) -> dict[str, Any]:
    """
Merge two dictionaries, renaming duplicate keys from the second dictionary.

Parameters
----------
dict1 : dict of str to Any
    First dictionary.

dict2 : dict of str to Any
    Second dictionary.

Returns
-------
dict of str to Any
    Merged dictionary with unique keys.
"""

    # Start with a copy of the first dictionary
    merged_dict = dict(dict1)

    for key, value in dict2.items():
        if key in merged_dict:
            # If the key already exists,
            # create a new key with '_duplicate' suffix
            new_key = f"{key}_duplicate"
            # Ensure the new key is unique
            while new_key in merged_dict:
                new_key += "_duplicate"
            merged_dict[new_key] = value
        else:
            merged_dict[key] = value

    return merged_dict


def add_inchi_to_dataframe(
    df: pd.DataFrame,
    loc: int,
    smiles_column_name: str
) -> pd.DataFrame:
    """
Add an InChI column to a DataFrame by converting SMILES strings.

Parameters
----------
df : pandas.DataFrame
    Input DataFrame.

loc : int
    Column index to insert the InChI column.

smiles_column_name : str
    Name of the column containing SMILES strings.

Returns
-------
pandas.DataFrame
    DataFrame with the added InChI column.
"""

    from rdkit import Chem

    inchi_list = [Chem.MolToInchi(Chem.MolFromSmiles(s))
                  for s in df[smiles_column_name]]
    df.insert(loc, 'INCHI', inchi_list)
    return df


def identify_df_duplicates(
    df: pd.DataFrame,
    column_name: str,
    keep: Literal['first', 'last'] = 'last'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
Identify and separate duplicate rows in a DataFrame based on a column.

Parameters
----------
df : pandas.DataFrame
    Input DataFrame.

column_name : str
    Column to check for duplicates.

keep : {'first', 'last'}, default='last'
    Which duplicate to keep.

Returns
-------
tuple of pandas.DataFrame
    Cleaned DataFrame and DataFrame of removed duplicates.
"""

    df_copy = df.copy()
    df_copy.drop_duplicates(subset=column_name, keep=keep, inplace=True)
    cleaned_df_copy = df_copy[df_copy.index.isin(df_copy.index)]
    duplicates_removed = df[~df.index.isin(df_copy.index)]
    return cleaned_df_copy, duplicates_removed


def create_structure_files(
    df: pd.DataFrame,
    structure_column_name: str,
    folder_name: str
) -> None:
    """
Create PNG structure files for molecules in a DataFrame.

Parameters
----------
df : pandas.DataFrame
    Input DataFrame.

structure_column_name : str
    Column containing molecular structures.

folder_name : str
    Folder to save the PNG images.

Returns
-------
None
"""


    import os
    import shutil
    from mlchem.chem.manipulation import create_molecule
    from rdkit import Chem
    from rdkit.Chem import Draw

    # Create the folder, removing it first if it already exists
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)

    for i, s in enumerate(df[structure_column_name]):
        try:
            mol = create_molecule(s)
            filename = f'{folder_name}/{i}.png'
            Draw.MolToFile(mol, filename)
        except Exception:
            # If molecule does not get read properly,
            # create a placeholder image
            filename = f'{folder_name}/{i}.png'
            Draw.MolToFile(Chem.MolFromSmarts('*'), filename)


def prepare_dataframe(
    df: pd.DataFrame,
    dir_name: str
) -> pd.DataFrame:
    """
Add a 'MOLFILE' column to a DataFrame with sorted file paths.

Parameters
----------
df : pandas.DataFrame
    Input DataFrame.

dir_name : str
    Directory containing the files.

Returns
-------
pandas.DataFrame
    DataFrame with the added 'MOLFILE' column.
"""

    import os

    # Sort list of files based on last modification time in ascending order
    list_of_files = filter(
        lambda x: os.path.isfile(os.path.join(dir_name, x)),
        os.listdir(dir_name)
    )
    sorted_files = sorted(
        list_of_files,
        key=lambda x: os.path.getmtime(os.path.join(dir_name, x))
    )
    file_paths = [os.path.join(dir_name, f) for f in sorted_files]
    df.insert(2, 'MOLFILE', file_paths)

    return df


def prepare_datatable(
    df: pd.DataFrame,
    height: int = 500,
    width: int = int(650 / 0.75)
) -> DataTable:
    """
Create a Bokeh DataTable from a DataFrame.

Parameters
----------
df : pandas.DataFrame
    Input DataFrame.

height : int, default=500
    Height of the DataTable.

width : int, default=int(650 / 0.75)
    Width of the DataTable.

Returns
-------
bokeh.models.DataTable
    Bokeh DataTable object.
"""

    from bokeh.models import ColumnDataSource, TableColumn

    source = ColumnDataSource(df)
    columns = [
        TableColumn(field='DIM_1', title='DIM_1'),
        TableColumn(field='DIM_2', title='DIM_2'),
        TableColumn(field='SMILES', title='SMILES'),
        TableColumn(field='MOLFILE', title='MOLFILE'),
        TableColumn(field='NAME', title='NAME'),
        TableColumn(field='NAME_SHORT', title='NAME_SHORT'),
        TableColumn(field='METADATA', title='METADATA')
    ]
    return DataTable(source=source, columns=columns,
                     height=height, width=width)


def compute_alpha(size: int) -> float:
    """
Compute transparency value based on sample size.

Parameters
----------
size : int
    Sample size.

Returns
-------
float
    Computed alpha value.
"""

    if size < 100:
        alpha = 0.95
    elif 100 < size < 300:
        alpha = 0.9
    elif 300 < size < 800:
        alpha = 0.8
    elif 800 < size < 2000:
        alpha = 0.5
    else:
        alpha = 0.2
    return alpha


def size_ratio(
    size1: int,
    size2: int
) -> float:
    """
Compute a ratio based on the relative sizes of two values.

Parameters
----------
size1 : int
    First size.

size2 : int
    Second size.

Returns
-------
float
    Computed ratio.
"""

    return 1 - ((size1 / (size1 + size2)) / 2)


def bokeh_plot(
    p: figure,
    classnames: list[str],
    dict_datatables: dict[str, DataTable]
) -> None:
    """
Display a Bokeh plot with associated DataTables.

Parameters
----------
p : bokeh.plotting.Figure
    Bokeh plot to display.

classnames : list of str
    List of class names.

dict_datatables : dict of str to DataTable
    Mapping of class names to DataTables.

Returns
-------
None
"""

    import bokeh.plotting as bp

    tables = [dict_datatables[classname] for classname in classnames]
    bp.show(bp.column(p, *tables))


def create_mask(
    array: np.ndarray,
    lower: float | int,
    upper: float | int
) -> np.ndarray:
    """
Create a boolean mask for values within a specified range.

Parameters
----------
array : numpy.ndarray
    Input array.

lower : float or int
    Lower bound.

upper : float or int
    Upper bound.

Returns
-------
numpy.ndarray
    Boolean mask array.
"""

    return (array <= upper) & (array > lower)


def assign_sign(x: float | int) -> str:
    """
Return the sign of a number as '+' or '-'.

Parameters
----------
x : float or int
    Input value.

Returns
-------
str
    '+' if x is non-negative, '-' otherwise.
"""

    return '+' if x >= 0 else '-'


def normalise_iterable(
    values: Iterable[float | int]
) -> list[float]:
    """
Normalise values in an iterable so that the maximum absolute value is 1.

Parameters
----------
values : iterable of float or int
    Iterable of numerical values.

Returns
-------
list of float
    Normalised values.
"""

    import math

    max_abs_weight = max(math.fabs(v) for v in values)

    if max_abs_weight > 0:
        return [v / max_abs_weight for v in values]
    return list(values)


# DESCRIPTORS #


def dfs_to_excel(file_name: str,
                 dfs: Iterable[pd.DataFrame],
                 sheet_names: Iterable[str]) -> None:
    """
Write multiple DataFrames to an Excel file, each on a separate sheet.

Parameters
----------
file_name : str
    Name of the Excel file.

dfs : iterable of pandas.DataFrame
    DataFrames to write.

sheet_names : iterable of str
    Names of the sheets.

Returns
-------
None
"""

    with pd.ExcelWriter(file_name) as writer:
        for desc_df, sheet_name in zip(dfs, sheet_names):
            desc_df.to_excel(writer, sheet_name=sheet_name, index=True)
