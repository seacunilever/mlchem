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

from typing import Literal, Iterable, Callable, Any, Sequence, Collection
import logging
import hashlib
import re
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

    Examples
    --------
    >>> convert_size(size=(2, 3), dpi=100)
    (200, 300)
    >>> convert_size(pixel_size=(200, 300), dpi=100)
    (2.0, 3.0)
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

    Examples
    --------
    >>> rgb = generate_random_rgb()
    >>> len(rgb), all(0 <= value <= 1 for value in rgb)
    (3, True)
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

    Examples
    --------
    >>> convert_rgb((255, 128, 64), 'normalise')
    (1.0, 0.5019607843137255, 0.25098039215686274)
    >>> convert_rgb((1.0, 0.5, 0.25), 'denormalise')
    (255, 127, 63)
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

Examples
--------
>>> make_rgb_transparent((1.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.5)
(1.0, 0.5, 0.5)
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

Examples
--------
>>> visualise_colour((1.0, 0.0, 0.0))
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

Examples
--------
>>> visualise_colour_grid({'red': (1.0, 0.0, 0.0), 'blue': (0.0, 0.0, 1.0)})
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

Examples
--------
>>> from io import BytesIO
>>> from PIL import Image
>>> image_data = BytesIO()
>>> Image.new('RGB', (1, 1), color='red').save(image_data, format='PNG')
>>> show_png(image_data.getvalue()).size
(1, 1)
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

Examples
--------
>>> create_smooth_gradient_circle(10, (255, 0, 0), 0.5).size
(20, 20)
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


def coerce_log_level(level: int | str) -> int:
    """
Resolve a logging level name or integer into a numeric logging level.

Parameters
----------
level : int or str
    Logging level as integer (e.g. 20) or level name (e.g. 'INFO').

Returns
-------
int
    Numeric logging level.

Raises
------
ValueError
    If level is not a valid logging level name or integer.

Examples
--------
>>> coerce_log_level('INFO')
20
"""

    if isinstance(level, int):
        return level

    if isinstance(level, str):
        level_map = logging.getLevelNamesMapping()
        key = level.upper()
        if key in level_map:
            return level_map[key]

    raise ValueError("'log_level' must be a valid logging level name or integer.")


class LogCapture:
    """
Configure mlchem pipeline logging with optional file persistence.

By default, mlchem functions emit INFO-level logs to console. Use LogCapture
to add file logging, change the log level, or silence console output.

Parameters
----------
log_level : int or str, default='INFO'
    Logging level threshold (e.g., 'INFO', 'DEBUG', 'WARNING').

to_console : bool, default=True
    Whether to print logs to console. Set False to log only to file.

to_file : str or None, default=None
    Optional file path to write logs to. If provided, logs are appended
    to the file as well as console (if to_console=True).

Attributes
----------
stream : StringIO
    In-memory buffer containing captured logs.

Examples
--------
>>> # Default: logs appear on console automatically
>>> sfs = SequentialForwardSelection(...)
>>> sfs.fit(X_train, y_train, X_test, y_test)
>>> # Output appears on console immediately

>>> # Optionally: capture to file + memory for inspection
>>> logs = LogCapture(to_file='pipeline.log')
>>> sfs.fit(X_train, y_train, X_test, y_test)
>>> print(logs.get_logs())  # View memory capture
>>> # Also check pipeline.log file

>>> # Silent mode: only to file
>>> logs = LogCapture(to_console=False, to_file='pipeline.log')
>>> sfs.fit(X_train, y_train, X_test, y_test)
"""

    def __init__(self, log_level: int | str | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = logging.INFO,
                 to_console: bool = True,
                 to_file: str | None = None) -> None:
        from io import StringIO

        self.to_console = to_console
        self.to_file = to_file
        self.level = coerce_log_level(log_level)
        self.stream = StringIO()
        self._memory_handler = None
        self._console_handler = None
        self._file_handler = None
        self._root_logger = logging.getLogger()
        self._original_level = self._root_logger.level
        self._original_handlers = list(self._root_logger.handlers)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        # Set up memory handler
        self._memory_handler = logging.StreamHandler(self.stream)
        self._memory_handler.setLevel(self.level)
        self._memory_handler.setFormatter(formatter)
        self._root_logger.addHandler(self._memory_handler)
        self._root_logger.setLevel(self.level)

        # Set up console handler if requested
        if self.to_console:
            self._console_handler = logging.StreamHandler()
            self._console_handler.setLevel(self.level)
            self._console_handler.setFormatter(formatter)
            self._root_logger.addHandler(self._console_handler)

        # Set up file handler if requested
        if self.to_file:
            self._file_handler = logging.FileHandler(self.to_file, mode='a')
            self._file_handler.setLevel(self.level)
            self._file_handler.setFormatter(formatter)
            self._root_logger.addHandler(self._file_handler)

    def get_logs(self) -> str:
        """
Retrieve all captured logs from memory as a string.

Returns
-------
str
    Complete captured log output from in-memory buffer.

Examples
--------
>>> import logging
>>> with LogCapture(to_console=False) as capture:
...     logging.warning('sample message')
...     captured = 'sample message' in capture.get_logs()
>>> captured
True
"""
        return self.stream.getvalue()

    def clear(self) -> None:
        """Clear captured logs from memory buffer.

Examples
--------
>>> import logging
>>> with LogCapture(to_console=False) as capture:
...     logging.warning('old message')
...     capture.clear()
...     capture.get_logs()
''
"""
        self.stream.truncate(0)
        self.stream.seek(0)

    def stop(self) -> None:
        """Stop capturing logs and remove handlers.

Examples
--------
>>> capture = LogCapture(to_console=False)
>>> capture.stop()
"""
        if self._memory_handler:
            self._root_logger.removeHandler(self._memory_handler)
        if self._console_handler:
            self._root_logger.removeHandler(self._console_handler)
        if self._file_handler:
            self._file_handler.close()
            self._root_logger.removeHandler(self._file_handler)
        self._root_logger.setLevel(self._original_level)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()


def start_logging(log_level: int | str | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = logging.INFO,
                  to_console: bool = True,
                  to_file: str | None = None) -> LogCapture:
    """
Configure mlchem pipeline logging with optional file persistence.

Convenience function to set up logging. Logs are emitted to console by
default; optionally also to a file for later inspection.

Parameters
----------
log_level : int or str, default='INFO'
    Logging level threshold (e.g., 'INFO', 'DEBUG', 'WARNING').

to_console : bool, default=True
    Whether to print logs to console.

to_file : str or None, default=None
    Optional file path to write logs to.

Returns
-------
LogCapture
    Log manager object with .get_logs() for memory inspection.

Examples
--------
>>> # Log to console + file
>>> logs = start_logging(to_file='pipeline.log')
>>> sfs.fit(X_train, y_train, X_test, y_test)
>>> print(logs.get_logs())  # Inspect memory

>>> # Log only to file (silent)
>>> logs = start_logging(to_console=False, to_file='pipeline.log')
>>> sfs.fit(X_train, y_train, X_test, y_test)
"""
    return LogCapture(log_level=log_level, to_console=to_console, to_file=to_file)


def validate_task_type(task_type: str) -> str:
    """
Validate task type used by ML helpers.

Parameters
----------
task_type : str
    Task kind.

Returns
-------
str
    The validated task type.

Raises
------
ValueError
    If task_type is not 'classification' or 'regression'.

Examples
--------
>>> validate_task_type('classification')
'classification'
"""

    if task_type not in ('classification', 'regression'):
        raise ValueError("'task_type' must be either 'classification' or 'regression'.")
    return task_type


def validate_dataframe_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[Any],
    dataframe_name: str = 'dataframe',
) -> None:
    """Raise a clear error when a DataFrame lacks required columns.

    Examples
    --------
    >>> import pandas as pd
    >>> validate_dataframe_columns(pd.DataFrame({'feature': [1]}), ['feature'])
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"'{dataframe_name}' must be a pandas DataFrame.")
    if isinstance(required_columns, (str, bytes)):
        raise TypeError("'required_columns' must be an iterable of column names, not a string.")

    missing_columns = list(dict.fromkeys(
        column for column in required_columns if column not in dataframe.columns
    ))
    if missing_columns:
        raise KeyError(
            f"Columns not found in {dataframe_name}: {missing_columns}"
        )


def resolve_n_jobs(n_jobs: int) -> int:
    """
Normalize n_jobs values used by parallel wrappers.

Parameters
----------
n_jobs : int
    Requested number of workers. -1 means all CPUs.

Returns
-------
int
    Resolved positive worker count.

Raises
------
ValueError
    If n_jobs is not -1 or a positive integer.

Examples
--------
>>> resolve_n_jobs(2)
2
"""

    import os

    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < -1 or n_jobs == 0:
        raise ValueError("'n_jobs' must be -1 or a positive integer.")
    return n_jobs


def suppress_warnings() -> None:
    """
Suppress all warnings in the current Python session.

Returns
-------
None

Examples
--------
>>> import warnings
>>> with warnings.catch_warnings(record=True) as caught:
...     suppress_warnings()
...     warnings.warn('suppressed', UserWarning)
>>> len(caught)
0
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

Examples
--------
>>> standardise_path('C:\\\\Users\\\\User\\\\Documents')
'C:/Users/User/Documents'
"""

    return path.replace("\\", "/")




def generate_cartesian_product(*lists: Sequence) -> list[list]:
    """
Generate the Cartesian product of multiple lists.

Produces all combinations where each combination contains exactly one element
from each input list. For example, given lists A, B, and C, returns all tuples
(a, b, c) where a ∈ A, b ∈ B, c ∈ C.

Parameters
----------
*lists : sequence
    Two or more non-string sequences to combine.

Returns
-------
list of list
    List of combinations, where each combination is a list containing
    one element from each input list.

Raises
------
ValueError
    If fewer than two lists are provided, or if any list is empty.

TypeError
    If any argument is not a non-string sequence.

Examples
--------
>>> generate_cartesian_product([1, 2], ['a', 'b'])
[[1, 'a'], [1, 'b'], [2, 'a'], [2, 'b']]

>>> generate_cartesian_product([True, False], ['x', 'y'], [1])
[[True, 'x', 1], [True, 'y', 1], [False, 'x', 1], [False, 'y', 1]]
"""
    from itertools import product

    # Validate input
    if not lists or len(lists) < 2:
        raise ValueError("At least two lists are required.")
    
    for idx, lst in enumerate(lists, start=1):
        if not isinstance(lst, Sequence) or isinstance(lst, (str, bytes)):
            raise TypeError(f"Argument {idx} must be a non-string sequence.")
        if len(lst) == 0:
            raise ValueError(f"List {idx} is empty; cannot generate combinations.")
    
    # Generate all combinations and convert tuples to lists
    return [list(combo) for combo in product(*lists)]


def disable_estimator_parallelization(estimator):
    """
Disable internal parallelization in sklearn estimators to avoid nested parallelization.

When wrapper-level parallelization (ThreadPoolExecutor) is used, individual
estimators should disable internal job parallelism to prevent nested
parallelism warnings and avoid thread conflicts.

Parameters
----------
estimator : object
    A scikit-learn estimator object.

Returns
-------
object
    The estimator with internal parallelization disabled if applicable,
    otherwise unchanged.

Examples
--------
>>> from sklearn.ensemble import RandomForestClassifier
>>> estimator = disable_estimator_parallelization(
...     RandomForestClassifier(n_jobs=4)
... )
>>> estimator.n_jobs is None
True
"""
    if hasattr(estimator, 'n_jobs'):
        try:
            estimator.set_params(n_jobs=None)
        except (ValueError, TypeError):
            try:
                estimator.set_params(n_jobs=1)
            except (ValueError, TypeError):
                pass  # Estimator doesn't support set_params or n_jobs parameter
    return estimator


def generate_combination_cascade(
    elements: Collection,
    n: int
) -> list[list]:
    """
Generate all combinations of elements from size 1 to n.

Parameters
----------
elements : collection
    Re-iterable collection of elements to combine.

n : int
    Maximum size of combinations.

Returns
-------
list of list
    List of combinations of elements.

Examples
--------
>>> generate_combination_cascade(['a', 'b', 'c'], 2)
[['a'], ['b'], ['c'], ['a', 'b'], ['a', 'c'], ['b', 'c']]
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

Examples
--------
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

Examples
--------
>>> from contextlib import redirect_stdout
>>> from io import StringIO
>>> output = StringIO()
>>> with redirect_stdout(output):
...     loadingbar(1, 3, 10)
>>> '001/003' in output.getvalue()
True
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

Examples
--------
>>> create_progressive_column_names('col', 3)
['col1', 'col2', 'col3']
"""

    return [f"{serial_name}{i + 1}" for i in range(n)]


def try_except(
    func: Callable[[], Any],
    exc: Any = None,
    exceptions: type[BaseException] | tuple[type[BaseException], ...] = Exception,
) -> Any:
    """
Execute a function and return its result or a fallback value on exception.

Parameters
----------
func : callable
    Function to execute.

exc : any, optional
    Value to return if an exception occurs. Default is None.

exceptions : exception type or tuple of exception types, optional
    Exception types to catch. Default catches Exception.

Returns
-------
any
    Result of the function or fallback value.

Raises
------
TypeError
    If `exceptions` is not an exception class or tuple of exception classes.

Examples
--------
>>> try_except(lambda: 1 / 0, exc='error')
'error'
>>> try_except(lambda: 1 / 1)
1.0
"""

    if isinstance(exceptions, tuple):
        if not exceptions or not all(
            isinstance(ex, type) and issubclass(ex, BaseException)
            for ex in exceptions
        ):
            raise TypeError(
                "'exceptions' must be an exception type or tuple of exception types."
            )
    elif not (isinstance(exceptions, type) and issubclass(exceptions, BaseException)):
        raise TypeError(
            "'exceptions' must be an exception type or tuple of exception types."
        )

    try:
        return func()
    except exceptions:
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

Examples
--------
>>> find_all_occurrences('test test test', 'test')
[0, 5, 10]
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

Examples
--------
>>> reset_string('Hello, World!')
'helloworld'
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

Examples
--------
>>> insert_string_piece('Hello World', 'Beautiful ', 5)
'Hello Beautiful World'
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

Examples
--------
>>> flatten([1, [2, 3], [[4, 5], 6]])
(1, 2, 3, 4, 5, 6)
"""

    if isinstance(args, (str, bytes, bytearray)):
        return (args,)

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

Examples
--------
>>> process_custom_string('Nc1ccccc1N', 'c1ccccc1', ['CYC'])
';CYC;'
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

Examples
--------
>>> sort_list_by_other_list(['apple', 'banana', 'cherry'], [3, -7, 2])
(['banana', 'apple', 'cherry'], [-7, 3, 2])
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

Examples
--------
>>> merge_dicts_with_duplicates({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
{'a': 1, 'b': 2, 'b_duplicate': 3, 'c': 4}
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

Examples
--------
>>> import pandas as pd
>>> df = add_inchi_to_dataframe(pd.DataFrame({'SMILES': ['CCO']}), 1, 'SMILES')
>>> df.columns.tolist()
['SMILES', 'INCHI']
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

Raises
------
KeyError
    If `column_name` does not exist in `df`.

ValueError
    If `keep` is not 'first' or 'last'.

Examples
--------
>>> import pandas as pd
>>> df = pd.DataFrame({'A': [1, 2, 2, 3]})
>>> cleaned, duplicates = identify_df_duplicates(df, 'A')
>>> len(cleaned), len(duplicates)
(3, 1)
"""

    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame.")

    if keep not in ('first', 'last'):
        raise ValueError("'keep' must be either 'first' or 'last'.")

    cleaned_df = df.drop_duplicates(subset=column_name, keep=keep)
    duplicates_removed = df.loc[~df.index.isin(cleaned_df.index)]
    return cleaned_df, duplicates_removed


_CLASS_FOLDER_PREFIX = '__mlchem_label__'


def _class_structure_folder_names(classnames) -> list[str]:
    """Map class labels to unique, portable ASCII folder components.

    Short, collision-free labels from a conservative ASCII allowlist are
    kept verbatim. Other labels get a reserved prefix, readable slug, and
    stable digest so generated names remain distinct from literal labels.
    """
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$',
        *(f'COM{i}' for i in range(1, 10)),
        *(f'LPT{i}' for i in range(1, 10)),
    }

    def can_use_label_directly(label) -> bool:
        if not isinstance(label, str):
            return False
        if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9._-]{0,62}', label):
            return False
        if label.endswith('.') or label.casefold().startswith(
            _CLASS_FOLDER_PREFIX.casefold()
        ):
            return False
        return label.split('.', 1)[0].upper() not in reserved_names

    direct_candidates = {}
    for index, label in enumerate(classnames):
        if can_use_label_directly(label):
            direct_candidates.setdefault(label.casefold(), []).append(index)

    direct_indices = {
        indices[0] for indices in direct_candidates.values()
        if len(indices) == 1
    }
    folder_names = []
    for index, label in enumerate(classnames):
        if index in direct_indices:
            folder_names.append(label)
            continue

        label_type = type(label)
        identity = (
            f'{label_type.__module__}.{label_type.__qualname__}:{label!r}'
        )
        digest = hashlib.sha256(
            identity.encode('utf-8', errors='surrogatepass')
        ).hexdigest()
        slug = re.sub(r'[^A-Za-z0-9_-]+', '_', str(label)).strip('_')[:32]
        folder_names.append(
            f'{_CLASS_FOLDER_PREFIX}{slug or "value"}__{digest}'
        )

    normalized_names = [name.casefold() for name in folder_names]
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError(
            'Class labels do not have unique stable representations for '
            'structure-folder naming.'
        )
    return folder_names


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

Notes
-----
The folder is created if needed but is not cleared. Existing files not
replaced by the generated numbered PNGs are preserved.

Returns
-------
None

Examples
--------
>>> import pandas as pd
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> with TemporaryDirectory() as temp_dir:
...     output_dir = Path(temp_dir) / 'structures'
...     create_structure_files(
...         pd.DataFrame({'SMILES': ['CCO']}), 'SMILES', str(output_dir)
...     )
...     len(list(output_dir.iterdir()))
1
"""


    import os
    from mlchem.chem.manipulation import create_molecule
    from rdkit import Chem
    from rdkit.Chem import Draw

    # Keep unrelated files in an existing output folder intact.
    os.makedirs(folder_name, exist_ok=True)

    for i, s in enumerate(df[structure_column_name]):
        try:
            mol = create_molecule(s)
            filename = os.path.join(folder_name, f'{i}.png')
            Draw.MolToFile(mol, filename)
        except Exception:
            # If molecule does not get read properly,
            # create a placeholder image
            filename = os.path.join(folder_name, f'{i}.png')
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

Examples
--------
>>> import pandas as pd
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> with TemporaryDirectory() as temp_dir:
...     (Path(temp_dir) / 'molecule.sdf').touch()
...     df = prepare_dataframe(pd.DataFrame({'A': [1], 'B': [2]}), temp_dir)
...     Path(df.loc[0, 'MOLFILE']).name
'molecule.sdf'
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

Raises
------
KeyError
    If required core columns are missing from the DataFrame.

Examples
--------
>>> import pandas as pd
>>> df = pd.DataFrame({
...     'DIM_1': [1], 'DIM_2': [2], 'SMILES': ['CCO'], 'MOLFILE': ['mol.png'],
...     'NAME': ['ethanol'], 'NAME_SHORT': ['eth']
... })
>>> table = prepare_datatable(df)
>>> len(table.columns)
6
"""

    from bokeh.models import ColumnDataSource, TableColumn

    required_columns = ['DIM_1', 'DIM_2', 'SMILES', 'MOLFILE', 'NAME', 'NAME_SHORT']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    source = ColumnDataSource(df)
    columns = [
        TableColumn(field='DIM_1', title='DIM_1'),
        TableColumn(field='DIM_2', title='DIM_2'),
        TableColumn(field='SMILES', title='SMILES'),
        TableColumn(field='MOLFILE', title='MOLFILE'),
        TableColumn(field='NAME', title='NAME'),
        TableColumn(field='NAME_SHORT', title='NAME_SHORT'),
    ]

    if 'METADATA' in df.columns:
        columns.append(TableColumn(field='METADATA', title='METADATA'))

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

Raises
------
ValueError
    If size is negative.

Examples
--------
>>> compute_alpha(800)
0.5
"""

    if size < 0:
        raise ValueError("'size' must be a non-negative integer.")

    if size < 100:
        alpha = 0.95
    elif size < 300:
        alpha = 0.9
    elif size < 800:
        alpha = 0.8
    elif size < 2000:
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

Raises
------
ValueError
    If either size is negative or both sizes are zero.

Examples
--------
>>> size_ratio(1, 3)
0.875
"""

    if size1 < 0 or size2 < 0:
        raise ValueError("'size1' and 'size2' must be non-negative.")

    if size1 + size2 == 0:
        raise ValueError("'size1' and 'size2' cannot both be zero.")

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

Examples
--------
>>> from bokeh.models import DataTable
>>> from bokeh.plotting import figure
>>> plot = figure()
>>> _ = plot.line([0, 1], [0, 1])
>>> bokeh_plot(plot, ['sample'], {'sample': DataTable()})
"""

    import bokeh.plotting as bp

    tables = [dict_datatables[classname] for classname in classnames]
    bp.show(bp.column(p, *tables))


def add_encoding_legend(
    plot: figure,
    colour_entries: Sequence[tuple[str, str]],
    shape_entries: Sequence[tuple[str, str]],
    text_entries: Sequence[str],
    proxy_x: float,
    proxy_y: float,
    proxy_size: int | float = 10,
    proxy_alpha: float = 1.0,
    colour_proxy_marker: str = 'circle',
    shape_proxy_colour: str = 'grey',
    ncols: int = 2,
):
    """Add an encoding legend with isolated proxy glyphs and text-only entries.

    Colour and shape proxies are invisible plot renderers located at a real
    data coordinate; they draw only in the legend. Text entries use
    ``LegendItem(renderers=[])`` and therefore have no glyph swatch.
    """
    from bokeh.models import Legend, LegendItem

    items = []
    for label, colour in colour_entries:
        renderer = plot.scatter(
            x=[proxy_x],
            y=[proxy_y],
            marker=colour_proxy_marker,
            size=proxy_size,
            color=colour,
            alpha=proxy_alpha,
            line_color='grey',
            line_alpha=proxy_alpha,
            visible=False,
            name='chemical-space-colour-legend-proxy',
        )
        items.append(LegendItem(label=label, renderers=[renderer], index=0))

    for label, shape in shape_entries:
        renderer = plot.scatter(
            x=[proxy_x],
            y=[proxy_y],
            marker=shape,
            size=proxy_size,
            color=shape_proxy_colour,
            alpha=proxy_alpha,
            line_color=shape_proxy_colour,
            line_alpha=proxy_alpha,
            visible=False,
            name='chemical-space-shape-legend-proxy',
        )
        items.append(LegendItem(label=label, renderers=[renderer], index=0))

    items.extend(
        LegendItem(label=label, renderers=[])
        for label in text_entries
    )
    if not items:
        return None

    legend = Legend(
        items=items,
        ncols=ncols,
        glyph_width=16,
        glyph_height=16,
        label_standoff=4,
        spacing=2,
        padding=4,
        margin=4,
    )
    plot.add_layout(legend)
    return legend


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

Examples
--------
>>> create_mask(np.array([1, 2, 3, 4, 5]), 1.9, 4.1)
array([False,  True,  True,  True, False])
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

Examples
--------
>>> assign_sign(-3)
'-'
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

Examples
--------
>>> normalise_iterable([2, 4])
[0.5, 1.0]
"""

    import math

    values_list = list(values)
    if not values_list:
        return []

    max_abs_weight = max(math.fabs(v) for v in values_list)

    if max_abs_weight > 0:
        return [v / max_abs_weight for v in values_list]
    return values_list


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

Examples
--------
>>> import pandas as pd
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> with TemporaryDirectory() as temp_dir:
...     output = Path(temp_dir) / 'data.xlsx'
...     dfs_to_excel(str(output), [pd.DataFrame({'value': [1]})], ['data'])
...     output.exists()
True
"""

    with pd.ExcelWriter(file_name) as writer:
        for desc_df, sheet_name in zip(dfs, sheet_names):
            desc_df.to_excel(writer, sheet_name=sheet_name, index=True)
