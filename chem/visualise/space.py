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

import logging
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class ChemicalSpace:
    """
A module to compute and visualise datasets in a compressed embedded space.

This class handles descriptor processing, dimensionality reduction,
and preparation of molecular data for interactive visualisation.

Parameters
----------
data : pd.DataFrame
    A DataFrame with 'SMILES', 'NAME', 'CLASS', and optional extra columns
    for independent plot encodings. ``prepare()`` aligns them by SMILES.
needs_cleaning : bool, optional
    Whether the data requires cleaning. Default is False.
df_descriptors : pd.DataFrame, optional
    A DataFrame with SMILES as index and descriptor columns.
metadata_exists : bool, optional
    Enable backwards-compatible handling of the special 'METADATA' column.
    This is not required for other optional columns or their plot encodings.
    Default is False.
metadata_categorical : bool, optional
    Treat 'METADATA' as categorical for legacy marker shapes. This only affects
    the legacy shape behavior when no 'shape_by' mapping is supplied.
    Default is False.

Raises
------
ValueError
    If the input `data` does not have the expected column structure.
"""


    def __init__(self,
                 data: pd.core.frame.DataFrame,
                 needs_cleaning: bool = False,
                 df_descriptors: pd.core.frame.DataFrame = None,
                 metadata_exists: bool = False,
                 metadata_categorical: bool = False):
        """
Initialise a ChemicalSpace object for visualising molecular datasets
in a compressed embedded space.

This constructor sets up the molecular data, descriptor matrix,
legacy metadata options, and validates the structure of the input
data and loads default Bokeh visualisation settings.

Parameters
----------
data : pd.DataFrame
    A DataFrame with required 'SMILES', 'NAME', and 'CLASS' columns. Any
    additional columns are available for independent plot encodings after
    ``prepare()``. 'METADATA' remains a special legacy column; other optional
    columns do not require metadata flags.
needs_cleaning : bool, optional
    Whether the data requires cleaning before processing. Default is False.
df_descriptors : pd.DataFrame, optional
    A DataFrame with SMILES as index and molecular descriptors as columns.
metadata_exists : bool, optional
    Enable backwards-compatible handling of the special 'METADATA' column;
    when True, that column is required in ``data``. It is not needed to use
    other optional columns with ``plot()`` encodings. Default is False.
metadata_categorical : bool, optional
    Treat 'METADATA' as categorical for legacy marker shapes. It only affects
    plotting when ``shape_by`` is omitted; explicit ``shape_by`` can select
    any prepared column without enabling either metadata flag. Default is False.

Raises
------
ValueError
    If the `data` DataFrame does not have the expected column structure.

Notes
-----
If `df_descriptors` is not provided, a warning is logged to remind the
user to supply descriptors before visualisation.

The metadata flags predate independent dataframe-column plot encodings and
remain for compatibility with callers that use the special `METADATA`
column and its legacy marker-shape behavior. They do not designate which
columns are available for the new `*_by` plot arguments.

Examples
--------
>>> cs = ChemicalSpace(data=df, df_descriptors=desc_df, metadata_exists=True)
"""

        

        from mlchem.importables import bokeh_dictionary, bokeh_tooltips

        self.data = data
        self.needs_cleaning = needs_cleaning
        self.df_descriptors = df_descriptors
        self.metadata_exists = metadata_exists
        self.metadata_categorical = metadata_categorical
        self.bokeh_dictionary = dict(bokeh_dictionary)
        self.bokeh_tooltips = bokeh_tooltips

        required = ['SMILES', 'NAME', 'CLASS']
        if self.metadata_exists:
            required.append('METADATA')
        missing = [column for column in required if column not in data.columns]
        if missing or not data.columns.is_unique:
            raise ValueError(
                f"'data' must have unique columns including {required}; missing: {missing}."
            )
        reserved = {'DIM_1', 'DIM_2', 'MOLFILE', 'NAME_SHORT'}.intersection(data.columns)
        if reserved:
            raise ValueError(f"'data' contains reserved prepared columns: {sorted(reserved)}.")

        if self.df_descriptors is None:
            logger.warning(
                'The given data have no descriptors available. To visualise '
                'the chemical space, provide a descriptor dataframe or use '
                'the mlchem.chem.Calculator.descriptors.py module.'
            )

    def process(self,
                diversity_filter: float | None,
                collinearity_filter: float | None,
                standardise: bool = True,
                skip_binary_columns: bool = True):
        """
Process the descriptor data by applying diversity and collinearity filters,
and optionally standardising the data.

Parameters
----------
diversity_filter : float or None
    A float between 0 and 1. Higher values apply stricter filtering
    to remove less diverse descriptors.
collinearity_filter : float or None
    A float between 0 and 1. Higher values apply looser filtering
    to remove more collinear descriptors.
standardise : bool, optional
    Whether to standardise the filtered descriptor data. Default is True.
skip_binary_columns : bool, optional
    Whether binary 0/1 columns are left unchanged during standardisation.
    Default is True.

Returns
-------
None
    Updates the instance with processed descriptor data.

Raises
------
ValueError
    If `diversity_filter` or `collinearity_filter` are outside [0, 1).

Examples
--------
>>> chem_space = ChemicalSpace(data, df_descriptors)
>>> chem_space.process(
...     diversity_filter=0.5,
...     collinearity_filter=0.3,
...     skip_binary_columns=False,
... )
"""


        from mlchem.ml.feature_selection import filters
        from mlchem.ml.preprocessing.scaling import scale_df_standard

        if diversity_filter is not None and not (0. <= diversity_filter < 1.):
            raise ValueError("'diversity_filter' argument must be a float 0 <= x < 1.")
        if collinearity_filter is not None and not (0. < collinearity_filter <= 1.):
            raise ValueError("'collinearity_filter' argument must be a float 0 < x <= 1.")
          
        if diversity_filter:
          logger.info("Before filtering: %s", self.df_descriptors.shape)
          df_filtered_1 = filters.diversity_filter(self.df_descriptors,
                                                   diversity_filter)
        else:
            df_filtered_1 = self.df_descriptors     # do nothing

        logger.info("After diversity filter: %s", df_filtered_1.shape)

        if collinearity_filter:
            df_filtered_2 = filters.collinearity_filter(df_filtered_1,
                                                        collinearity_filter)
        else:
            df_filtered_2 = df_filtered_1
        logger.info("After collinearity filter: %s", df_filtered_2.shape)

        if standardise is True:
            self.df_processed, self.scaler = scale_df_standard(
                df_filtered_2,
                skip_binary_columns=skip_binary_columns)
        else:
            self.df_processed = df_filtered_2

    def prepare(self, df_compressed: pd.core.frame.DataFrame):
      """
Prepare the compressed dataframe for visualisation.

This method adds molecular structure files, names, metadata, and all extra
source columns to the compressed coordinates, aligned by SMILES.

Parameters
----------
df_compressed : pd.DataFrame
    A DataFrame with two columns ['DIM_1', 'DIM_2'] representing
    the compressed coordinates of the molecules.

Returns
-------
None
    Updates the instance with prepared data for visualisation.

Raises
------
ValueError
    If `df_compressed` does not have exactly two columns named
    ['DIM_1', 'DIM_2'].
ValueError
    If the index of `df_compressed` or `df_processed` is not a
    series of SMILES strings.
ValueError
    If SMILES keys are duplicated, compressed rows have no processed/data
    match, or class labels are missing or cannot be sorted.

Examples
--------
>>> chem_space.prepare(df_compressed)

The compressed dataframe's SMILES index is the alignment key. Its row order
and coordinates are preserved; input and processed SMILES must be unique.
Class labels must be sortable. Structure images are generated in stable
SMILES order within each class, saved beneath a folder derived from that
class label, and mapped back by SMILES, independent of directory enumeration
and file modification times. Portable ASCII labels use their names directly;
other labels use deterministic encoded folder names, available in
``self.class_structure_folders``. The dedicated ``.mlchem_structures`` root
keeps generated files separate from existing working-directory folders.
``plot()`` renders existing ``MOLFILE`` paths and does not generate images.
Spaces, punctuation, reserved Windows names, and case-insensitive collisions
require encoding for portable, unambiguous paths. The readable slug plus
stable digest prevents different labels from collapsing to the same folder;
display labels are unchanged. Always use the prepared ``MOLFILE`` column
rather than reconstructing paths from class labels.
When legacy categorical metadata shapes are enabled, ``metadata_names`` is
derived only from the rows in this compressed dataframe; missing metadata
values are ignored here so explicit ``shape_by`` encodings can still be used.
Sortable metadata categories retain their sorted order; incomparable
categories use their first-appearance order. The legacy metadata shape path
reports an error if plotted rows have missing metadata, since those rows
cannot be assigned a shape.
"""


      import os
      from mlchem.helper import (
          _class_structure_folder_names,
          create_structure_files,
      )

      if list(df_compressed.columns) != ['DIM_1', 'DIM_2']:
          raise ValueError("'df_compressed' must have only 2 columns: ['DIM_1','DIM_2'].")

      if not all(isinstance(item, str) for item in df_compressed.index):
          raise ValueError("Index of df_compressed must be a series of SMILES strings.")
      if not all(isinstance(item, str) for item in self.df_processed.index):
          raise ValueError("Index of df_processed must be a series of SMILES strings.")
      if df_compressed.index.has_duplicates:
          raise ValueError("'df_compressed' must have unique SMILES keys for row alignment.")
      if self.df_processed.index.has_duplicates:
          raise ValueError("'df_processed' must have unique SMILES keys for row alignment.")

      missing_processed = df_compressed.index.difference(self.df_processed.index)
      if not missing_processed.empty:
          raise ValueError(
              "Every SMILES in 'df_compressed' must have a matching row in "
              "'df_processed'; missing: %s." % list(missing_processed)
          )

      source_rows = self.data[self.data.SMILES.isin(df_compressed.index)]
      if source_rows.SMILES.duplicated().any():
          duplicated = source_rows.loc[
              source_rows.SMILES.duplicated(keep=False), 'SMILES'
          ].unique()
          raise ValueError(
              "Rows in 'data' must have unique SMILES keys for row alignment; "
              "duplicates: %s." % list(duplicated)
          )
      missing_data = df_compressed.index.difference(source_rows.SMILES)
      if not missing_data.empty:
          raise ValueError(
              "Every SMILES in 'df_compressed' must have a matching row in "
              "'data'; missing: %s." % list(missing_data)
          )
      if source_rows.CLASS.isna().any():
          raise ValueError("'CLASS' labels must not be missing for row alignment.")

      try:
          self.classes = source_rows.set_index('SMILES').reindex(
              df_compressed.index
          ).CLASS.to_numpy()
          self.classnames = np.unique(self.classes)
      except TypeError as error:
          raise ValueError(
              "'CLASS' labels must be mutually sortable to establish class order."
          ) from error

      if self.metadata_categorical and self.metadata_exists:
          metadata_values = source_rows.METADATA.dropna()
          try:
              self.metadata_names = np.unique(metadata_values)
          except TypeError:
              self.metadata_names = pd.unique(metadata_values)

      prepared = df_compressed.copy()
      prepared.index = pd.Index(prepared.index, name='SMILES')
      prepared = prepared.reset_index()
      source_by_smiles = source_rows.set_index('SMILES')
      prepared['CLASS'] = prepared.SMILES.map(source_by_smiles.CLASS)
      prepared['NAME'] = prepared.SMILES.map(source_by_smiles.NAME)
      for column in source_by_smiles.columns:
          if column not in ('CLASS', 'NAME'):
              prepared[column] = prepared.SMILES.map(source_by_smiles[column])

      self.molfiles = []
      self.names = []
      self.metadata = []
      molfile_by_smiles = {}
      name_by_smiles = {}
      metadata_by_smiles = {}
      structure_root = '.mlchem_structures'
      structure_folder_names = _class_structure_folder_names(self.classnames)
      self.class_structure_folders = dict(zip(
          self.classnames,
          (os.path.join(structure_root, name) for name in structure_folder_names),
      ))

      for class_name in self.classnames:
          class_rows = prepared.loc[
              prepared.CLASS == class_name
          ].sort_values('SMILES', kind='mergesort')
          smiles = class_rows.SMILES.tolist()
          names = class_rows.NAME.tolist()
          structure_folder = self.class_structure_folders[class_name]
          create_structure_files(
              df=class_rows[['SMILES']],
              structure_column_name='SMILES',
              folder_name=structure_folder,
          )
          molfiles = [
              os.path.join(structure_folder, '%d.png' % i).replace(
                  os.sep, '/'
              )
              for i in range(len(smiles))
          ]
          molfile_by_smiles.update(zip(smiles, molfiles))
          name_by_smiles.update(zip(smiles, names))
          self.molfiles.append(np.asarray(molfiles))
          self.names.append(names)

          if self.metadata_exists:
              metadata = class_rows.METADATA.tolist()
              metadata_by_smiles.update(zip(smiles, metadata))
              self.metadata.append(metadata)

      prepared['MOLFILE'] = prepared.SMILES.map(molfile_by_smiles)
      prepared['NAME'] = prepared.SMILES.map(name_by_smiles)
      prepared['NAME_SHORT'] = prepared.NAME.map(
          lambda name: name[:20] if isinstance(name, str) else str(name)[:20]
      )

      image_indices = []
      for molfile in prepared.MOLFILE:
          stem = os.path.splitext(os.path.basename(molfile))[0]
          try:
              image_indices.append(int(stem))
          except ValueError:
              image_indices.append(stem)
      prepared.index = image_indices

      if self.metadata_exists:
          prepared['METADATA'] = prepared.SMILES.map(metadata_by_smiles)
      df_compressed.drop(columns=list(df_compressed.columns), inplace=True)
      df_compressed.index = prepared.index
      for column in prepared.columns:
          df_compressed[column] = prepared[column].to_numpy()
      self.df_compressed = df_compressed

    def update_bokeh_options(self, **args) -> None:
        """
Update the Bokeh plot options.

This method allows customisation of various Bokeh plot parameters
for visualisations. Users can pass keyword arguments to set options
such as title properties, legend properties, and axis properties. Only
documented keys are accepted; unknown keys raise ``ValueError`` instead of
being stored without affecting the plot. Updated settings persist on this
instance until changed again or reset with ``reset_bokeh_options()``. They
are used by ``plot()`` unless an option is explicitly supplied to that call.

Parameters
----------
title_location : str, optional
    Location of the title. Options: 'above', 'below', 'left', 'right'.
title_fontsize : str, optional
    Font size of the title (e.g., '25px').
title_align : str, optional
    Alignment of the title. Options: 'left', 'center', 'right'.
title_background_fill_colour : str, optional
    Background colour of the title.
title_text_colour : str, optional
    Text colour of the title.
legend_location : str, optional
    Location of the legend. Options: 'top_left', 'top_center',
    'top_right', 'center_right', 'bottom_right', 'bottom_center',
    'bottom_left', 'center_left', 'center'.
legend_title : str, optional
    Title of the legend.
legend_label_text_font : str, optional
    Font of the legend labels (e.g., 'times').
legend_label_text_font_style : str, optional
    Font style of the legend labels (e.g., 'italic').
legend_label_text_colour : str, optional
    Text colour of the legend labels.
legend_border_line_width : int, optional
    Line width of the legend border.
legend_border_line_colour : str, optional
    Line colour of the legend border.
legend_border_line_alpha : float, optional
    Transparency of the legend border (0 to 1).
legend_background_fill_colour : str, optional
    Background colour of the legend.
legend_background_fill_alpha : float, optional
    Transparency of the legend background (0 to 1).
legend_proxy_size : float, optional
    Uniform marker size for colour and shape legend swatches. Default is 10.
legend_proxy_alpha : float, optional
    Uniform visible opacity for colour and shape legend swatches. Default is 1.
legend_colour_proxy_marker : str, optional
    Neutral marker used by all categorical colour swatches. Default is 'circle'.
legend_shape_proxy_colour : str, optional
    Neutral colour used by all categorical shape swatches. Default is 'grey'.
legend_ncols : int, optional
    Number of compact columns in the encoding legend. Default is 2.
xaxis_label : str, optional
    Label of the x-axis.
xaxis_line_width : int, optional
    Line width of the x-axis.
xaxis_line_colour : str, optional
    Line colour of the x-axis.
xaxis_major_label_text_colour : str, optional
    Text colour of the x-axis major labels.
xaxis_major_label_orientation : str, optional
    Orientation of the x-axis major labels. Options: 'horizontal', 'vertical'.
yaxis_label : str, optional
    Label of the y-axis.
yaxis_line_width : int, optional
    Line width of the y-axis.
yaxis_line_colour : str, optional
    Line colour of the y-axis.
yaxis_major_label_text_colour : str, optional
    Text colour of the y-axis major labels.
yaxis_major_label_orientation : str, optional
    Orientation of the y-axis major labels. Options: 'horizontal', 'vertical'.
axis_minor_tick_in : int, optional
    Length of the minor ticks inward.
axis_minor_tick_out : int, optional
    Length of the minor ticks outward.

Returns
-------
None

Examples
--------
>>> plot.update_bokeh_options(
...     title_location='above',
...     title_fontsize='20px',
...     legend_location='top_right',
...     xaxis_label='PC1',
...     yaxis_label='PC2'
... )
"""

        unknown = set(args).difference(self.bokeh_dictionary)
        if unknown:
            raise ValueError(
                f"Unknown Bokeh option(s): {sorted(unknown)}. "
                f"Supported options are: {sorted(self.bokeh_dictionary)}."
            )
        if 'title_fontsize' in args and (
            not isinstance(args['title_fontsize'], str)
            or not args['title_fontsize'].strip()
        ):
            raise ValueError("'title_fontsize' must be a non-empty Bokeh font-size string.")
        self.bokeh_dictionary.update(args)

    def reset_bokeh_options(self):
        """
Reset Bokeh plot options to their default values.

Returns
-------
None

Examples
--------
>>> chem_space.reset_bokeh_options()
"""


        from mlchem.importables import bokeh_dictionary
        self.bokeh_dictionary = dict(bokeh_dictionary)

    def reset_bokeh_tooltips(self):
        """
Reset Bokeh tooltips to their default values.

Returns
-------
None

Examples
--------
>>> chem_space.reset_bokeh_tooltips()
"""

        from mlchem.importables import bokeh_tooltips

        self.bokeh_tooltips = bokeh_tooltips

    @staticmethod
    def _validate_encoding_values(encoding: str, values, argument: str) -> None:
        from bokeh.core.enums import MarkerType
        from bokeh.core.properties import Color

        for value in values:
            if encoding == 'colour':
                try:
                    Color().validate(value)
                except ValueError as error:
                    raise ValueError(f"'{argument}' contains an invalid colour: {value!r}.") from error
                if value is None:
                    raise ValueError(f"'{argument}' colours must not be missing.")
            elif encoding == 'shape':
                if not isinstance(value, str) or value not in MarkerType:
                    raise ValueError(f"'{argument}' contains an invalid marker shape: {value!r}.")
            elif (
                not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
                or not np.isfinite(value)
                or not (0 <= value <= 1 if encoding == 'opacity' else value > 0)
            ):
                bounds = 'in [0, 1]' if encoding == 'opacity' else 'positive'
                raise ValueError(f"'{argument}' values must be finite numbers {bounds}.")

    def _encoding_map(self, encoding: str, column: str | None, mapping, legacy,
                      continuous: bool = False):
        if column is None:
            if mapping is not None:
                raise ValueError(f"'{encoding}_map' requires '{encoding}_by'.")
            return None
        if legacy is not None:
            raise ValueError(f"Use either '{encoding}_list' or '{encoding}_by', not both.")
        if not isinstance(column, str) or column not in self.df_compressed.columns:
            raise ValueError(f"Unknown '{encoding}_by' column: {column!r}.")
        series = self.df_compressed[column]
        if series.isna().any():
            raise ValueError(f"'{encoding}_by' column {column!r} contains missing values.")
        if continuous:
            if mapping is not None:
                raise ValueError("'colour_continuous' conflicts with 'colour_map'.")
            if (
                not pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_bool_dtype(series)
                or pd.api.types.is_complex_dtype(series)
                or not np.isfinite(series.to_numpy(dtype=float)).all()
            ):
                raise ValueError("'colour_by' must contain finite numeric values for continuous colour.")
            return None
        if not isinstance(mapping, Mapping):
            raise ValueError(f"'{encoding}_by' requires a categorical '{encoding}_map'.")
        try:
            categories = list(pd.unique(series))
            missing = [category for category in categories if category not in mapping]
        except TypeError as error:
            raise ValueError(f"'{encoding}_by' categories must be hashable.") from error
        if missing:
            raise ValueError(f"'{encoding}_map' is missing categories: {missing!r}.")
        self._validate_encoding_values(encoding, mapping.values(), f'{encoding}_map')
        return mapping

    def plot(self,
             colour_list: list = None,
             shape_list: list = None,
             filename: str = '',
             title: str = '',
             title_fontsize: int | str | None = None,
             height: int = 650,
             width: int = int(650/0.75),
             marker_size: int = 10,
             save_html: bool = False,
             opacity_list: list[float] | None = None,
             marker_size_list: list[float] | None = None,
             *,
             colour_by: str | None = None,
             colour_map: Mapping | None = None,
             shape_by: str | None = None,
             shape_map: Mapping | None = None,
             opacity_by: str | None = None,
             opacity_map: Mapping | None = None,
             marker_size_by: str | None = None,
             marker_size_map: Mapping | None = None,
             colour_continuous: bool = False,
             colour_palette: Sequence[str] | None = None) -> None:
        """
Generate a 2D scatter plot of the chemical space using Bokeh.

This method visualises the compressed chemical space using a scatter plot,
with independently configurable colour, shape, opacity, and marker size.
Colour can be categorical or continuous, and the plot can optionally be saved
as an HTML file.

Parameters
----------
colour_list : list, optional
    List of colours for the plot markers. Each class will be assigned a
    colour in ``self.classnames`` order (sorted by ``prepare()``).
    If not provided, defaults to:
    ['Blue', 'Orange', 'Red', 'Black', 'Green', 'Cyan', 'Magenta', 'Yellow'].
shape_list : list, optional
    List of marker shapes. If not provided, defaults to:
    ['circle', 'triangle', 'square', 'star', 'diamond', 'cross', 'x', 'asterisk'].
    With categorical metadata, shapes follow ``self.metadata_names`` order.
filename : str, optional
    Name of the file to save the plot as (without extension).
title : str, optional
    Title of the plot.
title_fontsize : int or str, optional
    Font size of the plot title in pixels. An integer is interpreted as
    pixels; a string is passed to Bokeh as a CSS font size. If omitted,
    uses ``self.bokeh_dictionary['title_fontsize']`` (25px by default).
    An explicit ``plot(title_fontsize=...)`` overrides the configured value
    for that call.
height : int, optional
    Height of the plot in pixels. Default is 650.
width : int, optional
    Width of the plot in pixels. Default is 866 (650 / 0.75).
marker_size : int, optional
    Size of the plot markers when ``marker_size_list`` is omitted. Default is 10.
save_html : bool, optional
    Whether to save the plot as an HTML file. Default is False.
opacity_list : list of float, optional
    One finite opacity in [0, 1] per class, in ``self.classnames`` order,
    matching ``colour_list`` in both plotting branches. Explicit values
    replace automatic opacity, without scaling. If omitted, opacity uses
    ``compute_alpha(class_count) * size_ratio(class_count, remaining_count)``,
    with counts taken from ``self.df_compressed``.
marker_size_list : list of float, optional
    One finite positive marker size per class, in the same order as
    ``opacity_list``. Overrides ``marker_size``. With categorical metadata,
    all metadata subgroups within a class share its opacity and size;
    these lists do not follow ``shape_list``.
colour_by, shape_by, opacity_by, marker_size_by : str, optional
    Independently select any prepared dataframe column for each encoding.
    Categorical values require the corresponding ``*_map``. Missing column
    values or unmapped categories raise errors. A ``*_by`` argument conflicts
    with the legacy ``*_list`` for that same encoding only. Other encodings
    retain their legacy defaults. ``marker_size_map`` overrides scalar
    ``marker_size``, just as ``marker_size_list`` does.
colour_map, shape_map, opacity_map, marker_size_map : mapping, optional
    Category-to-style dictionaries. Extra keys are allowed for reuse on
    filtered data. Colours and Bokeh marker names must be valid; opacities
    must be finite in [0, 1], and sizes finite and positive.
colour_continuous : bool, optional (default=False)
    Explicitly map finite numeric ``colour_by`` values to a continuous palette
    with a colour bar. Conflicts with ``colour_map`` and ``colour_list``.
    Numeric columns otherwise use categorical mappings.
colour_palette : sequence of str, optional
    At least two valid colours for continuous colour only. Defaults to Viridis256.

Returns
-------
None
    Displays the plot in a Jupyter notebook and optionally saves it as HTML.

Raises
------
ValueError
    If columns, mappings, style values, conflicting arguments, or required
    legacy categorical metadata values are invalid.

Notes
-----
Colour and shape keys use dedicated invisible proxy renderers with unrelated
glyph properties held constant. Opacity and size keys are text-only Bokeh
legend items and include the mapped values. All labels identify the encoding
and source column. The legend is kept in one compact area; continuous colour
uses the colour bar without categorical colour swatches. Proxy appearance and
legend columns can be changed through ``update_bokeh_options()``. The figure
keeps its 866 by 650 pixel default.
Hover details and exported HTML use ``MOLFILE`` paths supplied by ``prepare()``.

Examples
--------
>>> chem_space.plot(
...     colour_list=['blue', 'green'],
...     shape_list=['circle', 'square'],
...     filename='my_plot',
...     title='Chemical Space',
...     opacity_list=[0.3, 0.9],
...     marker_size_list=[6, 12],
...     save_html=True
... )
"""


        from mlchem.helper import (prepare_datatable, compute_alpha,
                                   size_ratio, bokeh_plot, add_encoding_legend)
        import bokeh.plotting as bp
        from bokeh.io import output_notebook, reset_output
        from bokeh.models import HoverTool, ColorBar, LinearColorMapper
        from bokeh.palettes import Viridis256
        from bokeh.transform import transform

        if self.df_compressed.empty:
            raise ValueError("'df_compressed' must contain at least one datapoint.")
        if not self.df_compressed.columns.is_unique:
            raise ValueError("'df_compressed' must have unique column names.")
        if title_fontsize is not None and not (
            isinstance(title_fontsize, str) and title_fontsize.strip()
            or isinstance(title_fontsize, Integral)
            and not isinstance(title_fontsize, bool) and title_fontsize > 0
        ):
            raise ValueError("'title_fontsize' must be a positive integer or non-empty string.")
        if not isinstance(colour_continuous, bool):
            raise ValueError("'colour_continuous' must be a bool.")
        if colour_continuous and colour_by is None:
            raise ValueError("'colour_continuous' requires 'colour_by'.")
        if colour_palette is not None and not colour_continuous:
            raise ValueError("'colour_palette' requires 'colour_continuous=True'.")
        proxy_size = self.bokeh_dictionary['legend_proxy_size']
        proxy_alpha = self.bokeh_dictionary['legend_proxy_alpha']
        proxy_marker = self.bokeh_dictionary['legend_colour_proxy_marker']
        proxy_colour = self.bokeh_dictionary['legend_shape_proxy_colour']
        legend_ncols = self.bokeh_dictionary['legend_ncols']
        self._validate_encoding_values('marker_size', [proxy_size], 'legend_proxy_size')
        self._validate_encoding_values('opacity', [proxy_alpha], 'legend_proxy_alpha')
        if proxy_alpha == 0:
            raise ValueError("'legend_proxy_alpha' must be greater than zero.")
        self._validate_encoding_values('shape', [proxy_marker], 'legend_colour_proxy_marker')
        self._validate_encoding_values('colour', [proxy_colour], 'legend_shape_proxy_colour')
        if (
            not isinstance(legend_ncols, Integral)
            or isinstance(legend_ncols, bool)
            or legend_ncols < 1
        ):
            raise ValueError("'legend_ncols' must be a positive integer.")
        encoding_specs = [
            ('colour', colour_by, colour_map, colour_list),
            ('shape', shape_by, shape_map, shape_list),
            ('opacity', opacity_by, opacity_map, opacity_list),
            ('marker_size', marker_size_by, marker_size_map, marker_size_list),
        ]
        mappings = {
            encoding: self._encoding_map(
                encoding, column, mapping, legacy,
                continuous=encoding == 'colour' and colour_continuous,
            )
            for encoding, column, mapping, legacy in encoding_specs
        }
        if (
            shape_by is None and self.metadata_exists and self.metadata_categorical
            and self.df_compressed.METADATA.isna().any()
        ):
            raise ValueError(
                "'METADATA' contains missing values required by legacy categorical "
                "shapes; provide 'shape_by' or fill the missing values."
            )
        colour_mapper = None
        if colour_continuous:
            palette = list(Viridis256 if colour_palette is None else colour_palette)
            if len(palette) < 2:
                raise ValueError("'colour_palette' must contain at least two colours.")
            self._validate_encoding_values('colour', palette, 'colour_palette')
            colour_mapper = LinearColorMapper(
                palette=palette,
                low=float(self.df_compressed[colour_by].min()),
                high=float(self.df_compressed[colour_by].max()),
            )

        for name, values in (
                ('opacity_list', opacity_list),
                ('marker_size_list', marker_size_list)):
            if values is None:
                continue
            if len(values) != len(self.classnames):
                raise ValueError(
                    f"'{name}' must contain one value per class "
                    f"({len(self.classnames)} values in self.classnames order).")
            if not all(
                    isinstance(value, Real) and not isinstance(value, bool)
                    and np.isfinite(value)
                    and (0 <= value <= 1 if name == 'opacity_list' else value > 0)
                    for value in values):
                bounds = 'in [0, 1]' if name == 'opacity_list' else 'positive'
                raise ValueError(f"'{name}' values must be finite numbers {bounds}.")

        opacities = opacity_list
        if opacities is None:
            class_sizes = [
                len(self.df_compressed[self.df_compressed.CLASS == c])
                for c in self.classnames
            ]
            opacities = [
                compute_alpha(size) * size_ratio(size, len(self.df_compressed) - size)
                for size in class_sizes
            ]
        marker_sizes = (
            [marker_size] * len(self.classnames)
            if marker_size_list is None else marker_size_list
        )

        self.height = height
        self.width = width
        self.filename = filename
        self.marker_size = marker_size

        self.colour_list = colour_list
        if self.colour_list is None:
            self.colour_list = ['Blue', 'Orange', 'Red', 'Black',
                                'Green', 'Cyan', 'Magenta', 'Yellow']
        self.shape_list = shape_list
        if self.shape_list is None:
            self.shape_list = ['circle', 'triangle', 'square', 'star',
                               'diamond', 'cross', 'x', 'asterisk']
        if colour_by is None:
            if len(self.colour_list) < len(self.classnames):
                raise ValueError("'colour_list' must contain at least one colour per class.")
            self._validate_encoding_values('colour', self.colour_list, 'colour_list')
        if shape_by is None and self.metadata_exists and self.metadata_categorical:
            if len(self.shape_list) < len(self.metadata_names):
                raise ValueError("'shape_list' must contain at least one shape per metadata category.")
            self._validate_encoding_values('shape', self.shape_list, 'shape_list')
        if marker_size_by is None and marker_size_list is None:
            self._validate_encoding_values('marker_size', [marker_size], 'marker_size')

        reset_output()
        output_notebook()

        plot = bp.figure(title='%s' % (filename),
                         width=self.width,
                         height=self.height,
                         )

        self.data_table_combined = prepare_datatable(
            self.df_compressed, height=500, width=self.width)

        self.dict_datatables = {}
        plotted = []

        if any(column is not None for _, column, _, _ in encoding_specs):
            group_columns = ['CLASS']
            for encoding, column, _, _ in encoding_specs:
                if column is not None and not (encoding == 'colour' and colour_continuous):
                    if column not in group_columns:
                        group_columns.append(column)
            if shape_by is None and self.metadata_exists and self.metadata_categorical:
                if 'METADATA' not in group_columns:
                    group_columns.append('METADATA')
            for _, group in self.df_compressed.groupby(
                group_columns, sort=False, observed=True, dropna=False,
            ):
                row = group.iloc[0]
                class_index = list(self.classnames).index(row.CLASS)
                colour = (
                    transform(colour_by, colour_mapper) if colour_continuous
                    else mappings['colour'][row[colour_by]] if colour_by is not None
                    else self.colour_list[class_index]
                )
                shape = (
                    mappings['shape'][row[shape_by]] if shape_by is not None
                    else self.shape_list[list(self.metadata_names).index(row.METADATA)]
                    if self.metadata_exists and self.metadata_categorical else 'circle'
                )
                alpha = (
                    mappings['opacity'][row[opacity_by]] if opacity_by is not None
                    else opacities[class_index]
                )
                size = (
                    mappings['marker_size'][row[marker_size_by]] if marker_size_by is not None
                    else marker_sizes[class_index]
                )
                data_table = prepare_datatable(group)
                renderer = plot.scatter(
                    x='DIM_1', y='DIM_2', source=data_table.source,
                    color=colour, marker=shape, alpha=alpha, size=size, line_color='grey',
                )
                plotted.append(renderer)
            for c in self.classnames:
                self.dict_datatables[c] = prepare_datatable(
                    self.df_compressed[self.df_compressed.CLASS == c])
        elif self.metadata_exists and self.metadata_categorical:

            for i, c in enumerate(self.classnames):
                for i2, c2 in enumerate(self.metadata_names):
                    data_table = prepare_datatable(
                        self.df_compressed[
                            (self.df_compressed.METADATA == c2) &
                            (self.df_compressed.CLASS == c)])
                    renderer = plot.scatter(x="DIM_1", y='DIM_2',
                                 source=data_table.source,
                                 color=self.colour_list[i],
                                 size=marker_sizes[i],
                                 marker=self.shape_list[i2],
                                 line_color='grey',
                                 alpha=opacities[i]
                                 )
                    plotted.append(renderer)
                self.dict_datatables[c] = prepare_datatable(
                    self.df_compressed[self.df_compressed.CLASS == c])
        else:
            for i, c in enumerate(self.classnames):
                data_table = prepare_datatable(
                    self.df_compressed[self.df_compressed.CLASS == c])
                self.dict_datatables[c] = data_table
                renderer = plot.scatter(x="DIM_1",
                            y='DIM_2',
                            source=data_table.source,
                            color=self.colour_list[i],
                            size=marker_sizes[i],
                            line_color='grey',
                            alpha=opacities[i])
                plotted.append(renderer)

        colour_legend_entries = []
        if not colour_continuous:
            colour_column = colour_by or 'CLASS'
            for category in pd.unique(self.df_compressed[colour_column]):
                colour = (
                    mappings['colour'][category] if colour_by is not None
                    else self.colour_list[list(self.classnames).index(category)]
                )
                colour_legend_entries.append(
                    (f'Colour ({colour_column}): {category}', colour)
                )

        shape_legend_entries = []
        if shape_by is not None or (self.metadata_exists and self.metadata_categorical):
            shape_column = shape_by or 'METADATA'
            for category in pd.unique(self.df_compressed[shape_column]):
                shape = (
                    mappings['shape'][category] if shape_by is not None
                    else self.shape_list[list(self.metadata_names).index(category)]
                )
                shape_legend_entries.append(
                    (f'Shape ({shape_column}): {category}', shape)
                )

        text_legend_entries = []
        if opacity_by is not None or opacity_list is not None:
            opacity_column = opacity_by or 'CLASS'
            for category in pd.unique(self.df_compressed[opacity_column]):
                opacity = (
                    mappings['opacity'][category] if opacity_by is not None
                    else opacity_list[list(self.classnames).index(category)]
                )
                text_legend_entries.append(
                    f'Opacity ({opacity_column}): {category} = {opacity}'
                )
        if marker_size_by is not None or marker_size_list is not None:
            size_column = marker_size_by or 'CLASS'
            for category in pd.unique(self.df_compressed[size_column]):
                size = (
                    mappings['marker_size'][category] if marker_size_by is not None
                    else marker_size_list[list(self.classnames).index(category)]
                )
                text_legend_entries.append(
                    f'Size ({size_column}): {category} = {size} px'
                )

        if colour_legend_entries or shape_legend_entries or text_legend_entries:
            add_encoding_legend(
                plot,
                colour_entries=colour_legend_entries,
                shape_entries=shape_legend_entries,
                text_entries=text_legend_entries,
                proxy_x=self.df_compressed.DIM_1.iloc[0],
                proxy_y=self.df_compressed.DIM_2.iloc[0],
                proxy_size=proxy_size,
                proxy_alpha=proxy_alpha,
                colour_proxy_marker=proxy_marker,
                shape_proxy_colour=proxy_colour,
                ncols=legend_ncols,
            )
        if colour_mapper is not None:
            plot.add_layout(ColorBar(color_mapper=colour_mapper, title=colour_by), 'right')

        # Title
        plot.title_location = self.bokeh_dictionary['title_location']
        plot.title.text = title
        if title_fontsize is None:
            plot.title.text_font_size = self.bokeh_dictionary['title_fontsize']
        elif isinstance(title_fontsize, Integral) and not isinstance(title_fontsize, bool):
            plot.title.text_font_size = f'{int(title_fontsize)}px'
        else:
            plot.title.text_font_size = title_fontsize
        plot.title.align = self.bokeh_dictionary['title_align']
        plot.title.background_fill_color = self.bokeh_dictionary[
            'title_background_fill_colour']
        plot.title.text_color = self.bokeh_dictionary['title_text_colour']
        # Legend
        if plot.legend:
            plot.legend.location = self.bokeh_dictionary['legend_location']
            plot.legend.title = self.bokeh_dictionary['legend_title']
            plot.legend.label_text_font = self.bokeh_dictionary[
                'legend_label_text_font']
            plot.legend.label_text_font_style = self.bokeh_dictionary[
                'legend_label_text_font_style']
            plot.legend.label_text_color = self.bokeh_dictionary[
                'legend_label_text_colour']
            plot.legend.border_line_width = self.bokeh_dictionary[
                'legend_border_line_width']
            plot.legend.border_line_color = self.bokeh_dictionary[
                'legend_border_line_colour']
            plot.legend.border_line_alpha = self.bokeh_dictionary[
                'legend_border_line_alpha']
            plot.legend.background_fill_color = self.bokeh_dictionary[
                'legend_background_fill_colour']
            plot.legend.background_fill_alpha = self.bokeh_dictionary[
                'legend_background_fill_alpha']
        # Axes
        plot.xaxis.axis_label = self.bokeh_dictionary['xaxis_label']
        plot.xaxis.axis_line_width = self.bokeh_dictionary['xaxis_line_width']
        plot.xaxis.axis_line_color = self.bokeh_dictionary['xaxis_line_colour']
        plot.xaxis.major_label_orientation = self.bokeh_dictionary[
            'xaxis_major_label_orientation']
        plot.xaxis.major_label_text_color = self.bokeh_dictionary[
            'xaxis_major_label_text_colour']
        plot.yaxis.axis_label = self.bokeh_dictionary['yaxis_label']
        plot.yaxis.axis_line_width = self.bokeh_dictionary['yaxis_line_width']
        plot.yaxis.axis_line_color = self.bokeh_dictionary['yaxis_line_colour']
        plot.yaxis.major_label_orientation = self.bokeh_dictionary[
            'yaxis_major_label_orientation']
        plot.yaxis.major_label_text_color = self.bokeh_dictionary[
            'yaxis_major_label_text_colour']
        plot.axis.minor_tick_in = self.bokeh_dictionary['axis_minor_tick_in']
        plot.axis.minor_tick_out = self.bokeh_dictionary['axis_minor_tick_out']

        hover = HoverTool(tooltips=self.bokeh_tooltips, renderers=plotted)
        plot.add_tools(hover)

        if save_html is True:     # save html
            bp.output_file(
                filename='%s.html' % (self.filename), title="Static HTML file"
                )
        bokeh_plot(plot, self.classnames, self.dict_datatables)
