import pandas as pd
import numpy as np


class ChemicalSpace:
    """
A module to compute and visualise datasets in a compressed embedded space.

This class handles descriptor processing, dimensionality reduction,
and preparation of molecular data for interactive visualisation.

Parameters
----------
data : pd.DataFrame
    A DataFrame with 3 or 4 columns: 'SMILES', 'NAME', 'CLASS',
    and optionally 'METADATA'.
needs_cleaning : bool, optional
    Whether the data requires cleaning. Default is False.
df_descriptors : pd.DataFrame, optional
    A DataFrame with SMILES as index and descriptor columns.
metadata_exists : bool, optional
    True if a 4th column is present in `data`. Default is False.
metadata_categorical : bool, optional
    True if the 4th column is categorical. Default is False.

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
and metadata flags. It also validates the structure of the input
data and loads default Bokeh visualisation settings.

Parameters
----------
data : pd.DataFrame
    A DataFrame with either 3 or 4 columns:
    - 'SMILES': SMILES strings of the molecules.
    - 'NAME': Names or identifiers of the molecules.
    - 'CLASS': Class labels for grouping or colouring.
    - 'METADATA' (optional): Additional metadata for further grouping.
needs_cleaning : bool, optional
    Whether the data requires cleaning before processing. Default is False.
df_descriptors : pd.DataFrame, optional
    A DataFrame with SMILES as index and molecular descriptors as columns.
metadata_exists : bool, optional
    True if a fourth column ('METADATA') is present in `data`. Default is False.
metadata_categorical : bool, optional
    True if the metadata column is categorical. Default is False.

Raises
------
ValueError
    If the `data` DataFrame does not have the expected column structure.

Notes
-----
If `df_descriptors` is not provided, a message will be printed to
remind the user to supply descriptors before visualisation.

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

        if self.metadata_exists is False:
            if list(self.data.columns) != ['SMILES', 'NAME', 'CLASS']:
                raise ValueError("Please make sure data has only 3 "
                                 "columns called 'SMILES','NAME','CLASS'.")
        else:
            if list(self.data.columns) != [
                'SMILES', 'NAME', 'CLASS', 'METADATA'
                    ]:
                raise ValueError(
                    "Please make sure data has only 4 "
                    "columns called 'SMILES','NAME','CLASS','METADATA.")

        if self.df_descriptors is None:
            print('The given data have no descriptors available. '
                  'To visualise the chemical space, please provide '
                  'a descriptor dataframe or use the '
                  'mlchem.chem.Calculator.descriptors.py module.\n')

    def process(self,
                diversity_filter: float | None,
                collinearity_filter: float | None,
                standardise: bool = True):
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

Returns
-------
None
    Updates the instance with processed descriptor data.

Raises
------
AssertionError
    If `diversity_filter` or `collinearity_filter` are outside [0, 1).

Examples
--------
>>> chem_space = ChemicalSpace(data, df_descriptors)
>>> chem_space.process(diversity_filter=0.5, collinearity_filter=0.3)
"""


        from mlchem.ml.feature_selection import filters
        from mlchem.ml.preprocessing.scaling import scale_df_standard

        if diversity_filter:
            
          assert (0. <= diversity_filter < 1.,
                  "'diversity_filter' argument must be a float 0 <= x < 1.")
        if collinearity_filter:
          assert( 0. < collinearity_filter <= 1.,
                "'collinearity_filter' argument must be a float 0 < x <= 1.")
          
        if diversity_filter:
          print("Before filtering:", self.df_descriptors.shape)
          df_filtered_1 = filters.diversity_filter(self.df_descriptors,
                                                   diversity_filter)
        else:
            df_filtered_1 = self.df_descriptors     # do nothing

        print("After diversity filter:", df_filtered_1.shape)

        if collinearity_filter:
            df_filtered_2 = filters.collinearity_filter(df_filtered_1,
                                                        collinearity_filter)
        else:
            df_filtered_2 = df_filtered_1
        print("After collinearity filter:", df_filtered_2.shape)

        if standardise is True:
            self.df_processed, self.scaler = scale_df_standard(
                df_filtered_2)
        else:
            self.df_processed = df_filtered_2

    def prepare(self, df_compressed: pd.core.frame.DataFrame):
      """
Prepare the compressed dataframe for visualisation.

This method adds molecular structure files, names, and metadata
to the compressed coordinates.

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
AssertionError
    If `df_compressed` does not have exactly two columns named
    ['DIM_1', 'DIM_2'].
ValueError
    If the index of `df_compressed` or `df_processed` is not a
    series of SMILES strings.

Examples
--------
>>> chem_space.prepare(df_compressed)
"""


      import os
      from mlchem.helper import (create_structure_files,
                                prepare_dataframe)

      self.df_compressed = df_compressed

      assert list(self.df_compressed.columns) == [
          'DIM_1',
          'DIM_2'
          ]
      "'df_compressed' must have only 2 columns: ['DIM_1','DIM_2']."

      try:
          self.df_compressed.reset_index(inplace=True)
          self.df_compressed.columns = ['SMILES'] + list(self.df_compressed.columns)[1:]
      except Exception:
          raise ValueError(
              "Index of df_compressed must be a series of SMILES strings."
              )
      try:
          self.df_processed.reset_index(inplace=True)
          self.df_processed.columns = ['SMILES'] + list(self.df_processed.columns)[1:]
      except Exception:
          raise ValueError("Index of df_processed must be a series of SMILES strings.")


      # Classes taken from the list of molecules
      # retained after descriptor calculation.
      self.classes = self.data[
          self.data.SMILES.isin(
              self.df_processed.SMILES
              )
              ].CLASS.values

      if len(self.classes) != len(self.df_compressed):
          self.classes = self.data.CLASS.values
      self.classnames = np.unique(self.classes)
      if self.metadata_categorical+self.metadata_exists == 2:
          self.metadata_names = np.unique(self.data[self.data.columns[-1]])

      self.df_processed.insert(0, 'CLASS', self.classes)

      self.molfiles = []
      self.names = []
      self.metadata = []

      for _, c in enumerate(self.classnames):
          df = self.df_processed[self.df_processed.CLASS == c]

          # Check whether structural files for that dataframe already exist,
          # otherwise create them
          try:
              os.mkdir(c)
          except Exception:
              pass

          # If there are as many structural files as the number of molecules
          # for that specific class:
          if len(os.listdir(c)) == len(df):
              pass
          else:
              create_structure_files(df=df,
                                    structure_column_name='SMILES',
                                    folder_name=c)

          names = []
          metadata = []
          for s in df.SMILES:
              temp_df = self.data[self.data.SMILES == s]
              name = temp_df.NAME.values[0]
              names.append(name)
              try:
                  metainfo = temp_df.METADATA.values[0]
                  metadata.append(metainfo)
              except Exception:
                  pass
          df.insert(1, 'NAME', names)
          df_prepared = prepare_dataframe(df, c)
          df_prepared.index = np.arange(0, len(df_prepared))
          self.molfiles.append(df_prepared.MOLFILE.values)
          self.names.append(names)
          try:
              self.metadata.append(metadata)
              df_prepared['METADATA'] = metadata
          except Exception:
              pass

      self.df_compressed['MOLFILE'] = np.hstack(self.molfiles)
      self.df_compressed['NAME'] = np.hstack(self.names)
      self.df_compressed['NAME_SHORT'] = [a[:20] for
                                    a in self.df_compressed.NAME.values]
      self.df_compressed['CLASS'] = np.hstack(self.df_processed.CLASS.values)

      # Update the index of df_compressed
      new_index = []
      for a in self.df_compressed.MOLFILE.values:
          start = a.find('/') + 1
          end = a.find('.')
          substring = a[start:end]
          try:
              new_index.append(int(substring))
          except ValueError:
              # Handle non-numeric substrings
              new_index.append(substring)

      self.df_compressed.index = new_index

      try:
          self.df_compressed['METADATA'] = np.hstack(self.metadata)
      except Exception:
          pass

    def update_bokeh_options(self, **args) -> None:
        """
Update the Bokeh plot options.

This method allows customisation of various Bokeh plot parameters
for visualisations. Users can pass keyword arguments to set options
such as title properties, legend properties, and axis properties.

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

    def plot(self,
             colour_list: list = None,
             shape_list: list = None,
             filename: str = '',
             title: str = '',
             title_fontsize: int = 25,
             height: int = 650,
             width: int = int(650/0.75),
             marker_size: int = 10,
             save_html: bool = False) -> None:
        """
Generate a 2D scatter plot of the chemical space using Bokeh.

This method visualises the compressed chemical space using a scatter plot,
with options to customise colours, shapes, size, and layout. It supports
categorical metadata and can optionally save the plot as an HTML file.

Parameters
----------
colour_list : list, optional
    List of colours for the plot markers. Each class will be assigned a
    colour. If not provided, defaults to:
    ['Blue', 'Orange', 'Red', 'Black', 'Green', 'Cyan', 'Magenta', 'Yellow'].
shape_list : list, optional
    List of marker shapes. If not provided, defaults to:
    ['circle', 'triangle', 'square', 'star', 'diamond', 'cross', 'x', 'asterisk'].
filename : str, optional
    Name of the file to save the plot as (without extension).
title : str, optional
    Title of the plot.
title_fontsize : int, optional
    Font size of the plot title. Default is 25.
height : int, optional
    Height of the plot in pixels. Default is 650.
width : int, optional
    Width of the plot in pixels. Default is 866 (650 / 0.75).
marker_size : int, optional
    Size of the plot markers. Default is 10.
save_html : bool, optional
    Whether to save the plot as an HTML file. Default is False.

Returns
-------
None
    Displays the plot in a Jupyter notebook and optionally saves it as HTML.

Examples
--------
>>> chem_space.plot(
...     colour_list=['blue', 'green'],
...     shape_list=['circle', 'square'],
...     filename='my_plot',
...     title='Chemical Space',
...     save_html=True
... )
"""


        import os
        from mlchem.helper import (prepare_datatable, compute_alpha,
                                   size_ratio, bokeh_plot)
        import bokeh.plotting as bp
        from bokeh.io import output_notebook, reset_output
        output_notebook()
        from bokeh.models import HoverTool

        # TODO: implement other functions using  other modules:
        # from bokeh.models import (ColorBar, LinearColorMapper, Div,
        # RangeSlider, Spinner, TabPanel, Tabs, NumberEditor,
        # NumberFormatter, SelectEditor, StringEditor, StringFormatter)
        # from bokeh.palettes import Turbo256, Spectral6, RedBu
        # from bokeh.transform import linear_cmap, transform

        reset_output()
        
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

        plot = bp.figure(title='%s' % (filename),
                         width=self.width,
                         height=self.height,
                         )

        self.data_table_combined = prepare_datatable(
            self.df_compressed, height=500, width=self.width)

        self.dict_datatables = {}

        if self.metadata_exists and self.metadata_categorical:

            for i, c in enumerate(self.classnames):
                size = len(os.listdir(c))
                for i2, c2 in enumerate(self.metadata_names):
                    data_table = prepare_datatable(
                        self.df_compressed[
                            (self.df_compressed.METADATA == c2) &
                            (self.df_compressed.CLASS == c)])
                    plot.scatter(x="DIM_1", y='DIM_2',
                                 source=data_table.source,
                                 color=self.colour_list[i],
                                 size=self.marker_size,
                                 marker=self.shape_list[i2],
                                 line_color='grey',
                                 legend_label=c,
                                 alpha=compute_alpha(size) *
                                 size_ratio(size, len(self.df_compressed)-size)
                                 )
                self.dict_datatables[c] = prepare_datatable(
                    self.df_compressed[self.df_compressed.CLASS == c])
        else:
            for i, c in enumerate(self.classnames):
                size = len(os.listdir(c))
                data_table = prepare_datatable(
                    self.df_compressed[self.df_compressed.CLASS == c])
                self.dict_datatables[c] = data_table
                plot.scatter(x="DIM_1",
                            y='DIM_2',
                            source=data_table.source,
                            color=self.colour_list[i],
                            size=self.marker_size,
                            line_color='grey',
                            legend_label=c,
                            alpha=compute_alpha(size) *
                            size_ratio(size, len(self.df_compressed)-size))

        # Title
        plot.title_location = self.bokeh_dictionary['title_location']
        plot.title.text = title
        plot.title.text_font_size = "%dpx" % (title_fontsize)
        plot.title.align = self.bokeh_dictionary['title_align']
        plot.title.background_fill_color = self.bokeh_dictionary[
            'title_background_fill_colour']
        plot.title.text_color = self.bokeh_dictionary['title_text_colour']
        # Legend
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

        hover = HoverTool(tooltips=self.bokeh_tooltips)
        plot.add_tools(hover)

        if save_html is True:     # save html
            bp.output_file(
                filename='%s.html' % (self.filename), title="Static HTML file"
                )
        bokeh_plot(plot, self.classnames, self.dict_datatables)
