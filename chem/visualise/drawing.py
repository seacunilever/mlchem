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

from typing import Iterable, Literal
from rdkit import Chem
from PIL import Image

class MolDrawer:
    """
Render molecules and image grids with configurable RDKit drawing options.

`MolDrawer` centralises drawing configuration (`drawing_options`) and provides
helpers for:

- single-molecule rendering (`draw_mol`),
- palette preview (`show_palette`),
- loading molecules/images (`load_mols`, `load_images`),
- and gallery output (`show_images_grid`).

The class can operate with:

- mlchem defaults (`MLCHEM_DEFAULTS`),
- RDKit defaults (`get_rdkit_defaults`),
- and per-instance option overrides (`update_drawing_options`).

Typical usage:

>>> from rdkit import Chem
>>> drawer = MolDrawer(size=[400, 300], legend="Example")
>>> drawer.update_drawing_options(atomPalette='cdk', highlightColour='tomato')
>>> mol = Chem.MolFromSmiles("CCO")
>>> image = drawer.draw_mol(mol, highlightAtoms=[1])

Batch usage:

>>> drawer = MolDrawer(size=[300, 300])
>>> drawer.load_mols([Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")])
>>> drawer.show_images_grid(n_columns=2)

Attributes
----------
mol : rdkit.Chem.rdchem.Mol or None
    Default molecule used by `draw_mol` when no molecule is passed.
highlightAtoms : Iterable
    Default atom indices to highlight when `draw_mol` receives none.
size : Iterable
    Default canvas size in pixels as `(width, height)`.
legend : str
    Default legend used by `draw_mol` when none is passed.
mol_list : list
    Internal list of molecules accumulated via `load_mols`.
highlightAtoms_list : list
    History of highlight index sets used in drawing calls.
size_list : list
    History of canvas sizes used in drawing calls.
legend_list : list
    History of legends used in drawing calls.
img_list : list[PIL.Image.Image]
    Internal image collection used by `load_images`, `load_mols`, and
    `show_images_grid`.
colour_dictionary : dict
    Named RGB palette imported from `mlchem.importables`.
drawing_options : dict
    Active drawing configuration (starts as a copy of `MLCHEM_DEFAULTS`).

Methods
-------
__init__(...)
    Initialise the drawer and instance-level defaults.
get_rdkit_defaults(...)
    Return RDKit native drawing defaults as a dictionary.
show_palette(...)
    Visualise or save a colour dictionary.
update_drawing_options(...)
    Update instance drawing options via keyword arguments.
reset_drawing_options(...)
    Reset options to mlchem or RDKit defaults.
load_images(...)
    Add one or more pre-rendered PIL images to the internal image list.
load_mols(...)
    Add molecules and immediately render/store their images.
show_images_grid(...)
    Display and optionally save images as a tiled grid.
draw_mol(...)
    Render a single molecule with highlights, maps, and optional ACS style.
"""
    MLCHEM_DEFAULTS = {


    # colours #


    # Possible values: 'avalon', 'cdk', 'bw',
    # or a dictionary like this: {atomic_number:(R,G,B)),}
    'atomPalette': 'cdk',

    # Possible values: any present in 'colour_dictionary'
    # attribute of the MolDrawer class or any RGB tuple
    'backgroundColour': 'white',

    # Possible values: any present in 'colour_dictionary'
    # attribute of the MolDrawer class or any RGB tuple
    'highlightColour': 'tomato',

    # The transparency of the highlighting colour
    'highlightAlpha': 1,

    # Colour of the SMARTS query
    'queryColour': 'red',

    # The color used for RDKit query annotations
    'annotationColour': 'black',

    # The color used for custom atom notes (from assign_atom_notes)
    'atomNoteColour': 'black',

    # The color used for custom bond notes (from assign_bond_notes)
    'bondNoteColour': 'black',

    # The color used for legend text
    'legendColour': 'black',

    # The color used for symbol text
    'symbolColour': 'black',

    # The color used for variable attachment points
    'variableAttachmentColour': 'black',

    # Drawing style controls #

    # Display dummy atoms as dummy attachment points
    'dummiesAreAttachments': False,

    # Shortcut for set_property(mol, property_type='atomnote',
    #                           atoms=[],custom_string='')
    'addAtomIndices': False,

    # Display bond indices
    'addBondIndices': False,

    # Hide all atom labels
    'noAtomLabels': False,

    # Show explicit methyl
    'explicitMethyl': False,

    # Include radicals
    'includeRadicals': True,

    # If False, simplify drawing of standard query atoms
    # (Q, QH, X, XH, A, AH, M, MH) from mol files or CXSMILES
    'useComplexQueryAtomSymbols': True,

    # Single or double coloured chiral bonds
    'singleColourWedgeBonds': False,

    # If True, draw molecules having same scale
    'drawMolsSameScale': False,

    # Highlighting #


    # Include bonds if highlighted atoms are adjacent
    'continuousHighlight': True,

    # Atoms are highlighted with small circles
    'circleAtoms': True,

    # Exclude H from atom circles
    'atomHighlightsAreCircles': True,

    # Fill bonds
    'fillHighlights': True,

    # Highlighting width of atoms
    'highlightRadius': 0.3,

    # Highlighting width of bonds scaling factor
    'highlightBondWidthMultiplier': 10,



    # Stereochemistry ##


    # Display R,S notations; display, if specified, abs,and,or.
    # For abs: (a:atom_number+1), for and, or: &, o
    'addStereoAnnotation': False,

    # Draw unspecified stereo atoms/bonds as unknown
    'unspecifiedStereoIsUnknown': False,


    # Fonts and text #


    # Sets the initial font size, which can be scaled based on
    # the molecule’s size. All elements are involved
    'baseFontSize': 0.6,

    # Increase or decrease font size of annotations
    'annotationFontScale': 0.5,

    # Increase or decrease font size of legends
    'legendFontSize': 25,

    # Set to any positive number to force the base fontsize
    # to remain unchanged even if canvas size varies.
    'fixedFontSize': -1,

    # Ensures a minimum font size, preventing labels and notes
    # from becoming too small.
    'minFontSize': 6,

    # Set to any number to set a ceiling to the base fontsize.
    'maxFontSize': 40,

    # Specify the path where the font file is stored
    'fontFile': '',


    # Bond drawing parameters #


    # How distant the additional lines
    # of a double/triple bond have to be from the single bond line
    'multipleBondOffset': 0.15,

    # Fraction of fontsize. How much buffer space around atoms
    'additionalAtomLabelPadding': 0,

    # How wide bonds are
    'bondLineWidth': 2,

    # Adapt bond witdth to highlight width
    'scaleBondWidth': False,

    # Adapt hilight width to bond width
    'scaleHighlightBondWidth': True,

    # If different from -1, forces molecule to
    # have the same scale. The higher the value
    # the larger the scale.
    'fixedBondLength': -1,


    # Weight and similarity maps parameters #


    # Similarity maps drawing style
    # 'GC': Gaussian Contours - smooth gaussian overlay (RDKit-based)
    #       Best for continuous properties (electrostatic maps, similarity)
    # 'C':  Circle Contours - concentric gradient circles per atom (mlchem-custom)
    #       Best for discrete atom properties (SHAP values, per-atom charges)
    # Note: mlchem's circle contours are a custom implementation providing
    #       an alternative to RDKit's gaussian contours, with better atom-level clarity.
    'mapStyle': 'GC',

    # Colour map for similarity maps; default is None.
    # A list of 3 tuples/colour names is accepted too.
    'colourMap': None,

    # Colour of atoms having positive weights (applies to both GC and C styles)
    'positiveColour': 'green',

    # Colour of atoms having negative weights (applies to both GC and C styles)
    'negativeColour': 'mediumvioletred',

    # List of atomic weights to visualize via contours.
    # Should be a list/array with one value per atom in the molecule.
    # If provided with mapStyle='GC' or mapStyle='C', contours will be drawn.
    # Example: atomWeights = [0.1, -0.2, 0.3, 0.05, ...]
    'atomWeights': [],

    # Baseline alpha (transparency) of weight color overlays (style: both GC and C)
    'weightAlpha': 0.2,

    # Circle radius scale factor (style: circles only)
    # Controls how large circles scale based on atomic weight magnitude
    'scalingFactor': 2,

    # Minimum circle radius in pixels (style: circles only)
    'minRadius': 2,

    # Maximum circle radius in pixels (style: circles only)
    'maxRadius': 30,

    # Number of concentrical circles/contour lines per visualization (style: both)
    # For circles: number of concentric rings per atom
    # For gaussian: number of contour levels
    'numContours': 10,

    # Line width of the gaussian contours (style: gaussian contours only)
    'contourWidth': 1,

    # Resolution (grid step) of gaussian contour calculation (style: gaussian contours only)
    # Lower values = higher resolution but slower computation
    'mapRes': 0.05,

    # Contour line colour (style: gaussian contours only)
    'contourColour': 'black',

    # Whether to display negative weights as dashed lines (style: both)
    # True: negative weights shown with dashed contours for clear distinction
    'dashNegative' : True,



    # Miscellaneous #

    # Set to False to have transparent background
    'clearBackground': True,

    # Set to False to disable kekulisation prior to rendering
    'prepareMolsBeforeDrawing': True,

    # Rotation angle in degrees
    'rotate': 0,

    # Add or remove extra buffer zone. If value > 0.5,
    # molecule flips (unwanted behaviour).
    # At the moment, legend does not show when padding > 0.05.
    'padding': 0.05,

    # Set to True to show H isotopes as D and T rather
    # than as 2H and 3H
    'atomLabelDeuteriumTritium': False
    }

    @staticmethod
    def get_rdkit_defaults() -> dict[str, object]:
        """Retrieve the default RDKit drawing options.

        This method creates a temporary RDKit drawing context and reads the
        values from its drawOptions object. The returned dictionary can be
        used to compare or restore RDKit default drawing settings.

        Returns
        -------
        dict[str, object]
            Dictionary of RDKit drawing option names and default values.
        """
        from rdkit.Chem.Draw import rdMolDraw2D

        opts = rdMolDraw2D.MolDraw2DCairo(300, 300).drawOptions()
        defaults = {}
        for name in dir(opts):
            if name.startswith("_"):
                continue
            try:
                value = getattr(opts, name)
                if callable(value):
                    continue
                defaults[name] = value
            except Exception:
                pass
        return defaults
    
    def __init__(self,
                 mol: Chem.rdchem.Mol | None = None,
                 highlightAtoms: Iterable = [],
                 size: Iterable = [300, 300],
                 legend: str = '') -> None:
        """
Initialise a `MolDrawer` instance and its default drawing state.

The constructor stores defaults that are reused by `draw_mol` unless per-call
arguments are provided.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol or None, optional
    Molecule to keep as instance default. If provided, `draw_mol()` can be
    called without passing `mol`.
highlightAtoms : Iterable, optional
    Default atom indices for highlighting in `draw_mol`.
size : Iterable, optional
    Default canvas size as `(width, height)` in pixels.
legend : str, optional
    Default legend text shown under molecules.

Returns
-------
None

Examples
--------
>>> from rdkit import Chem
>>> drawer = MolDrawer(
...     mol=Chem.MolFromSmiles("CCO"),
...     highlightAtoms=[1],
...     size=[400, 250],
...     legend="Ethanol",
... )
"""


        self.mol = mol
        self.mol_list = []
        self.highlightAtoms = highlightAtoms
        self.highlightAtoms_list = []
        self.size = size
        self.size_list = []
        self.legend = legend
        self.legend_list = []
        self.img_list = []

        from mlchem.importables import colour_dictionary

        self.colour_dictionary = colour_dictionary

        self.drawing_options = MolDrawer.MLCHEM_DEFAULTS.copy()



    def show_palette(
        self, palette: dict | None = None, save: bool = False,
        filename: str = '', size: Iterable = [1000, 300]
    ) -> Image.Image:
        """
Display a colour palette as an image.

If `palette` is not provided, `self.colour_dictionary` is used.

Parameters
----------
palette : dict or None, optional
    Colour mapping `{name: (r, g, b)}`. When `None`, uses
    `self.colour_dictionary`.
save : bool, optional
    If `True`, write the generated image to `filename`.
filename : str, optional
    Output path used when `save=True`.
size : Iterable, optional
    Output size in pixels as `(width, height)`.

Returns
-------
PIL.Image.Image
    Palette image.

Examples
--------
>>> drawer = MolDrawer()
>>> drawer.show_palette()
>>> drawer.show_palette(save=True, filename="palette.png")
"""

        from mlchem.helper import visualise_colour_grid, convert_size

        if palette is None:
            palette = self.colour_dictionary
        converted_figsize = convert_size(pixel_size=size)
        return visualise_colour_grid(palette, save,
                                     filename, converted_figsize)

    def update_drawing_options(self, **args) -> None:
        """
Update instance drawing options used by `draw_mol`.

The provided keyword arguments are merged into `self.drawing_options`.
Unknown keys are kept in the dictionary, but they only affect rendering if
they are consumed in `draw_mol` or recognised by RDKit draw options.

Parameters
----------
**args : dict
    Keyword arguments overriding one or more entries from
    `MolDrawer.MLCHEM_DEFAULTS`. Common groups include:

    - Colours: `atomPalette`, `backgroundColour`, `highlightColour`,
      `highlightAlpha`, `queryColour`, `annotationColour`.
    - Style/layout: `bondLineWidth`, `baseFontSize`, `padding`, `rotate`.
    - Similarity maps: `atomWeights`, `mapStyle`, `colourMap`, `mapRes`,
      `numContours`.
    - Optional shapes: `shapeTypes`, `shapeSizes`, `shapeColours`,
      `shapeCoords`.

Notes
-----
- For a complete list of default option names and values, inspect
  `MolDrawer.MLCHEM_DEFAULTS`.
- The default `mapRes` in mlchem is `0.05`.

Returns
-------
None

Examples
--------
>>> drawer = MolDrawer()
>>> drawer.update_drawing_options(atomPalette='avalon', backgroundColour='white')

>>> options = {'highlightColour': 'orange', 'rotate': 90}
>>> drawer.update_drawing_options(**options)
"""

        self.drawing_options.update(args)

    def reset_drawing_options(self, source: Literal['mlchem', 'rdkit'] = 'mlchem') -> None:
        """
Reset drawing options from a predefined source.

Parameters
----------
source : {'mlchem', 'rdkit'}, default='mlchem'
    Reset source:

    - `'mlchem'`: copy `MolDrawer.MLCHEM_DEFAULTS`.
    - `'rdkit'`: use values returned by `get_rdkit_defaults()`.

Returns
-------
None

Examples
--------
>>> drawer = MolDrawer()
>>> drawer.update_drawing_options(atomPalette='avalon', rotate=90)
>>> drawer.reset_drawing_options(source='mlchem')  # Reverts to default mlchem settings
>>> drawer.reset_drawing_options(source='rdkit')  # Reverts to native RDKit defaults
"""

        if source=='mlchem':
            self.drawing_options = MolDrawer.MLCHEM_DEFAULTS.copy()
        elif source=='rdkit':
            self.drawing_options = MolDrawer.get_rdkit_defaults().copy()

    
    def load_images(self, img_list:
                    Iterable[Image.Image]
                    | Image.Image) -> None:
        """
Load one or more images into `self.img_list`.

The method appends to existing images and flattens nested iterables.

Parameters
----------
img_list : PIL.Image.Image or Iterable[PIL.Image.Image]
    Single image or iterable of images to append.

Returns
-------
None

Examples
--------
>>> from PIL import Image
>>> img = Image.open("example.png")
>>> drawer = MolDrawer()
>>> drawer.load_images(img)

>>> drawer.load_images([img1, img2, img3])
"""

        from mlchem.helper import flatten

        self.img_list.append(img_list)
        self.img_list = list(flatten(self.img_list))

    def load_mols(self, mols:
                  Chem.rdchem.Mol | Iterable[Chem.rdchem.Mol]) -> None:
        """
Load molecules and render/store their images.

Molecules are appended to `self.mol_list`, then each molecule in the full
stored list is rendered via a temporary `MolDrawer` and collected in
`self.img_list`.

Parameters
----------
mols : rdkit.Chem.rdchem.Mol or Iterable[rdkit.Chem.rdchem.Mol]
    Single molecule or iterable of molecules.

Returns
-------
None

Examples
--------
>>> from rdkit import Chem
>>> mol1 = Chem.MolFromSmiles("CCO")
>>> mol2 = Chem.MolFromSmiles("c1ccccc1")
>>> drawer = MolDrawer()
>>> drawer.load_mols([mol1, mol2])
"""

        from mlchem.helper import flatten

        self.mol_list.append(mols)
        self.mol_list = list(flatten(self.mol_list))

        internal_drawer = MolDrawer()
        self.img_list.append(
            [internal_drawer.draw_mol(mol) for mol in self.mol_list]
        )
        self.img_list = list(flatten(self.img_list))

    def show_images_grid(
        self,
        images: Iterable[Image.Image] = None,
        n_columns: int = 4,
        size: Iterable = None,
        buffer: int = 5,
        empty_tile_colour: str = 'white',
        save: bool = False,
        filename: str = ''
    ) -> None:
        """
Display a set of images as a tiled grid.

If `images` is `None`, the method uses `self.img_list`. The output is shown
with `IPython.display.display` and can optionally be saved.

Parameters
----------
images : Iterable[PIL.Image.Image], optional
    Images to arrange. If `None`, uses `self.img_list`.
n_columns : int, default=4
    Number of columns in the grid layout.
size : Iterable[int, int], optional
    Size of each image in pixels as (width, height). If None, uses the
    default size from `self.size`.
buffer : int, default=5
    Space in pixels between images in the grid.
empty_tile_colour : str, default='white'
    Background colour for empty grid tiles. Must be a key in
    `self.colour_dictionary`.
save : bool, default=False
    If True, saves the grid image to a file.
filename : str, default=''
    Filename to save the image if `save` is True.

Returns
-------
None

Raises
------
AssertionError
    If `empty_tile_colour` is not in the colour dictionary or if `size`
    is not a 2-element iterable.

Examples
--------
>>> drawer = MolDrawer()
>>> drawer.load_images([img1, img2, img3])
>>> drawer.show_images_grid(n_columns=2, buffer=10)

>>> drawer.show_images_grid(save=True, filename="grid_output.png")
"""


        from IPython.display import display

        assert empty_tile_colour in self.colour_dictionary.keys(), (
            "'%s' is not a known colour.\nFor an extensive list"
            "of accepted colours, look at the 'colour_dictionary.keys()'"
            "attribute of the class." % empty_tile_colour)

        if size is None:
            size = self.size
        assert len(size) == 2, "'size' argument must have lenght == 2."

        if images is None:
            images = self.img_list

        img_width, img_height = size

        # Calculate the number of rows needed
        n_images = len(images)
        n_rows = (n_images + n_columns - 1) // n_columns

        grid_width = img_width * n_columns + buffer * (n_columns - 1)
        grid_height = img_height * n_rows + buffer * (n_rows - 1)
        grid_img = Image.new('RGB', (grid_width, grid_height),
                             empty_tile_colour)

        # Paste images into grid
        for i, img in enumerate(images):
            x = (i % n_columns) * (img_width + buffer)
            y = (i // n_columns) * (img_height + buffer)
            img_resized = img.resize(size)
            grid_img.paste(img_resized, (x, y))
        if save:
            grid_img.save(filename)

        display(grid_img)

    def draw_mol(self,
                 mol: Chem.rdchem.Mol = None,
                 legend: str = '',
                 highlightAtoms: Iterable = [],
                 size: Iterable = None,
                 ACS1996_mode: bool = False
                 ) -> Image.Image:
        """
Render and return a single molecule image.

The method applies the active `drawing_options`, supports atom highlighting,
optional contour visualization of atomic weights (`mapStyle='GC'` or `'C'` 
when `atomWeights` are provided), optional custom shape overlays, and ACS 1996 
styling.

**Contour Visualization (Atomic Weights)**

When `atomWeights` is provided in drawing_options, mlchem can visualize these 
values using two different contour styles:

- `mapStyle='GC'` (Gaussian Contours): Renders smooth gaussian overlays using 
  RDKit's ContourAndDrawGaussians. Best for continuous properties like 
  electrostatic potential or similarity gradients. Contours blend smoothly 
  across the molecular surface.

- `mapStyle='C'` (Circle Contours): Custom mlchem implementation using gradient 
  circles around atoms. Best for discrete atom-by-atom properties like 
  per-atom charges or SHAP contributions. Provides clear visual separation 
  between atoms.

This extends RDKit's native gaussian contour support with:
- A second visualization style (circle contours)
- Seamless integration with all other drawing options
- Proper legend handling (solves RDKit's alignment issues)
- Fine-grained color control via atomNoteColour, legendColour, etc.
- Automatic 2D coordinate handling

Parameters
----------
mol : rdkit.Chem.rdchem.Mol, optional
    Molecule to draw. If `None`, `self.mol` is used.
legend : str, optional
    Legend text. If empty, `self.legend` is used.
highlightAtoms : Iterable, optional
    Atom indices to highlight. If empty, `self.highlightAtoms` is used.
size : Iterable[int, int], optional
    Canvas size `(width, height)`. If `None`, `self.size` is used.
ACS1996_mode : bool, default=False
    If `True`, draw using `Draw.DrawMoleculeACS1996`.

Returns
-------
PIL.Image.Image
    Rendered molecule image.

Raises
------
AssertionError
    If no molecule is available (`mol is None` and `self.mol is None`).
ValueError
    If a named colour is unknown.
TypeError
    If provided colour tuples/palettes are not valid.

Examples
--------
>>> mol = Chem.MolFromSmiles("CCO")
>>> drawer = MolDrawer()
>>> img = drawer.draw_mol(mol, legend="Ethanol", highlightAtoms=[1])

Draw with gaussian contour similarity map (smooth overlay):

>>> from mlchem.chem.manipulation import PropManager as PM
>>> weights = PM.Mol.get_gasteiger_charges(mol)
>>> drawer.update_drawing_options(atomWeights=weights, mapStyle='GC')
>>> img = drawer.draw_mol(mol, legend="Gaussian contours")

Draw with circle-style weight map (atom-by-atom visualization):

>>> drawer.update_drawing_options(atomWeights=weights, mapStyle='C')
>>> img = drawer.draw_mol(mol, legend="Circle contours")
"""

        from rdkit.Chem import Draw
        from rdkit.Chem import rdDepictor
        from mlchem.helper import (make_rgb_transparent,
                                   show_png,
                                   convert_rgb,
                                   create_smooth_gradient_circle)
        import io
        import numpy as np
        # Keep track of mols, highlightAtoms, sizes, legends used
        if mol is None:
            assert self.mol is not None, "No valid molecule was passed."
            mol = self.mol

        if not mol.GetNumConformers():
            rdDepictor.Compute2DCoords(mol)

        if len(highlightAtoms) == 0:
            highlightAtoms = self.highlightAtoms
        if type(highlightAtoms) == tuple:
            highlightAtoms = list(highlightAtoms)
        elif type(highlightAtoms) == np.ndarray:
            highlightAtoms = np.ndarray.tolist(highlightAtoms)
        self.highlightAtoms_list.append(highlightAtoms)

        if size is None:
            size = self.size
        self.size_list.append(size)

        if legend == '':
            legend = self.legend
        self.legend_list.append(legend)

        d2d = Draw.MolDraw2DCairo(size[0], size[1])
        dopts = d2d.drawOptions()

        # apply drawing options #

        atom_weights = self.drawing_options['atomWeights']
        if atom_weights and self.drawing_options['mapStyle'] == 'C':
            self.drawing_options['padding'] = 0.07 \
                * self.drawing_options['scalingFactor']

        ## all options a function is not needed for ##

        # colour options refusing the 'setattr()' method

        COLOUR_OPTIONS = {
            "backgroundColour": dopts.setBackgroundColour,
            "highlightColour": dopts.setHighlightColour,
            "queryColour": dopts.setQueryColour,
            "annotationColour": dopts.setAnnotationColour,
            "atomNoteColour": dopts.setAtomNoteColour,
            "bondNoteColour": dopts.setBondNoteColour,
            "legendColour": dopts.setLegendColour,
            "symbolColour": dopts.setSymbolColour,
            "variableAttachmentColour": dopts.setVariableAttachmentColour,
        }


        for option, value in self.drawing_options.items():

            if option in COLOUR_OPTIONS:     # 
                continue

            if hasattr(dopts, option):
                try:
                    setattr(dopts, option, value)
                except Exception:
                    print("Problem encountered with the '%s' option. "
                          "Please disable it in the original definition"
                          " in self.drawing_options attribute." % option)

        background_colour = self.drawing_options['backgroundColour']
        assert (isinstance(background_colour, str) or
                isinstance(background_colour, tuple)), (
                    "Background colour must be a valid string or tuple."
                )
        if isinstance(background_colour, str):
            if background_colour in self.colour_dictionary.keys():
                background_tuple = self.colour_dictionary[background_colour]
                dopts.setBackgroundColour(background_tuple)
            else:
                raise ValueError(
                    "An improper colour string was passed. '%s' is not"
                    " a valid colour in mlchem.importables."
                    "colour_dictionary palette." %
                    background_colour)
        else:
            try:
                background_tuple = background_colour
                dopts.setBackgroundColour(background_tuple)
            except Exception:
                raise TypeError(
                    "An improper colour tuple was passed. Correct "
                    "custom palette has to be: (R,G,B)\nExample: "
                    "(0.7,0.0,0.7)) will set overwrite colour "
                    "to purple.")

        # all other options a function is needed for

        chosen_palette = self.drawing_options['atomPalette']
        assert (
            chosen_palette in ('avalon', 'cdk', 'bw') or
            isinstance(chosen_palette, dict)
            ), (
                "'atomPalette' property must be one of the following:\n"
                "'avalon', 'cdk', 'bw', or a dict(atomic_number:"
                "(R,G,B),)"
                )

        if chosen_palette == 'avalon':
            dopts.useAvalonAtomPalette()
        if chosen_palette == 'cdk':
            dopts.useCDKAtomPalette()
        if chosen_palette == 'bw':
            dopts.useBWAtomPalette()
        if isinstance(chosen_palette, dict):
            try:
                dopts.updateAtomPalette(chosen_palette)
            except Exception:
                raise TypeError(
                    "An improper palette dictionary was passed. Correct custom"
                    " palette has to be: dict(atomic_number:(R,G,B),)"
                    "\nexample:\n""dict(6: (0.7,0.0,0.7)) will overwrite"
                    " carbon black colour with purple.")

        highlight_colour = self.drawing_options['highlightColour']
        assert (isinstance(highlight_colour, str) or
                isinstance(highlight_colour, tuple)), (
                    "Highlight colour must be a valid string or tuple."
                    )
        if isinstance(highlight_colour, str):
            if highlight_colour in self.colour_dictionary.keys():
                highlight_tuple = self.colour_dictionary[highlight_colour]
                colour_tuple = make_rgb_transparent(
                    highlight_tuple,
                    background_tuple,
                    self.drawing_options['highlightAlpha']
                    )
                dopts.setHighlightColour(colour_tuple)
            else:
                raise ValueError(
                    "An improper colour string was passed. '%s' is not a valid"
                    " colour in mlchem.importables."
                    "colour_dictionary palette." %
                    highlight_colour)
        else:
            try:
                highlight_tuple = highlight_colour
                dopts.setHighlightColour(highlight_tuple)
            except Exception:
                raise TypeError(
                    "An improper colour tuple was passed. "
                    "Correct custom palette has to be: (R,G,B)\nExample: "
                    "(0.7,0.0,0.7)) will set overwrite colour "
                    "to purple.")

        query_colour = self.drawing_options['queryColour']
        assert (isinstance(query_colour, str) or
                isinstance(query_colour, tuple)), (
                    "Query colour must be a valid string or tuple."
                    )
        if isinstance(query_colour, str):
            if query_colour in self.colour_dictionary.keys():
                query_tuple = self.colour_dictionary[query_colour]
                dopts.setQueryColour(query_tuple)
            else:
                raise ValueError(
                    "An improper colour string was passed. '%s' is not a "
                    "valid colour in mlchem.importables."
                    "colour_dictionary palette." %
                    query_colour)

        else:
            try:
                query_tuple = query_colour
                dopts.setQueryColour(query_tuple)
            except Exception:
                raise ValueError(
                    "An improper colour tuple was passed. Correct custom"
                    "palette has to be: (R,G,B)\nExample: (0.7,0.0,0.7)) will"
                    " set overwrite colour to purple.")

        annotation_colour = self.drawing_options['annotationColour']
        assert (isinstance(annotation_colour, str) or
                isinstance(annotation_colour, tuple)), (
                    "Annotation colour must be a valid string or tuple."
                    )
        if isinstance(annotation_colour, str):
            if annotation_colour in self.colour_dictionary.keys():
                annotation_tuple = self.colour_dictionary[annotation_colour]
                dopts.setAnnotationColour(annotation_tuple)
            else:
                raise ValueError
            ("An improper colour string was passed. '%s' is not a valid "
             "colour in mlchem.importables.colour_dictionary palette." %
             annotation_colour)

        else:
            try:
                annotation_tuple = annotation_colour
                dopts.setAnnotationColour(annotation_colour)
            except Exception:
                raise ValueError
            ("An improper colour tuple was passed. Correct custom palette "
             "has to be: (R,G,B)\nExample: (0.7,0.0,0.7)) "
             "will set overwrite colour ""to purple.")

        # process atomNoteColour
        atom_note_colour = self.drawing_options['atomNoteColour']
        assert (isinstance(atom_note_colour, str) or
                isinstance(atom_note_colour, tuple)), (
                    "Atom note colour must be a valid string or tuple."
                    )
        if isinstance(atom_note_colour, str):
            if atom_note_colour in self.colour_dictionary.keys():
                atom_note_tuple = self.colour_dictionary[atom_note_colour]
                dopts.setAtomNoteColour(atom_note_tuple)
            else:
                raise ValueError(
                    "An improper colour string was passed. '%s' is not a valid "
                    "colour in mlchem.importables.colour_dictionary palette." %
                    atom_note_colour)
        else:
            try:
                dopts.setAtomNoteColour(atom_note_colour)
            except Exception:
                raise ValueError(
                    "An improper colour tuple was passed. Correct custom palette "
                    "has to be: (R,G,B)\nExample: (0.7,0.0,0.7)) "
                    "will set atom note colour to purple.")

        # process bondNoteColour
        bond_note_colour = self.drawing_options['bondNoteColour']
        assert (isinstance(bond_note_colour, str) or
                isinstance(bond_note_colour, tuple)), (
                    "Bond note colour must be a valid string or tuple."
                    )
        if isinstance(bond_note_colour, str):
            if bond_note_colour in self.colour_dictionary.keys():
                bond_note_tuple = self.colour_dictionary[bond_note_colour]
                dopts.setBondNoteColour(bond_note_tuple)
            else:
                raise ValueError(
                    "An improper colour string was passed. '%s' is not a valid "
                    "colour in mlchem.importables.colour_dictionary palette." %
                    bond_note_colour)
        else:
            try:
                dopts.setBondNoteColour(bond_note_colour)
            except Exception:
                raise ValueError(
                    "An improper colour tuple was passed. Correct custom palette "
                    "has to be: (R,G,B)\nExample: (0.7,0.0,0.7)) "
                    "will set bond note colour to purple.")

        # process legendColour
        legend_colour = self.drawing_options['legendColour']
        assert (isinstance(legend_colour, str) or
                isinstance(legend_colour, tuple)), (
                    "Legend colour must be a valid string or tuple."
                    )
        if isinstance(legend_colour, str):
            if legend_colour in self.colour_dictionary.keys():
                legend_tuple = self.colour_dictionary[legend_colour]
                dopts.setLegendColour(legend_tuple)
            else:
                raise ValueError(
                    "An improper colour string was passed. '%s' is not a valid "
                    "colour in mlchem.importables.colour_dictionary palette." %
                    legend_colour)
        else:
            try:
                dopts.setLegendColour(legend_colour)
            except Exception:
                raise ValueError(
                    "An improper colour tuple was passed. Correct custom palette "
                    "has to be: (R,G,B)\nExample: (0.7,0.0,0.7)) "
                    "will set legend colour to purple.")

        # process symbolColour
        symbol_colour = self.drawing_options['symbolColour']
        assert (isinstance(symbol_colour, str) or
                isinstance(symbol_colour, tuple)), (
                    "Symbol colour must be a valid string or tuple."
                    )
        if isinstance(symbol_colour, str):
            if symbol_colour in self.colour_dictionary.keys():
                symbol_tuple = self.colour_dictionary[symbol_colour]
                dopts.setSymbolColour(symbol_tuple)
            else:
                raise ValueError(
                    "An improper colour string was passed. '%s' is not a valid "
                    "colour in mlchem.importables.colour_dictionary palette." %
                    symbol_colour)
        else:
            try:
                dopts.setSymbolColour(symbol_colour)
            except Exception:
                raise ValueError(
                    "An improper colour tuple was passed. Correct custom palette "
                    "has to be: (R,G,B)\nExample: (0.7,0.0,0.7)) "
                    "will set symbol colour to purple.")

        # process variableAttachmentColour
        variable_attachment_colour = self.drawing_options['variableAttachmentColour']
        assert (isinstance(variable_attachment_colour, str) or
                isinstance(variable_attachment_colour, tuple)), (
                    "Variable attachment colour must be a valid string or tuple."
                    )
        if isinstance(variable_attachment_colour, str):
            if variable_attachment_colour in self.colour_dictionary.keys():
                variable_attachment_tuple = self.colour_dictionary[variable_attachment_colour]
                dopts.setVariableAttachmentColour(variable_attachment_tuple)
            else:
                raise ValueError(
                    "An improper colour string was passed. '%s' is not a valid "
                    "colour in mlchem.importables.colour_dictionary palette." %
                    variable_attachment_colour)
        else:
            try:
                dopts.setVariableAttachmentColour(variable_attachment_colour)
            except Exception:
                raise ValueError(
                    "An improper colour tuple was passed. Correct custom palette "
                    "has to be: (R,G,B)\nExample: (0.7,0.0,0.7)) "
                    "will set variable attachment colour to purple.")

        if ACS1996_mode is True:
            d2d = Draw.MolDraw2DCairo(-1, -1)
            Draw.DrawMoleculeACS1996(d2d, mol, legend=legend)
            d2d.FinishDrawing()
            self.DrawingText = d2d.GetDrawingText()
            return show_png(self.DrawingText)
        else:
            # Draw molecule for non-GC styles
            # For GC, let SimMaps handle both molecule and contours
            if not (atom_weights and self.drawing_options['mapStyle'] == 'GC'):
                d2d.DrawMolecule(mol, legend=legend, highlightAtoms=highlightAtoms)
            
            # Shapes will be drawn via PIL after rendering for consistent coordinate transformation
            
            if atom_weights and self.drawing_options['mapStyle'] == 'C':
                max_weight = max(atom_weights)
                min_weight = min(atom_weights)
                normalised_weights = [
                    (weight / max_weight if weight > 0 else
                     weight / abs(min_weight)) if
                    max_weight is not min_weight else
                    0 for weight in atom_weights
                     ]

                # Get the atomic coordinates
                atom_coords = {
                    atom.GetIdx():
                    d2d.GetDrawCoords(atom.GetIdx()) for
                    atom in mol.GetAtoms()
                    }

                # Create a base image from d2d (now includes shapes for non-GC)
                molecule_image = Image.open(
                    io.BytesIO(
                        d2d.GetDrawingText()
                        )
                        )

                # Draw concentric smooth gradient circles around each atom
                from PIL import ImageDraw
                for i, atom in enumerate(mol.GetAtoms()):
                    x, y = atom_coords[atom.GetIdx()]
                    weight = atom_weights[i]
                    if weight == 0:
                        continue     # Skip fully transparent circles

                    # Determine color and alpha based on weight
                    weight_baseline_alpha = self.drawing_options['weightAlpha']
                    if weight < 0:
                        weight_neg_colour = self.\
                          drawing_options['negativeColour']
                        colour = convert_rgb(
                            self.colour_dictionary[weight_neg_colour],
                            'denormalise'
                            )
                        normalised_alpha = weight_baseline_alpha * abs(
                            normalised_weights[i]
                            )
                    else:
                        weight_pos_colour = self.\
                          drawing_options['positiveColour']
                        colour = convert_rgb(
                            self.colour_dictionary[weight_pos_colour],
                            'denormalise'
                            )
                        normalised_alpha = weight_baseline_alpha * \
                            normalised_weights[i]

                    # Calculate radius and ensure it is within min and max bounds
                    weight_scaling_factor = self.\
                      drawing_options['scalingFactor']
                    weight_min_radius = self.drawing_options['minRadius']
                    weight_max_radius = self.drawing_options['maxRadius']
                    radius = weight_scaling_factor * abs(weight)
                    radius = max(
                        weight_min_radius,
                        min(radius, weight_max_radius)
                            )

                    weight_num_circles = self.drawing_options['numContours']
                    for j in range(1, weight_num_circles + 1):
                        gradient_circle = create_smooth_gradient_circle(
                            int(radius * j),
                            colour,
                            normalised_alpha
                            )
                        molecule_image.paste(
                            gradient_circle,
                            (int(x - radius * j),
                                int(y - radius * j)), gradient_circle)
                return molecule_image

            # Draw gaussian contours for GC mapStyle
            if atom_weights and self.drawing_options['mapStyle'] == 'GC':
                from mlchem.chem.visualise.simmaps import SimMaps as SM
                
                resolution = self.drawing_options['mapRes']
                if self.drawing_options['colourMap'] is None:
                    map_background_colour = self.drawing_options['backgroundColour']
                    map_negative_colour = self.drawing_options['negativeColour']
                    map_positive_colour = self.drawing_options['positiveColour']
                    
                    map_background_tuple = self.colour_dictionary[map_background_colour] if isinstance(map_background_colour, str) else map_background_colour
                    map_negative_tuple = self.colour_dictionary[map_negative_colour] if isinstance(map_negative_colour, str) else map_negative_colour
                    map_positive_tuple = self.colour_dictionary[map_positive_colour] if isinstance(map_positive_colour, str) else map_positive_colour
                    
                    colourMap = [
                        map_negative_tuple,
                        map_background_tuple,
                        map_positive_tuple
                    ]
                else:
                    colourMap = self.drawing_options['colourMap']
                
                d2d = SM.get_similarity_map_from_weights(
                    mol=mol,
                    weights=atom_weights,
                    draw2d=d2d,
                    resolution=resolution,
                    contourLines=self.drawing_options['numContours'],
                    contour_width=self.drawing_options['contourWidth'],
                    colorMap=colourMap,
                    dash_negative=self.drawing_options['dashNegative'],
                    contour_colour=self.drawing_options['contourColour'],
                    draw_molecule=True,  # Always draw molecule + contours for proper rendering
                    legend=legend)

            d2d.FinishDrawing()
            self.DrawingText = d2d.GetDrawingText()
            
            # For GC contours with legend, add legend manually using PIL
            # to prevent RDKit from shifting the molecule and breaking contour alignment
            if atom_weights and self.drawing_options['mapStyle'] == 'GC' and legend:
                import io
                img = Image.open(io.BytesIO(self.DrawingText))
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img)
                
                # Try to use a reasonable font size based on legendFontSize
                legend_font_size = self.drawing_options['legendFontSize']
                try:
                    # Try to load a default font at the specified size
                    font = ImageFont.truetype("arial.ttf", legend_font_size)
                except (IOError, OSError):
                    # Fall back to default font if arial is not available
                    font = ImageFont.load_default()
                
                # Calculate text position (centered at bottom)
                bbox = draw.textbbox((0, 0), legend, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (img.width - text_width) // 2
                y = img.height - text_height - 10
                
                # Get legend color
                legend_colour = self.drawing_options['legendColour']
                if isinstance(legend_colour, str):
                    legend_tuple = self.colour_dictionary[legend_colour]
                else:
                    legend_tuple = legend_colour
                # Convert from 0-1 to 0-255
                legend_rgb = tuple(int(c * 255) for c in legend_tuple)
                
                # Draw the legend text
                draw.text((x, y), legend, fill=legend_rgb, font=font)
                
                # Convert back to PNG bytes
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                self.DrawingText = buffer.getvalue()
            
            return show_png(self.DrawingText)
        

