from typing import Iterable
from rdkit import Chem
from PIL import Image


class MolDrawer:
    """
MolDrawer is a utility class for rendering molecular structures with
customisable visual styles and annotations.

This class provides a flexible interface for drawing molecules using RDKit,
with support for highlighting atoms, adjusting drawing styles, displaying
legends, and visualising similarity or weight maps.

Attributes
----------
mol : rdkit.Chem.rdchem.Mol or None
    The molecule to be drawn.
highlightAtoms : list
    List of atom indices to highlight.
size : list[int, int]
    Canvas size in pixels as [width, height].
legend : str
    Text to display as a legend below the molecule.
colour_dictionary : dict
    Predefined colour palette for drawing.
drawing_options : dict
    Dictionary of drawing parameters and visual styles.

Methods
-------
__init__(...)
    Initialise the MolDrawer with molecule, size, highlights, and legend.
show_palette(...)
    Display or save the colour palette used for drawing.
update_drawing_options(...)
    Update drawing options using a dictionary or keyword arguments.
reset_drawing_options()
    Reset drawing options to their default values.
draw_mol(...)
    Render a single molecule with the current drawing settings.
load_images(...)
    Load external images into the drawer.
load_mols(...)
    Load molecules and generate their images.
show_images_grid(...)
    Display all loaded images in a grid layout.
"""

    def __init__(self,
                 mol: Chem.rdchem.Mol | None = None,
                 highlightAtoms: Iterable = [],
                 size: Iterable = [300, 300],
                 legend: str = '') -> None:
        """
Initialise the MolDrawer class for molecular visualisation.

This class sets up the molecule, drawing canvas, highlighting options,
and default drawing styles for rendering molecular structures.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol or None, optional
    The molecule to be drawn. Default is None.
highlightAtoms : Iterable, optional
    List of atom indices to highlight. Default is an empty list.
size : Iterable, optional
    Canvas size as (width, height) in pixels. Default is [300, 300].
legend : str, optional
    Text to display as a legend below the molecule. Default is an empty string.

Returns
-------
None
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

        self.drawing_options = {


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

            # The color used for molecule, atom, bond, and SGroup notes
            'annotationColour': 'black',

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
            # (GC or C; Gaussian Contours or Circles)
            'mapStyle': 'GC',

            # Colour map for similarity maps; default is None.
            # A list of 3 tuples/colour names is accepted too.
            'colourMap': None,

            # Colour of atoms having positive weights (style: both)
            'positiveColour': 'green',

            # Colour of atoms having negative weights (style: both)
            'negativeColour': 'mediumvioletred',

            # List of atom indices with numerical property
            # to display (style: both)
            'atomWeights': [],

            # Baseline alpha of weight colour (style: both)
            'weightAlpha': 0.2,

            # Circle radius scale (style: circles)
            'scalingFactor': 2,

            # Minimum circle radius (style: circles)
            'minRadius': 2,

            # Maximum circle radius (style: circles)
            'maxRadius': 30,

            # Number of concentrical circles per atom (style: both)
            'numContours': 10,

            # Line width of the contours (style: gaussian contours)
            'contourWidth': 1,

            # Resolution of gaussian contours (style: gaussian contours)
            'mapRes': 0.05,

            # Contour colour (style: gaussian contours)
            'contourColour': 'black',

            # Whether to display negative weights as dashed
            'dashNegative' : True,


            # Optional shapes #


            # Sets the type of shape to render. Current choices are: 'circle'.
            'shapeTypes': [],

            # Sets the size of the shape to render. Int or float
            # values are accepted.
            'shapeSizes': [],

            # Sets the colour of the shapes to draw.
            # Every colour in the iterable should be either a string
            # (any colour present in the 'colour_dictionary')
            # or a RGB/RGBA tuple
            'shapeColours': [],

            # Sets the 2D coordinates of the shapes. Accepts an
            # iterable per shape.
            'shapeCoords': [],


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

    def show_palette(
        self, palette: dict | None = None, save: bool = False,
        filename: str = '', size: Iterable = [1000, 300]
    ) -> Image.Image:
        """
Display the colour palette used for molecular drawings.

This method visualises the colour palette as an image. If no palette is
provided, the default `colour_dictionary` is used. The image can also be
saved to a file.

Parameters
----------
palette : dict or None, optional
    A dictionary mapping colour names to RGB values. If None, uses the
    default palette from `self.colour_dictionary`.
save : bool, optional
    If True, saves the palette image to a file. Default is False.
filename : str, optional
    Filename to save the image if `save` is True. Default is an empty string.
size : Iterable, optional
    Size of the image in pixels as (width, height). Default is [1000, 300].

Returns
-------
PIL.Image.Image
    The image object representing the colour palette.

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
Update the RDKit drawing options.

This method allows customisation of the molecule rendering style by updating
the internal drawing options dictionary. Users can pass either a full or
partial dictionary of options, or use keyword arguments directly.

Parameters
----------
**args : dict
    Keyword arguments representing drawing options. These can include:

    Colours
    -------
    - atomPalette : str or dict, default='cdk'
        Atom colour scheme. Options: 'avalon', 'cdk', 'bw', or a dictionary
        mapping atomic numbers to RGB tuples.
    - backgroundColour : str or tuple, default='white'
        Background colour of the canvas.
    - highlightColour : str or tuple, default='tomato'
        Colour used for highlighting atoms or bonds.
    - highlightAlpha : float, default=1
        Transparency of the highlight colour (0 = transparent, 1 = opaque).
    - queryColour : str or tuple, default='red'
        Colour used for SMARTS query atoms.
    - annotationColour : str or tuple, default='black'
        Colour for annotations (e.g., atom/bond notes).

    Drawing Style Options
    ---------------------
    - dummiesAreAttachments : bool, default=False
    - addAtomIndices : bool, default=False
    - addBondIndices : bool, default=False
    - noAtomLabels : bool, default=False
    - explicitMethyl : bool, default=False
    - includeRadicals : bool, default=True
    - useComplexQueryAtomSymbols : bool, default=True
    - singleColourWedgeBonds : bool, default=False
    - drawMolsSameScale : bool, default=False
    Highlighting
    ------------
    - continuousHighlight : bool, default=True
    - circleAtoms : bool, default=True
    - atomHighlightsAreCircles : bool, default=True
    - fillHighlights : bool, default=True
    - highlightRadius : float, default=0.3
    - highlightBondWidthMultiplier : float, default=10

    Stereochemistry
    ---------------
    - addStereoAnnotation : bool, default=False
    - unspecifiedStereoIsUnknown : bool, default=False

    Fonts and Text
    --------------
    - baseFontSize : float, default=0.6
    - annotationFontScale : float, default=0.5
    - legendFontSize : int, default=25
    - fixedFontSize : int, default=-1
    - minFontSize : int, default=6
    - maxFontSize : int, default=40
    - fontFile : str, default=''
        Path to a custom font file.

    Bond Drawing Parameters
    -----------------------
    - multipleBondOffset : float, default=0.15
    - additionalAtomLabelPadding : float, default=0
    - bondLineWidth : float, default=2
    - scaleBondWidth : bool, default=False
    - scaleHighlightBondWidth : bool, default=True
    - fixedBondLength : float, default=-1

    Similarity Maps Parameters
    --------------------------
    - atomWeights : list, default=[]
    - mapStyle : {'GC', 'C'}, default='GC'
    - colourMap : str or list of RGB tuples, default=None
    - positiveColour : str or tuple, default='green'
    - negativeColour : str or tuple, default='mediumvioletred'
    - numContours : int, default=10
    - weightAlpha : float, default=0.2
    - scalingFactor : float, default=2
    - minRadius : float, default=2
    - maxRadius : float, default=30
    - mapRes : float, default=0.5
    - contourColour : str or tuple, default='black'
    - contourWidth : float, default=1
    - dashNegative : bool, default=True

    Optional Shapes
    ---------------
    - shapeTypes : list, default=[]
    - shapeSizes : list, default=[]
    - shapeColours : list, default=[]
    - shapeCoords : list, default=[]

    Miscellaneous
    -------------
    - clearBackground : bool, default=True
    - prepareMolsBeforeDrawing : bool, default=True
    - rotate : float, default=0
    - padding : float, default=0.05
    - atomLabelDeuteriumTritium : bool, default=False
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

    def reset_drawing_options(self) -> None:
        """
Reset the drawing options to their default values.

This method reinitialises the internal drawing options dictionary to the
default configuration defined in a fresh instance of the `MolDrawer` class.
It is useful when you want to discard all customisations and revert to the
original visual style.

Returns
-------
None

Examples
--------
>>> drawer = MolDrawer()
>>> drawer.update_drawing_options(atomPalette='avalon', rotate=90)
>>> drawer.reset_drawing_options()  # Reverts to default settings
"""

        internal_drawer = MolDrawer()
        self.drawing_options = internal_drawer.drawing_options

    def load_images(self, img_list:
                    Iterable[Image.Image]
                    | Image.Image) -> None:
        """
Load images into the drawer without adding molecules.

This method allows you to load one or more pre-rendered images into the
MolDrawer instance. These images can be used for visualisation, layout
composition, or exporting as part of a grid.
This method is equivalent to overwrite the img_list attribute.

Parameters
----------
img_list : PIL.Image.Image or Iterable[PIL.Image.Image]
    A single image or a list of images to be added to the drawer.

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
Load molecules and generate standard images in the drawer.

This method accepts one or more RDKit molecule objects, stores them in the
drawer, and generates their corresponding images using the default or
customised drawing options. These images are stored internally and can be
displayed later in a grid or individually.

Parameters
----------
mols : rdkit.Chem.rdchem.Mol or Iterable[rdkit.Chem.rdchem.Mol]
    A single molecule or an iterable of molecules to be loaded and rendered.

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
Display all images arranged in a grid layout.

This method arranges a list of images into a grid and displays them using
IPython. It supports customisation of the number of columns, image size,
spacing between images, and background colour for empty tiles. Optionally,
the grid can be saved to a file.

Parameters
----------
images : Iterable[PIL.Image.Image], optional
    A list of images to be arranged in the grid. If None, uses the images
    stored in `self.img_list`.
n_columns : int, default=4
    Number of columns in the grid layout.
size : Iterable[int, int], optional
    Size of each image in pixels as (width, height). If None, uses the
    default size from `self.size`.
buffer : int, default=5
    Space in pixels between images in the grid.
empty_tile_colour : str, default='white'
    Background colour for empty tiles. Must be a valid key in the
    `colour_dictionary` attribute.
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
Render and return an image of a single molecule using the current drawing options.

This method draws a molecule with optional atom highlighting, legend text,
custom canvas size, and support for ACS 1996-style rendering. It uses the
drawing configuration stored in the `MolDrawer` instance.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol, optional
    The molecule to be drawn. If None, uses the molecule stored in `self.mol`.
legend : str, optional
    Text to display below the molecule as a legend. Default is an empty string.
highlightAtoms : Iterable, optional
    List of atom indices to highlight. Default is an empty list.
size : Iterable[int, int], optional
    Canvas size in pixels as (width, height). If None, uses `self.size`.
ACS1996_mode : bool, default=False
    If True, applies ACS 1996-style drawing conventions.

Returns
-------
PIL.Image.Image
    The rendered image of the molecule.

Examples
--------
>>> mol = Chem.MolFromSmiles("CCO")
>>> drawer = MolDrawer()
>>> img = drawer.draw_mol(mol, legend="Ethanol", highlightAtoms=[1])
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

        # Apply similaritymap GaussianContour options
        atom_weights = self.drawing_options['atomWeights']
        resolution = self.drawing_options['mapRes']
        if atom_weights and self.drawing_options['mapStyle'] == 'GC':
            

            from mlchem.chem.visualise.simmaps import SimMaps as SM

            if self.drawing_options['colourMap'] is None:
                map_background_colour = self.\
                  drawing_options['backgroundColour']
                map_negative_colour = self.\
                  drawing_options['negativeColour']
                map_positive_colour = self.\
                  drawing_options['positiveColour']

                # Background colour

                assert (isinstance(map_background_colour, str) or
                        isinstance(map_background_colour, Iterable)), (
                            "Colour map background colour must be"
                            "a valid string or Iterable.")
                if isinstance(map_background_colour, str):
                    if \
                       map_background_colour in self.colour_dictionary.keys():
                        map_background_tuple = self.colour_dictionary[
                            map_background_colour
                            ]
                    else:
                        raise ValueError(
                            "An improper colour string was passed. '%s' is not"
                            "a valid colour in mlchem.importables."
                            "colour_dictionary palette." %
                            map_background_colour)

                else:
                    assert len(map_background_colour) == 3, (
                        "Colour iterable must have 3 elements (RGB)."
                        )

                    assert all(isinstance(c, (int, float)) for c in
                               map_background_colour), (
                        "Some elements of the colour RGB iterable "
                        "are not numbers."
                    )

                    map_background_tuple = map_background_colour

                # Negative colour

                assert (isinstance(map_negative_colour, str) or
                        isinstance(map_negative_colour, Iterable))
                if isinstance(map_negative_colour, str):
                    if map_negative_colour in self.colour_dictionary.keys():
                        map_negative_tuple = self.colour_dictionary[
                            map_negative_colour]
                    else:
                        raise ValueError(
                            "An improper colour string was passed."
                            " '%s' is not a valid colour in "
                            "mlchem.importables.colour_dictionary palette." %
                            map_negative_colour)

                else:
                    assert len(map_negative_colour) == 3, (
                        "Colour iterable must have 3 elements (RGB)."
                        )

                    assert all(isinstance(c, (int, float)) for c in
                               map_negative_colour), (
                        "Some elements of the colour RGB iterable are"
                        " not numbers."
                    )

                    map_negative_tuple = map_negative_colour

                # Positive colour

                assert (isinstance(map_positive_colour, str) or
                        isinstance(map_positive_colour, Iterable))
                if isinstance(map_positive_colour, str):
                    if map_positive_colour in self.colour_dictionary.keys():
                        map_positive_tuple = self.colour_dictionary[
                            map_positive_colour]
                    else:
                        raise ValueError(
                            "An improper colour string was passed."
                            " '%s' is not a valid colour in "
                            "mlchem.importables.colour_dictionary palette." %
                            map_positive_colour)

                else:
                    assert len(map_positive_colour) == 3, (
                        "Colour iterable must have 3 elements (RGB)."
                        )

                    assert all(isinstance(c, (int, float)) for c in
                               map_positive_colour), (
                        "Some elements of the colour RGB iterable "
                        "are not numbers."
                    )

                    map_positive_tuple = map_positive_colour

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
                contour_colour=self.drawing_options['contourColour'])
        dopts = d2d.drawOptions()

        # apply drawing options #

        atom_weights = self.drawing_options['atomWeights']
        if atom_weights:
            self.drawing_options['padding'] = 0.07 \
                * self.drawing_options['scalingFactor']

        # all options a function is not needed for

        for option, value in self.drawing_options.items():
            if hasattr(dopts, option):
                try:
                    setattr(dopts, option, value)
                except Exception:
                    print("Problem encountered with the '%s' option. "
                          "Please disable it in the original definition"
                          " in self.drawing_options attribute.")

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

        # draw custom shapes

        shape_types = self.drawing_options['shapeTypes']
        shape_sizes = self.drawing_options['shapeSizes']
        shape_colours = self.drawing_options['shapeColours']
        shape_coords = self.drawing_options['shapeCoords']
        if (len(shape_types) > 0 and
            len(shape_sizes) > 0 and
            len(shape_colours) > 0 and
            len(shape_coords) > 0) and (
                len(shape_types) ==
                len(shape_sizes) ==
                len(shape_colours) ==
                len(shape_coords)
                                        ):

            from rdkit.Geometry import Point2D
            import numpy as np

            d2d.DrawMolecule(mol)

            for typ, siz, col, pos in zip(
                shape_types,
                shape_sizes,
                shape_colours,
                shape_coords,
                            ):

                assert isinstance(typ, str)
                assert (isinstance(siz, float) or
                        isinstance(siz, int))
                assert (isinstance(col, str) or
                        isinstance(col, Iterable))
                assert isinstance(pos, Iterable)
                pos = np.array(pos)
                if isinstance(col, str):
                    if col in self.colour_dictionary.keys():
                        d2d.SetColour(self.colour_dictionary[col])
                    else:
                        raise
                    ValueError
                    ("An improper colour string was passed. '%s' is not a "
                     "valid colour in mlchem.importables."
                     "colour_dictionary palette." %
                     col)

                elif isinstance(col, tuple):      # if colour is str + alpha
                    if isinstance(col[0], str) and (isinstance(col[1], float)
                                                    or isinstance(
                                                        col[1], int)):
                        try:
                            initial_tuple = self.colour_dictionary[col[0]]
                            final_tuple = initial_tuple+(col[1],)

                            d2d.SetColour(final_tuple)
                        except Exception:
                            raise
                        ValueError
                        (
                         "An improper colour tuple was passed. Try pass "
                         "[(<colour_string>, <alpha_float_value>)] "
                         "as argument.")
                    else:
                        try:
                            d2d.SetColour(col)
                        except Exception:
                            raise
                        ValueError
                        ("An improper colour tuple was passed. Correct custom"
                         " palette has to be: (R,G,B)or (R,G,B,A)\nExample: "
                         "(0.7,0.0,0.7,0.5)) will set overwrite colour to "
                         "purple with 50% transparency.")

                shape_center = Point2D(0, 0)
                shape_center.x, shape_center.y, = pos[:2]
                if typ == 'circle':
                    d2d.DrawArc(shape_center, siz, 0, 359.9999999999)

        if ACS1996_mode is True:
            d2d = Draw.MolDraw2DCairo(-1, -1)
            Draw.DrawMoleculeACS1996(d2d, mol, legend=legend)
            d2d.FinishDrawing()
            self.DrawingText = d2d.GetDrawingText()
            return show_png(self.DrawingText)
        else:
            d2d.DrawMolecule(mol, legend=legend, highlightAtoms=highlightAtoms)
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

                # Create a base image for the molecule
                molecule_image = Image.open(
                    io.BytesIO(
                        d2d.GetDrawingText()
                        )
                        )

                # Draw concentric smooth gradient circles around each atom
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

            d2d.FinishDrawing()
            self.DrawingText = d2d.GetDrawingText()
            return show_png(self.DrawingText)
