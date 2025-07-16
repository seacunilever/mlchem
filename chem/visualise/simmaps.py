from typing import Literal, Iterable
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
from matplotlib.figure import Figure


class SimMaps:

	@staticmethod
	def get_weights_from_model(
		mol_input: Chem.rdchem.Mol | str, estimator,
		estimator_cols: Iterable,
		model_type: Literal['regression', 'classification'],
		actual_val: float,
		fp_type: Literal['m', 'ap', 'tt', 'rk'],
		normalise: bool = False,
		return_df: bool = False,
		**kwargs) -> Iterable | pd.DataFrame:
		"""
Get atomic importance weights from a predictive model using masked fingerprints.

This method calculates atomic contributions by iteratively masking each atom
in the molecule and evaluating the change in model prediction.

Parameters
----------
mol_input : rdkit.Chem.rdchem.Mol or str
    The input molecule as an RDKit Mol object or a SMILES string.
estimator : sklearn.base.BaseEstimator
    A trained scikit-learn estimator.
estimator_cols : Iterable
    Feature names used by the estimator.
model_type : {'regression', 'classification'}
    Type of model. Must be either 'regression' or 'classification'.
actual_val : float
    The actual value predicted by the model. For classification, this is the
    probability of class 1; for regression, the continuous target value.
fp_type : {'m', 'ap', 'tt', 'rk'}
    Type of fingerprint to use:
    - 'm': Morgan
    - 'ap': Atom Pair
    - 'tt': Topological Torsion
    - 'rk': RDKit
normalise : bool, optional
    If True, normalises the weights to the range (-1, 1). Default is False.
return_df : bool, optional
    If True, returns a pandas DataFrame. Otherwise, returns a NumPy array.
**kwargs : dict
    Additional fingerprint-specific parameters. See below.

Fingerprint Parameters
----------------------
Morgan ('m'):
- radius : int, default=2
    Radius of the Morgan fingerprint.
- fpType : {'count', 'bv'}, default='bv'
    Type of fingerprint: bit vector ('bv') or count-based.
- atomId : int, default=-1
    Atom to mask. -1 means no masking.
- nBits : int, default=2048
    Size of the bit vector.
- useFeatures : bool, default=False
    If True, uses FeatureMorgan; otherwise, ConnectivityMorgan.

Atom Pair ('ap'):
- fpType : {'normal', 'hashed', 'bv'}, default='normal'
- atomId : int, default=-1
- nBits : int, default=2048
- minLength : int, default=1
- maxLength : int, default=30
- nBitsPerEntry : int, default=4

Topological Torsion ('tt'):
- fpType : {'normal', 'hashed', 'bv'}, default='normal'
- atomId : int, default=-1
- nBits : int, default=2048
- targetSize : int, default=4
- nBitsPerEntry : int, default=4

RDKit ('rk'):
- fpType : {'', 'bv'}, default='bv'
- atomId : int, default=-1
- nBits : int, default=2048
- minPath : int, default=1
- maxPath : int, default=5
- nBitsPerHash : int, default=2

Returns
-------
Iterable or pandas.DataFrame
    Atomic contributions to model prediction. Format depends on `return_df`.

Examples
--------
>>> SimMaps.get_weights_from_model("CCO", model, feature_names, "regression", 0.85, "m")
"""


		from rdkit.Chem.Draw import SimilarityMaps
		from mlchem.helper import normalise_iterable
		from mlchem.helper import create_progressive_column_names

		# Dictionary to map fp_type to the corresponding function
		fp_function_dict = {
			'm': SimilarityMaps.GetMorganFingerprint,
			'ap': SimilarityMaps.GetAPFingerprint,
			'tt': SimilarityMaps.GetTTFingerprint,
			'rk': SimilarityMaps.GetRDKFingerprint
		}

		# Get the appropriate fingerprint function
		fp_function = fp_function_dict[fp_type]
		results = []
		for atomId, atom in enumerate(mol_input.GetAtoms()):
			probe_fp = fp_function(
				mol=mol_input,
				atomId=atomId,
				**kwargs).ToList()
			fp_names = create_progressive_column_names(
				fp_type, len(probe_fp))
			fp_dataframe = pd.DataFrame(probe_fp, index=fp_names).T
			if model_type == 'classification':
				predicted_val = estimator.predict_proba(
					fp_dataframe[estimator_cols].values)[0][1]
			else:
				predicted_val = estimator.predict(
					fp_dataframe[estimator_cols].values)[0]
			delta = actual_val - predicted_val
			results.append((atom.GetSymbol(), atomId, predicted_val, delta))
		df = pd.DataFrame(
			results, columns=[
				'Atom_Symbol', 'Atom_Index', 'Predicted_Proba', 'Delta',
				])
		if normalise is True:
			df['Delta'] = normalise_iterable(df['Delta'].values)
		return df if return_df else df.Delta.values


	@staticmethod
	def get_weights_from_fingerprint(
		refmol: Chem.rdchem.Mol, probemol: Chem.rdchem.Mol,
		fp_type: Literal['m', 'ap', 'rk', 'tt'] = 'm',
		similarity_metric: Literal[
			'Tanimoto', 'Dice', 'Cosine',
			'Sokal', 'Russel', 'RogotGoldberg',
			'AllBit', 'OnBit', 'Kulczynski',
			'McConnaughey', 'Asymmetric',
			'BraunBlanquet', 'Tversky'
			] = 'Tanimoto',
		normalise: bool = False,
		return_df: bool = False,
		**kwargs
			) -> Iterable | pd.DataFrame:
		"""
Get atomic importance weights based on fingerprint similarity.

This method calculates atomic contributions by masking atoms in the probe
molecule and computing the change in similarity to a reference molecule.

Parameters
----------
refmol : rdkit.Chem.rdchem.Mol
    The reference molecule.
probemol : rdkit.Chem.rdchem.Mol
    The probe molecule whose atoms will be masked.
fp_type : {'m', 'ap', 'rk', 'tt'}, default='m'
    Type of fingerprint to use.
similarity_metric : str, default='Tanimoto'
    Similarity metric to use. Options include:
    'Tanimoto', 'Dice', 'Cosine', 'Sokal', 'Russel', 'RogotGoldberg',
    'AllBit', 'OnBit', 'Kulczynski', 'McConnaughey', 'Asymmetric',
    'BraunBlanquet', 'Tversky'.
normalise : bool, optional
    If True, normalises the weights to the range (-1, 1). Default is False.
return_df : bool, optional
    If True, returns a pandas DataFrame. Otherwise, returns a NumPy array.
**kwargs : dict
    Additional fingerprint-specific parameters. See below.

Fingerprint Parameters
----------------------
Morgan ('m'):
- radius : int, default=2
- fpType : {'count', 'bv'}, default='bv'
- atomId : int, default=-1
- nBits : int, default=2048
- useFeatures : bool, default=False

Atom Pair ('ap'):
- fpType : {'normal', 'hashed', 'bv'}, default='normal'
- atomId : int, default=-1
- nBits : int, default=2048
- minLength : int, default=1
- maxLength : int, default=30
- nBitsPerEntry : int, default=4
Topological Torsion ('tt'):
- fpType : {'normal', 'hashed', 'bv'}, default='normal'
- atomId : int, default=-1
- nBits : int, default=2048
- targetSize : int, default=4
- nBitsPerEntry : int, default=4

RDKit ('rk'):
- fpType : {'', 'bv'}, default='bv'
- atomId : int, default=-1
- nBits : int, default=2048
- minPath : int, default=1
- maxPath : int, default=5
- nBitsPerHash : int, default=2

Returns
-------
Iterable or pandas.DataFrame
    Atomic contributions to similarity. Format depends on `return_df`.

Examples
--------
>>> SimMaps.get_weights_from_fingerprint(refmol, probemol, fp_type='m')
"""


		from mlchem.importables import similarity_metric_dictionary
		from rdkit.Chem.Draw import SimilarityMaps
		from mlchem.helper import normalise_iterable
		import pandas as pd

		# Dictionary to map fp_type to the corresponding function
		fp_function_dict = {
			'm': SimilarityMaps.GetMorganFingerprint,
			'ap': SimilarityMaps.GetAPFingerprint,
			'tt': SimilarityMaps.GetTTFingerprint,
			'rk': SimilarityMaps.GetRDKFingerprint
		}

		# Get the appropriate fingerprint function
		fp_function = fp_function_dict[fp_type]

		# Get the similarity metric function
		metric = similarity_metric_dictionary[similarity_metric]

		# Calculate the base fingerprints and similarity
		base_ref_fp = fp_function(refmol, atomId=-1, **kwargs)
		base_probe_fp = fp_function(probemol, atomId=-1, **kwargs)
		base_similarity = metric(base_ref_fp, base_probe_fp)

		results = []

		# Iterate over atoms in the probe molecule
		for atomId, atom in enumerate(probemol.GetAtoms()):
			# Calculate the fingerprint with the current atom masked
			probe_fp = fp_function(mol=probemol, atomId=atomId, **kwargs)
			# Calculate the difference in similarity
			delta = base_similarity - metric(base_ref_fp, probe_fp)
			results.append((atom.GetSymbol(), atomId, base_similarity, delta))

		# Create a DataFrame from the results
		df = pd.DataFrame(results, columns=[
			'Atom_Symbol', 'Atom_Index', 'Predicted_Proba', 'Delta'])

		# Normalise the delta values if required
		if normalise:
			df['Delta'] = normalise_iterable(df['Delta'].values)

		# Return the results as a DataFrame or an iterable
		return df if return_df else df.Delta.values


	@staticmethod
	def get_similarity_map_from_weights(
		mol: Chem.rdchem.Mol, weights: Iterable, colorMap=None,
		scale: int =-1, size: tuple = (250, 250), sigma=None,
		coordScale: int | float = 1.5, step: float = 0.01,
		contour_colour: str | tuple = 'black',
		contourLines: int = 10, alpha: float = 0.5,
		contour_width: int | float = 1, resolution: float = 0.05,
		dash_negative : bool = True, draw2d = None, **kwargs
	) -> Draw.rdMolDraw2D.MolDraw2D | Figure:
		"""
Generate a similarity map visualisation from atomic weights.

This method overlays a similarity map on a molecule using atomic weights,
with optional contour lines and colour maps.

Adapted from RDKit (https://www.rdkit.org), originally licensed under the BSD 3-Clause License.

Parameters
----------
mol : rdkit.Chem.rdchem.Mol
    The molecule to visualise.
weights : Iterable
    Atomic weights to visualise.
colorMap : str or list or matplotlib colormap, optional
    Colour map to use. If None, a custom PiWG colour map is used.
scale : int, default=-1
    Scaling factor. If negative, uses the maximum absolute weight.
size : tuple, default=(250, 250)
    Size of the output image.
sigma : float, optional
    Gaussian width. If None, estimated from bond lengths.
coordScale : float, default=1.5
    Scaling factor for coordinates.
step : float, default=0.01
    Step size for Gaussian calculation.
contour_colour : str or tuple, default='black'
    Colour of contour lines. Can be a string or RGB tuple.
contourLines : int or list, default=10
    Number of contour lines or specific contour levels.
alpha : float, default=0.5
    Transparency of contour lines.
contour_width : float, default=1
    Width of contour lines.
resolution : float, default=0.05
    Grid resolution for contour plotting.
dash_negative : bool, default=True
    Whether to use dashed lines for negative weights.
draw2d : rdkit.Chem.Draw.rdMolDraw2D.MolDraw2D, optional
    RDKit drawing object. Required for rendering.
**kwargs : dict
    Additional keyword arguments passed to matplotlib drawing.

Returns
-------
rdMolDraw2D.MolDraw2D or matplotlib.figure.Figure
    If `draw2d` is provided, returns the modified drawing object.
    Otherwise, returns a matplotlib figure.

Raises
------
ValueError
    If `draw2d` is not provided or the molecule has fewer than 2 atoms.

Examples
--------
>>> SimMaps.get_similarity_map_from_weights(mol, weights, draw2d=drawer)
"""


		import numpy as np
		import math
		from rdkit import Geometry
		from rdkit.Chem import Draw
		try:
			from matplotlib import cm
			from matplotlib import colormaps
			from matplotlib.colors import LinearSegmentedColormap
		except ImportError:
			cm = None
		except RuntimeError:
			cm = None

		if isinstance(contour_colour, str):
			# Convert string color to a tuple (e.g., 'black' to (0, 0, 0))
			import matplotlib.colors as mcolors
			contour_colour_tuple = mcolors.to_rgb(contour_colour)
		else:
			contour_colour_tuple = contour_colour

		if mol.GetNumAtoms() < 2:
			raise ValueError("too few atoms")

		if draw2d is not None:
			if sigma is None:
				if mol.GetNumBonds() > 0:
					bond = mol.GetBondWithIdx(0)
					idx1 = bond.GetBeginAtomIdx()
					idx2 = bond.GetEndAtomIdx()
					sigma = 0.3 * (
						mol.GetConformer().GetAtomPosition(idx1) -
						mol.GetConformer().GetAtomPosition(idx2)
						).Length()
				else:
					sigma = 0.3 * (
						mol.GetConformer().GetAtomPosition(0) -
						mol.GetConformer().GetAtomPosition(1)
						).Length()
			sigma = round(sigma, 2)

			sigmas = [sigma] * mol.GetNumAtoms()
			locs = [Geometry.Point2D(mol.GetConformer().GetAtomPosition(i).x,
								mol.GetConformer().GetAtomPosition(i).y) for i in
								range(mol.GetNumAtoms())]
			draw2d.ClearDrawing()

			ps = Draw.ContourParams()
			ps.fillGrid = True
			ps.gridResolution = resolution
			ps.contourWidth = contour_width
			ps.dashNegative = dash_negative

			if colorMap is not None:
				if cm is not None and isinstance(colorMap, type(cm.Blues)):
					clrs = [tuple(x) for x in colorMap([0, 0.5, 1])]
				elif isinstance(colorMap, str):
					if cm is None:
						raise
					ValueError(
						"cannot provide named colormaps unless "
						"matplotlib is installed"
						)
					# cm.get_cmap() deprecated with matplotlib 3.11
					#clrs = [tuple(x) for x in cm.get_cmap(colorMap)([0, 0.5, 1])]
					clrs = [tuple(x) for x in colormaps.get_cmap(colorMap)([0, 0.5, 1])]
				else:
					clrs = [colorMap[0], colorMap[1], colorMap[2]]
				ps.setColourMap(clrs)

			ps.setContourColour(contour_colour_tuple)
			Draw.ContourAndDrawGaussians(
				draw2d, locs, weights, sigmas, nContours=contourLines, params=ps
				)
			draw2d.drawOptions().clearBackground = False
			draw2d.DrawMolecule(mol)
			return draw2d
		raise ValueError("the draw2d argument must be provided")
