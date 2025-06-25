from typing import Literal, Iterable, Optional
import numpy as np

# STATISTICS & MATH #


def shannon_entropy(vector: np.ndarray) -> float:
    """
Calculate Shannon entropy of a vector.

This function computes the entropy based on the frequency of unique
elements in the input array.

Parameters
----------
vector : np.ndarray
    Input array for which Shannon entropy is calculated.

Returns
-------
float
    Shannon entropy of the input vector.

Examples
--------
>>> shannon_entropy(np.array([1, 1, 2, 2, 3, 3]))
"""

    from scipy.stats import entropy

    values, counts = np.unique(vector, return_counts=True)
    probabilities = counts / len(vector)
    return entropy(probabilities, base=2)


def bernoulli(n: int, k: int, p: float) -> float:
    """
Calculate the Bernoulli probability of k successes in n trials.

This uses the binomial distribution formula.

Parameters
----------
n : int
    Number of trials.
k : int
    Number of successes.
p : float
    Probability of success on a single trial.

Returns
-------
float
    Bernoulli probability of k successes in n trials.

Examples
--------
>>> bernoulli(10, 3, 0.5)
"""

    import math

    q = 1 - p
    combinations = math.factorial(n) / (
        math.factorial(n - k) * math.factorial(k))
    probabilities = (p ** k) * (q ** (n - k))
    return combinations * probabilities


def logit_to_proba(logit: float) -> float:
    """
Convert a logit value to probability.

This applies the logistic (sigmoid) function.

Parameters
----------
logit : float
    Logit value to be converted.

Returns
-------
float
    Corresponding probability.

Examples
--------
>>> logit_to_proba(0)
0.5
"""

    return np.exp(logit) / (np.exp(logit) + 1)


def pairwise_euclidean_distance(matrix: np.ndarray) -> np.ndarray:
    """
Calculate the pairwise Euclidean distances between rows of a matrix.

Uses SciPy's `pdist` and `squareform` functions.

Parameters
----------
matrix : np.ndarray
    Input 2D array of shape (N, D) where N is the number of points.

Returns
-------
np.ndarray
    2D array of pairwise Euclidean distances.

Examples
--------
>>> pairwise_euclidean_distance(np.array([[0, 0], [1, 0], [0, 1]]))
"""

    from scipy.spatial.distance import pdist, squareform

    return squareform(pdist(matrix, 'euclidean'))


# CHEMISTRY & PHYSICS #


def calc_logD_HH(pH: float, logP: float, pKa: float,
                 behaviour: Literal['acid', 'base']) -> tuple:
    """
Calculate the distribution coefficient (logD) at a given pH using
the Henderson-Hasselbalch equation.

Parameters
----------
pH : float
    The pH at which to calculate the distribution coefficient.
logP : float
    The logarithm of the partition coefficient.
pKa : float
    The acid dissociation constant.
behaviour : {'acid', 'base'}
    The behaviour of the molecule.

Returns
-------
tuple
    A tuple containing:
    - Ion-neutral ratio (float)
    - Ionised percentage (float)
    - logD (float)

Examples
--------
>>> calc_logD_HH(7.4, 3.0, 4.5, 'acid')
"""


    # TODO: move from acid/base dichotomy and use
    # protonated/deprotonated forms instead
    import math

    A = 1 if behaviour == 'acid' else -1
    ion_neutral_ratio = 10 ** (A * (pH - pKa))
    ionised_percentage = 1 - (1 / (ion_neutral_ratio + 1))

    # Cap the ion_neutral_ratio to 5000 to avoid extreme values
    #ion_neutral_ratio = min(ion_neutral_ratio, 5000)

    logD = logP - math.log10(ion_neutral_ratio + 1)

    return ion_neutral_ratio, ionised_percentage, logD


def calc_centroid(coordinates: np.ndarray,
                  masses: Optional[Iterable] = None) -> np.ndarray:
    """
Calculate the centroid of a set of points, optionally weighted by masses.

Parameters
----------
coordinates : np.ndarray
    A 2D array of shape (N, D) where N is the number of points.
masses : Iterable, optional
    An iterable of length N representing the masses of each point.

Returns
-------
np.ndarray
    The coordinates of the centroid.

Examples
--------
>>> calc_centroid(np.array([[0, 0], [2, 0], [1, 2]]))
"""

    coordinates = np.array(coordinates)
    if masses is None:
        centroid = coordinates.mean(axis=0)
    else:
        masses = np.array(masses)
        total_mass = np.sum(masses)
        centroid = np.sum(coordinates.T * masses, axis=1) / total_mass
    return centroid


def calc_gyration_tensor(coordinates: np.ndarray,
                         masses: Optional[Iterable] = None) -> np.ndarray:
    """
Calculate the gyration tensor of a set of coordinates.

Parameters
----------
coordinates : np.ndarray
    A 2D array of shape (N, 3) representing spatial coordinates.
masses : Iterable, optional
    An iterable of length N representing the masses of each point.

Returns
-------
np.ndarray
    The 3x3 gyration tensor.

Examples
--------
>>> calc_gyration_tensor(np.random.rand(5, 3))
"""

    coordinates = np.array(coordinates)

    if masses is None:
        center = coordinates.mean(axis=0)
        normed_points = coordinates - center
        gyration_tensor = np.einsum('ij,ik->jk', normed_points, normed_points)
        gyration_tensor /= len(coordinates)

    else:
        masses = np.array(masses)
        total_mass = np.sum(masses)
        center_of_mass = np.sum(coordinates.T * masses, axis=1) / total_mass
        shifted_coordinates = coordinates - center_of_mass
        gyration_tensor = np.einsum('ij,i,ik->jk', shifted_coordinates,
                                    masses, shifted_coordinates)
        gyration_tensor /= total_mass

    return gyration_tensor


def calc_shape_descriptors_from_gyration_tensor(
        gyration_tensor: np.ndarray) -> dict:
    """
Calculate shape descriptors from a 3x3 gyration tensor.

Parameters
----------
gyration_tensor : np.ndarray
    A 3x3 gyration tensor.

Returns
-------
dict
    A dictionary containing:
    - 'moments_of_inertia'
    - 'principal_axes'
    - 'asphericity'
    - 'acylindricity'
    - 'relative_shape_anisotropy'

Examples
--------
>>> tensor = calc_gyration_tensor(np.random.rand(5, 3))
>>> calc_shape_descriptors_from_gyration_tensor(tensor)
"""

    if gyration_tensor.shape != (3, 3):
        raise ValueError("Gyration tensor must be a 3x3 matrix.")

    moments, principal_axes = np.linalg.eigh(gyration_tensor)
    total_moments = np.sum(moments)

    asphericity = ((moments[2] - 0.5 * (moments[0] + moments[1])) /
                   total_moments)
    acylindricity = (moments[1] - moments[0]) / total_moments
    relative_shape_anisotropy = (1.5 * (np.sum(moments ** 2) /
                                        (total_moments ** 2)) - 0.5)

    return {
        'moments_of_inertia': moments,
        'principal_axes': principal_axes,
        'asphericity': asphericity,
        'acylindricity': acylindricity,
        'relative_shape_anisotropy': relative_shape_anisotropy,
    }


def boltzmann_probability(energy_levels: Iterable[float],
                          temperature: int | float,
                          energy_unit: Literal[
                              'eV', 'J', 'cal', 'kJ',
                              'kcal', 'kJ/mol', 'kcal/mol',
                              ] = 'kcal/mol') -> list[float]:
    """
Calculate the Boltzmann probability for a set of energy levels
at a given temperature.

Parameters
----------
energy_levels : Iterable[float]
    A list or array of energy levels.
temperature : float or int
    Temperature in Kelvin.
energy_unit : str, optional
    Unit of energy levels. Default is 'kcal/mol'.
    Supported units: 'eV', 'J', 'cal', 'kJ', 'kcal', 'kJ/mol', 'kcal/mol'.

Returns
-------
list[float]
    Boltzmann probabilities for the given energy levels.

Examples
--------
>>> boltzmann_probability([0, 1, 2], 298, 'kcal/mol')
"""

    import math

    if temperature == 0:
        raise ValueError("Temperature must be greater than zero.")

    # Boltzmann constant in different units
    k_B_values = {
        'eV': 8.617333262145e-5,     # eV/K
        'J': 1.380649e-23,     # J/K
        'cal': 3.2976230e-24,     # cal/K
        'kJ': 1.380649e-26,     # kJ/K
        'kcal': 3.2976230e-27,     # kcal/K
        'kJ/mol': 8.31446261815324e-3,     # kJ/(mol*K)
        'kcal/mol': 1.98720425864083e-3     # kcal/(mol*K)
    }

    # Ensure the provided energy unit is supported
    if energy_unit not in k_B_values:
        raise ValueError(
            f"Unsupported energy unit: {energy_unit}. "
            f"Supported units are: {', '.join(k_B_values.keys())}")

    k_B = k_B_values[energy_unit]

    # Calculate the partition function
    partition_function = sum(
        math.exp(-E / (k_B * temperature))
        for E in energy_levels
        )

    if partition_function == 0:
        raise
    ValueError(
        "Partition function is zero, check the energy levels and temperature.")

    # Calculate the probabilities
    probabilities = [
        math.exp(
            -E / (k_B * temperature)) / partition_function
        for E in energy_levels
        ]

    return probabilities
