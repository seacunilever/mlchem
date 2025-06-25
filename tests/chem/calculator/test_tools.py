import pytest
import numpy as np
from mlchem.chem.calculator.tools import (
    shannon_entropy,
    bernoulli,
    logit_to_proba,
    pairwise_euclidean_distance,
    calc_logD_HH,
    calc_centroid,
    calc_gyration_tensor,
    calc_shape_descriptors_from_gyration_tensor,
    boltzmann_probability
)


def test_shannon_entropy():
    vector = np.array([1, 1, 2, 2, 3, 3])
    result = shannon_entropy(vector)
    assert isinstance(result, float)
    assert result > 0


def test_bernoulli():
    n, k, p = 10, 3, 0.5
    result = bernoulli(n, k, p)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_logit_to_proba():
    logit = 0.5
    result = logit_to_proba(logit)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_pairwise_euclidean_distance():
    matrix = np.array([[0, 0], [3, 4], [6, 8]])
    result = pairwise_euclidean_distance(matrix)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    assert np.allclose(result[0, 1], 5.0)


def test_calc_logD_HH_acid():
    pH, logP, pKa = 7.4, 1.0, 4.75
    behaviour = 'acid'
    result = calc_logD_HH(pH, logP, pKa, behaviour)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[1] < logP


def test_calc_logD_HH_base():
    pH, logP, pKa = 7.4, 1.0, 9.25
    behaviour = 'base'
    result = calc_logD_HH(pH, logP, pKa, behaviour)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[1] < logP


def test_calc_centroid():
    coordinates = np.array([[0, 0], [1, 1], [2, 2]])
    result = calc_centroid(coordinates)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert np.allclose(result, [1, 1])


def test_calc_centroid_with_masses():
    coordinates = np.array([[0, 0], [1, 1], [2, 2]])
    masses = [1, 2, 3]
    result = calc_centroid(coordinates, masses)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert np.allclose(result, [1.33333333, 1.33333333])


def test_calc_gyration_tensor():
    coordinates = np.array([[0, 0], [1, 1], [2, 2]])
    result = calc_gyration_tensor(coordinates)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


def test_calc_gyration_tensor_with_masses():
    coordinates = np.array([[0, 0], [1, 1], [2, 2]])
    masses = [1, 2, 3]
    result = calc_gyration_tensor(coordinates, masses)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


def test_calc_shape_descriptors_from_gyration_tensor():
    gyration_tensor = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = calc_shape_descriptors_from_gyration_tensor(gyration_tensor)
    assert isinstance(result, dict)
    assert 'moments_of_inertia' in result
    assert 'principal_axes' in result


def test_boltzmann_probability():
    energy_levels = [0, 1, 2]
    temperature = 300
    result = boltzmann_probability(energy_levels, temperature)
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(p, float) for p in result)
    assert np.isclose(sum(result), 1)


def test_boltzmann_probability_invalid_temperature():
    energy_levels = [0, 1, 2]
    temperature = 0
    with pytest.raises(ValueError):
        boltzmann_probability(energy_levels, temperature)


if __name__ == "__main__":
    pytest.main()