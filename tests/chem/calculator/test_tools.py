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