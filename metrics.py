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
from rdkit.Chem import DataStructs
import numpy as np
import warnings


def get_sensitivity(
    y_true: Iterable[int | str],
    y_pred: Iterable[int | str],
) -> float:
    """
    Compute the sensitivity (recall) of a prediction.
    The `labels` argument has been removed in v1.1.3 as ininfluent.

    Parameters
    ----------
    y_true : Iterable[int or str]
        True labels.

    y_pred : Iterable[int or str]
        Predicted labels.

    Returns
    -------
    float
        The sensitivity (recall) of the prediction.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denominator = tp + fn
    if denominator == 0:
        warnings.warn("Denominator is zero, sensitivity is undefined but it will be coerced to 0.", UserWarning)
    return tp / denominator if denominator > 0 else 0



def get_specificity(
    y_true: Iterable[int | str],
    y_pred: Iterable[int | str],
) -> float:
    """
    Compute the specificity of a prediction.
    The `labels` argument has been removed in v1.1.3 as ininfluent.

    Parameters
    ----------
    y_true : Iterable[int or str]
        True labels.

    y_pred : Iterable[int or str]
        Predicted labels.


    Returns
    -------
    float
        The specificity of the prediction.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    denominator = tn + fp
    if denominator == 0:
        warnings.warn("Denominator is zero, specificity is undefined but it will be coerced to 0.", UserWarning)
    return tn / denominator if denominator > 0 else 0



def get_geometric_S(
    y_true: Iterable[int | str],
    y_pred: Iterable[int | str],
) -> float:
    """
    Compute the geometric mean of sensitivity and specificity.
    The `labels` argument has been removed in v1.1.3 as ininfluent.

    Parameters
    ----------
    y_true : Iterable[int or str]
        True labels.

    y_pred : Iterable[int or str]
        Predicted labels.

    Returns
    -------
    float
        The geometric mean of sensitivity and specificity.
"""


    unique_true = np.unique(y_true)

    # Degenerate cases: only one class present in y_true
    if len(unique_true) == 1:
        warnings.warn("Only one class is present in y_true", UserWarning)
        return float(np.array_equal(y_true, y_pred))

    sensitivity = get_sensitivity(y_true, y_pred)
    specificity = get_specificity(y_true, y_pred)

    return (sensitivity * specificity) ** 0.5

def get_mcc(
    y_true: Iterable[int | str],
    y_pred: Iterable[int | str]
) -> float:
    """
    Compute the Matthews Correlation Coefficient (MCC).

    Parameters
    ----------
    y_true : Iterable[int or str]
        True labels.

    y_pred : Iterable[int or str]
        Predicted labels.

    Returns
    -------
    float
        The Matthews Correlation Coefficient.
"""

    from sklearn.metrics import matthews_corrcoef

    return matthews_corrcoef(y_true, y_pred)


def get_rmse(
    y_true: Iterable[float | int],
    y_pred: Iterable[float | int]
) -> float:
    """
    Compute the root mean squared error (RMSE) of a prediction.

    Parameters
    ----------
    y_true : Iterable[float or int]
        True values.

    y_pred : Iterable[float or int]
        Predicted values.

    Returns
    -------
    float
        The root mean squared error.
"""

    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y_true, y_pred)
    rmse = mse**0.5
    return rmse


def calculate_reliability_components(
    train_score: float,
    cv_score: float,
    test_score: float,
    logic: Literal['lower', 'greater'],
) -> dict[str, float]:
    """
    Calculate reliability-score components from train, CV, and test scores.

    Parameters
    ----------
    train_score : float
        Score obtained on the training set.
    cv_score : float
        Cross-validation score.
    test_score : float
        Score obtained on the test set.
    logic : {'lower', 'greater'}
        Whether lower or greater metric values are better.

    Returns
    -------
    dict
        Dictionary containing ``geometric_mean``, ``performance_score``,
        ``instability_score``, and ``reliability_score``.
    """

    if logic not in ('lower', 'greater'):
        raise ValueError("'logic' must be either 'lower' or 'greater'.")

    geometric_mean = (train_score * cv_score * test_score) ** (1/3)
    performance_score = geometric_mean
    if logic == 'lower':
        performance_score = np.inf if geometric_mean == 0 else 1 / geometric_mean

    instability_score = (
        abs(train_score - cv_score) +
        abs(train_score - test_score) +
        abs(cv_score - test_score)
    )
    reliability_score = performance_score / (1 + instability_score)

    return {
        'geometric_mean': geometric_mean,
        'performance_score': performance_score,
        'instability_score': instability_score,
        'reliability_score': reliability_score,
    }


def rmse_to_std_ratio(
    y_true: Iterable[float | int],
    y_pred: Iterable[float | int]
) -> float:
    """
    Compute the ratio of the standard deviation of true values to RMSE.

    Parameters
    ----------
    y_true : Iterable[float or int]
        True values.

    y_pred : Iterable[float or int]
        Predicted values.

    Returns
    -------
    float
        The ratio of standard deviation to RMSE.
"""


    rmse = get_rmse(y_true, y_pred)
    if rmse == 0:
        return np.inf
    std = np.std(y_true)
    return std / rmse


def get_r2(
    y_true: Iterable[float | int],
    y_pred: Iterable[float | int]
) -> float:
    """
    Compute the R-squared value using Pearson's correlation coefficient.

    Parameters
    ----------
    y_true : Iterable[float or int]
        True values.

    y_pred : Iterable[float or int]
        Predicted values.

    Returns
    -------
    float
        The R-squared value.
    """

    from scipy.stats import pearsonr

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        return np.nan

    if np.std(y_true_arr) == 0 or np.std(y_pred_arr) == 0:
        return np.nan

    return pearsonr(y_true, y_pred)[0] ** 2


def DiceSimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect'
) -> float:
    """
    Compute the Dice similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.DiceSimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first molecular fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second molecular fingerprint.

    Returns
    -------
    float
        The Dice similarity coefficient, ranging from 0 (no similarity) to 1 (identical).
    """


    from rdkit import DataStructs
    return DataStructs.DiceSimilarity(fp1, fp2)


def OnBitSimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect'
) -> float:
    """
    Compute the OnBit similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.OnBitSimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first molecular fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second molecular fingerprint.

    Returns
    -------
    float
        The OnBit similarity, based on the number of bits set in both fingerprints.
    """

    from rdkit import DataStructs
    return DataStructs.OnBitSimilarity(fp1, fp2)


def SokalSimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect'
) -> float:
    """
    Compute the Sokal similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.SokalSimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first molecular fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second molecular fingerprint.

    Returns
    -------
    float
        The Sokal similarity coefficient, a normalized measure of bit overlap.
    """

    from rdkit import DataStructs
    return DataStructs.SokalSimilarity(fp1, fp2)


def AllBitSimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect'
) -> float:
    """
    Compute the AllBit similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.AllBitSimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first molecular fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second molecular fingerprint.

    Returns
    -------
    float
        The AllBit similarity, considering both on and off bits in the fingerprints.
    """

    from rdkit import DataStructs
    return DataStructs.AllBitSimilarity(fp1, fp2)


def CosineSimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect'
) -> float:
    """
    Compute the Cosine similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.CosineSimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first molecular fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second molecular fingerprint.

    Returns
    -------
    float
        The Cosine similarity, measuring the cosine of the angle between two bit vectors.
    """

    from rdkit import DataStructs
    return DataStructs.CosineSimilarity(fp1, fp2)


def RusselSimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect'
) -> float:
    """
    Compute the Cosine similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.CosineSimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first molecular fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second molecular fingerprint.

    Returns
    -------
    float
        The Cosine similarity, measuring the cosine of the angle between two bit vectors.
    """

    from rdkit import DataStructs
    return DataStructs.RusselSimilarity(fp1, fp2)


def TverskySimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect',
    a: float = 0.5,
    b: float = 0.5
) -> float:
    """
    Compute the Tversky similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.TverskySimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second fingerprint.

    a : float, optional
        Weight for features in `fp1`. Default is 0.5.

    b : float, optional
        Weight for features in `fp2`. Default is 0.5.

    Returns
    -------
    float
        The Tversky similarity between the two fingerprints.
"""

    from rdkit import DataStructs
    return DataStructs.TverskySimilarity(fp1, fp2, a, b)


def TanimotoSimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect'
) -> float:
    """
    Compute the Tanimoto similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.TanimotoSimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first molecular fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second molecular fingerprint.

    Returns
    -------
    float
        The Tanimoto similarity coefficient, commonly used for chemical structure comparison.
    """

    from rdkit import DataStructs
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def AsymmetricSimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect'
) -> float:
    """
    Compute the Asymmetric similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.AsymmetricSimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first molecular fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second molecular fingerprint.

    Returns
    -------
    float
        The Asymmetric similarity, emphasizing features present in the first fingerprint.
    """

    from rdkit import DataStructs
    return DataStructs.AsymmetricSimilarity(fp1, fp2)


def KulczynskiSimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect'
) -> float:
    """
    Compute the Kulczynski similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.KulczynskiSimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first molecular fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second molecular fingerprint.

    Returns
    -------
    float
        The Kulczynski similarity, a symmetric measure of bit overlap.
    """

    from rdkit import DataStructs
    return DataStructs.KulczynskiSimilarity(fp1, fp2)


def OffBitProjSimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect'
) -> float:
    """
    Compute the OffBitProj similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.OffBitProjSimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first molecular fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second molecular fingerprint.

    Returns
    -------
    float
        The OffBitProj similarity, based on the projection of off bits between fingerprints.
    """

    from rdkit import DataStructs
    return DataStructs.OffBitProjSimilarity(fp1, fp2)


def McConnaugheySimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect'
) -> float:
    """
    Compute the McConnaughey similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.McConnaugheySimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first molecular fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second molecular fingerprint.

    Returns
    -------
    float
        The McConnaughey similarity, a measure of structural similarity based on bit patterns.
    """

    from rdkit import DataStructs
    return DataStructs.McConnaugheySimilarity(fp1, fp2)


def BraunBlanquetSimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect'
) -> float:
    """
    Compute the Braun-Blanquet similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.BraunBlanquetSimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first molecular fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second molecular fingerprint.

    Returns
    -------
    float
        The Braun-Blanquet similarity, calculated as the intersection over the maximum bit count.
    """

    from rdkit import DataStructs
    return DataStructs.BraunBlanquetSimilarity(fp1, fp2)


def RogotGoldbergSimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect'
) -> float:
    """
    Compute the Rogot-Goldberg similarity between two fingerprints.

    This function is a shortcut for the RDKit method `DataStructs.RogotGoldbergSimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first molecular fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second molecular fingerprint.

    Returns
    -------
    float
        The Rogot-Goldberg similarity, a weighted measure of bit agreement.
    """

    from rdkit import DataStructs
    return DataStructs.RogotGoldbergSimilarity(fp1, fp2)


def FingerprintSimilarity(
    fp1: 'DataStructs.cDataStructs.ExplicitBitVect',
    fp2: 'DataStructs.cDataStructs.ExplicitBitVect',
    metric: callable
) -> float:
    """
    Compute the fingerprint similarity using a specified similarity metric.

    This function is a shortcut for the RDKit method `DataStructs.FingerprintSimilarity`.

    Parameters
    ----------
    fp1 : DataStructs.cDataStructs.ExplicitBitVect
        The first fingerprint.

    fp2 : DataStructs.cDataStructs.ExplicitBitVect
        The second fingerprint.

    metric : callable
        The similarity metric function to use.

    Returns
    -------
    float
        The fingerprint similarity between the two fingerprints.
    """

    from rdkit import DataStructs
    return DataStructs.FingerprintSimilarity(fp1, fp2, metric=metric)
