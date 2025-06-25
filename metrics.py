from typing import Iterable
from rdkit.Chem import DataStructs


def get_sensitivity(
    y_true: Iterable[int | str],
    y_pred: Iterable[int | str],
    labels: Iterable[int | str] | None = None,
) -> float:
    """
    Compute the sensitivity (recall) of a prediction.

    Parameters
    ----------
    y_true : Iterable[int or str]
        True labels.

    y_pred : Iterable[int or str]
        Predicted labels.

    labels : Iterable[int or str], optional
        Label names to include in the calculation. If None, all labels 
        are used.

    Returns
    -------
    float
        The sensitivity (recall) of the prediction.
    """
    from sklearn.metrics import confusion_matrix

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    if len(labels) < 2:
        labels = list(labels) + ['__dummy__']

    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    if confmat.shape[0] < 2 or confmat.shape[1] < 2:
        return 0.0

    tp = confmat[1][1]
    fn = confmat[1][0]
    denom = tp + fn
    return tp / denom if denom != 0 else 0.0



def get_specificity(
    y_true: Iterable[int | str],
    y_pred: Iterable[int | str],
    labels: Iterable[int | str] | None = None,
) -> float:
    """
    Compute the specificity of a prediction.

    Parameters
    ----------
    y_true : Iterable[int or str]
        True labels.

    y_pred : Iterable[int or str]
        Predicted labels.

    labels : Iterable[int or str], optional
        Label names to include in the calculation. If None, all labels are used.

    Returns
    -------
    float
        The specificity of the prediction.
    """
    from sklearn.metrics import confusion_matrix

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    if len(labels) < 2:
        labels = list(labels) + ['__dummy__']

    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    if confmat.shape[0] < 2 or confmat.shape[1] < 2:
        return 0.0

    tn = confmat[0][0]
    fp = confmat[0][1]
    denom = tn + fp
    return tn / denom if denom != 0 else 0.0



def get_geometric_S(
    y_true: Iterable[int | str],
    y_pred: Iterable[int | str],
    labels: Iterable[int | str] | None = None,
) -> float:
    """
    Compute the geometric mean of sensitivity and specificity.

    Parameters
    ----------
    y_true : Iterable[int or str]
        True labels.

    y_pred : Iterable[int or str]
        Predicted labels.

    labels : Iterable[int or str], optional
        Label names to include in the calculation. If None, all labels are used.

    Returns
    -------
    float
        The geometric mean of sensitivity and specificity.
"""

    sensitivity = get_sensitivity(y_true, y_pred, labels=labels)
    specificity = get_specificity(y_true, y_pred, labels=labels)
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

    import numpy as np

    rmse = get_rmse(y_true, y_pred)
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
