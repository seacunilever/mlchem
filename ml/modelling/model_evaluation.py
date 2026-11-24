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

from typing import Literal, Callable, Iterable, Sequence
from dataclasses import dataclass
from uuid import uuid4
import pandas as pd
import numpy as np
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from mlchem.helper import (
    coerce_log_level,
    validate_task_type,
    resolve_n_jobs,
    disable_estimator_parallelization,
    validate_dataframe_columns,
)


logger = logging.getLogger(__name__)


def _configure_module_logging(level: int) -> None:
    """Configure module logger to emit at the specified level."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(level)


def _n_samples(X: np.ndarray | pd.DataFrame) -> int:
    return len(X)


def validate_cv_indices(
    cv_indices: Iterable,
    n_samples: int,
    strict_coverage: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Validate an explicit list of (train_idx, valid_idx) cross-validation folds.

    Parameters
    ----------
    cv_indices : iterable of (array-like, array-like)
        Explicit train/validation index pairs, one per fold.

    n_samples : int
        Number of samples the indices are expected to reference.

    strict_coverage : bool, optional (default=False)
        If True, raise a ``ValueError`` when some samples are never used
        as validation data in any fold. If False (default), only emit a
        ``RuntimeWarning`` in that case, since some valid fold manifests
        (e.g. a ``PredefinedSplit`` excluding held-out samples) may
        legitimately not cover every sample.

    Returns
    -------
    list of (numpy.ndarray, numpy.ndarray)
        Normalised, validated (train_idx, valid_idx) pairs.

    Raises
    ------
    ValueError
        If folds overlap, contain out-of-range/duplicate indices, are
        empty, or are otherwise inconsistently defined.

    Examples
    --------
    Validate two disjoint folds whose validation sets cover all samples:

    >>> import numpy as np
    >>> folds = [
    ...     (np.array([2, 3]), np.array([0, 1])),
    ...     (np.array([0, 1]), np.array([2, 3])),
    ... ]
    >>> checked = validate_cv_indices(folds, n_samples=4)
    >>> len(checked)
    2
    """

    if cv_indices is None:
        raise ValueError("'cv_indices' must not be None.")

    try:
        fold_list = list(cv_indices)
    except TypeError as exc:
        raise ValueError(
            "'cv_indices' must be an iterable of (train_idx, valid_idx) pairs."
        ) from exc

    if len(fold_list) == 0:
        raise ValueError("'cv_indices' must contain at least one (train_idx, valid_idx) fold.")

    normalised = []
    covered_as_valid = set()
    for fold_number, fold in enumerate(fold_list, start=1):
        try:
            train_idx, valid_idx = fold
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Fold {fold_number} in 'cv_indices' must be a (train_idx, valid_idx) pair."
            ) from exc

        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)

        if train_idx.ndim != 1 or valid_idx.ndim != 1:
            raise ValueError(f"Fold {fold_number}: train/valid indices must be 1-D arrays.")

        if train_idx.size == 0:
            raise ValueError(f"Fold {fold_number}: training indices are empty.")
        if valid_idx.size == 0:
            raise ValueError(f"Fold {fold_number}: validation indices are empty.")

        combined = np.concatenate([train_idx, valid_idx])
        if combined.min() < 0 or combined.max() >= n_samples:
            raise ValueError(
                f"Fold {fold_number}: indices out of range for {n_samples} samples "
                f"(expected values in [0, {n_samples - 1}])."
            )

        if len(np.unique(train_idx)) != len(train_idx):
            raise ValueError(f"Fold {fold_number}: training indices contain duplicates.")
        if len(np.unique(valid_idx)) != len(valid_idx):
            raise ValueError(f"Fold {fold_number}: validation indices contain duplicates.")

        overlap = np.intersect1d(train_idx, valid_idx)
        if overlap.size > 0:
            raise ValueError(
                f"Fold {fold_number}: train and validation indices overlap "
                f"({overlap.size} shared sample(s))."
            )

        covered_as_valid.update(valid_idx.tolist())
        normalised.append((train_idx, valid_idx))

    missing = set(range(n_samples)) - covered_as_valid
    if missing:
        message = (
            f"'cv_indices' folds never use {len(missing)} of {n_samples} sample(s) "
            f"as validation data (e.g. indices {sorted(missing)[:5]}...)."
        )
        if strict_coverage:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning)

    return normalised


def generate_cv_indices(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.DataFrame | None = None,
    cv_iter: int = 5,
    cv_splitter=None,
    cv_indices: Iterable | None = None,
    groups: np.ndarray | pd.Series | None = None,
    task_type: Literal['classification', 'regression'] = 'classification',
    shuffle: bool = False,
    random_state: int | None = None,
    strict_coverage: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Resolve any supported cross-validation input into explicit fold indices.

    This is the single entry point that converts ``cv_iter`` (existing
    behaviour), a scikit-learn ``cv_splitter`` (optionally combined with
    ``groups``), or a user-supplied ``cv_indices`` manifest into the same
    canonical representation: a validated list of ``(train_idx, valid_idx)``
    index pairs. Fold-generation logic external to mlchem (scaffold splits,
    UMAP-cluster splits, etc.) only needs to produce ``cv_indices`` in this
    format.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Feature matrix of shape (n_samples, n_features). Only its length
        is used unless a ``cv_splitter`` requires ``X`` for splitting.

    y : numpy.ndarray, pandas.DataFrame, or None, optional
        Target vector, forwarded to ``cv_splitter.split`` when needed
        (e.g. for stratified splitters). Ignored when ``cv_indices`` is supplied.

    cv_iter : int, optional (default=5)
        Number of folds to generate when neither ``cv_splitter`` nor
        ``cv_indices`` is supplied. Mirrors the legacy behaviour.

    cv_splitter : object, optional
        A scikit-learn compatible cross-validation splitter (e.g.
        ``GroupKFold``, ``StratifiedGroupKFold``, ``PredefinedSplit``)
        exposing a ``split(X, y, groups)`` method. Ignored when ``cv_indices``
        is supplied.

    cv_indices : iterable of (array-like, array-like), optional
        Explicit, pre-computed ``(train_idx, valid_idx)`` pairs. Takes
        precedence over fold generation when provided; ``cv_iter``,
        ``cv_splitter``, ``groups``, ``task_type``, ``shuffle`` and
        ``random_state`` are then ignored.
        This is the most generic API and is suitable for externally
        generated fold manifests (scaffold-based folds, UMAP-cluster
        folds, etc.).

    groups : array-like, optional
        Group labels forwarded to the splitter when ``cv_indices`` is not
        supplied. The default splitters do not use groups; a custom
        ``cv_splitter`` determines whether to use them.

    task_type : {'classification', 'regression'}, optional (default='classification')
        Used only to pick the default splitter (``StratifiedKFold`` or
        ``KFold``) when neither ``cv_splitter`` nor ``cv_indices`` is given.

    shuffle : bool, optional (default=False)
        Whether to shuffle samples before splitting, for the default
        ``cv_iter``-based splitter. Ignored when ``cv_splitter`` or
        ``cv_indices`` is supplied.

    random_state : int or None, optional (default=None)
        Random seed for the default ``cv_iter``-based splitter when
        ``shuffle=True``. Ignored when ``shuffle=False``, or when
        ``cv_splitter`` or ``cv_indices`` is supplied.

    strict_coverage : bool, optional (default=False)
        Forwarded to :func:`validate_cv_indices`.

    Returns
    -------
    list of (numpy.ndarray, numpy.ndarray)
        Validated ``(train_idx, valid_idx)`` pairs.

    Examples
    --------
    Generate reproducible stratified folds from a small labeled dataset:

    >>> import pandas as pd
    >>> X = pd.DataFrame({"signal": [0, 1, 2, 3, 4, 5]})
    >>> y = [0, 0, 0, 1, 1, 1]
    >>> folds = generate_cv_indices(
    ...     X, y, cv_iter=3, shuffle=True, random_state=0
    ... )
    >>> len(folds)
    3
    """

    n_samples = _n_samples(X)

    if cv_indices is not None:
        return validate_cv_indices(cv_indices, n_samples, strict_coverage=strict_coverage)

    if cv_splitter is not None:
        splitter = cv_splitter
    else:
        validate_task_type(task_type)
        cv_kwargs = {
            'n_splits': cv_iter,
            'shuffle': shuffle,
        }
        if shuffle:
            cv_kwargs['random_state'] = random_state

        if task_type == 'classification':
            from sklearn.model_selection import StratifiedKFold
            splitter = StratifiedKFold(**cv_kwargs)
        else:
            from sklearn.model_selection import KFold
            splitter = KFold(**cv_kwargs)

    pairs = list(splitter.split(X, y, groups))
    return validate_cv_indices(pairs, n_samples, strict_coverage=strict_coverage)

def summarise_fold_scores(scores: Iterable[float]) -> dict:
    """
    Compute mean and standard error from fold scores.
    
    Parameters
    ----------
    scores : Iterable[float]
        Iterable of fold scores.

    Returns
    -------
    dict
        Dictionary containing the fold scores, the mean and
        standard error of the fold scores.

    Examples
    --------
    Summarise three fold scores:

    >>> summary = summarise_fold_scores([0.8, 0.9, 1.0])
    >>> summary["mean"]
    0.9
    >>> round(summary["se"], 3)
    0.047
    """
    scores_arr = np.asarray(list(scores), dtype=float)
    mean = float(np.mean(scores_arr))
    if len(scores_arr) <= 1:
        se = 0.0
    else:
        se = float(np.std(scores_arr) / np.sqrt(len(scores_arr)))
    return {
        'scores': scores_arr,
        'mean': mean,
        'se': se,
    }

def _make_metric_scorer(
    metric: Callable,
    response_method: Literal['predict', 'predict_proba', 'decision_function'],
):
    from sklearn.metrics import make_scorer

    return make_scorer(metric, response_method=response_method)


def _score_estimator(
    metric: Callable,
    estimator,
    X,
    y,
    response_method: Literal['predict', 'predict_proba', 'decision_function'],
) -> float:
    """Score a fitted estimator using the same response handling as crossval."""
    scorer = _make_metric_scorer(metric, response_method)
    return float(scorer(estimator, X, y))


def _second_class_probability(estimator, X) -> np.ndarray:
    """Return probability for ``classes_[1]``, as sklearn does for binary AUC."""
    probabilities = np.asarray(estimator.predict_proba(X))
    classes = np.asarray(estimator.classes_)
    if (
        probabilities.ndim != 2
        or classes.ndim != 1
        or probabilities.shape[1] != classes.size
        or classes.size < 2
    ):
        raise ValueError(
            "predict_proba output columns must align with at least two "
            "estimator.classes_ labels."
        )

    selected_class = classes[1]
    selected_indices = np.flatnonzero(classes == selected_class)
    if selected_indices.size != 1:
        raise ValueError(
            "Could not uniquely select estimator.classes_[1] from "
            "predict_proba output."
        )
    return probabilities[:, int(selected_indices[0])]


def crossval(estimator,
             X: np.ndarray | pd.DataFrame,
             y: np.ndarray | pd.DataFrame,
             metric: Callable,
             n_fold: int = 5,
             task_type: Literal['classification',
                                'regression'] = 'classification',
             random_state: int | None = None,
             shuffle: bool = False,
             cv_splitter=None,
             cv_indices: Iterable | None = None,
             groups: np.ndarray | pd.Series | None = None,
             response_method: Literal['predict',
                                      'predict_proba',
                                      'decision_function'] = 'predict',
             ) -> dict[str, np.ndarray | float]:
    """
Evaluate an estimator using cross-validation.

This function performs K-fold cross-validation on the given dataset using
the specified estimator and metric function. It supports both classification
and regression tasks.

By default (``cv_splitter=None`` and ``cv_indices=None``), behaviour is
unchanged from previous releases: ``n_fold`` folds are generated internally
using ``StratifiedKFold``/``KFold``. For fully index-driven, realistic
validation strategies (``GroupKFold``, ``StratifiedGroupKFold``,
``PredefinedSplit``, scaffold-based folds, UMAP-cluster folds, or any
externally generated fold manifest), pass ``cv_splitter`` and/or
``cv_indices`` instead. Internally, all cross-validation execution operates
on explicit, validated train/validation index pairs regardless of which
input mode is used (see :func:`generate_cv_indices`).

Parameters
----------
estimator : object
    A scikit-learn compatible estimator.

X : numpy.ndarray or pandas.DataFrame
    Feature matrix of shape (n_samples, n_features).

y : numpy.ndarray or pandas.DataFrame
    Target vector of shape (n_samples,) or (n_samples, 1).

metric : callable
    A scoring function that accepts (y_true, y_pred) as arguments.

response_method : {'predict', 'predict_proba', 'decision_function'}, optional (default='predict')
    Estimator method used to obtain metric inputs. The estimator must
    implement the selected method. For binary ``predict_proba``, sklearn
    selects the probability for ``estimator.classes_[1]``; for multiclass
    classification it passes the full score matrix. Use ``predict_proba`` or
    ``decision_function`` for ROC-AUC; label-based metrics should use
    ``predict``.

n_fold : int, optional (default=5)
    Number of folds for cross-validation. If equal to n_samples, performs
    leave-one-out cross-validation. Ignored when ``cv_splitter`` or
    ``cv_indices`` is provided.

task_type : {'classification', 'regression'}, optional (default='classification')
    Type of task to determine the cross-validation strategy.

random_state : int or None, optional (default=None)
    Random seed for reproducibility.

shuffle : bool, optional (default=False)
    Whether to shuffle samples before splitting into batches.

cv_splitter : object, optional
    A scikit-learn compatible cross-validation splitter (e.g.
    ``GroupKFold(n_splits=5)``, ``StratifiedGroupKFold(...)``, or
    ``PredefinedSplit(...)``). When provided, it takes precedence over
    ``n_fold``/``task_type``-based fold generation. Combine with
    ``groups`` for group-aware splitters.

cv_indices : iterable of (array-like, array-like), optional
    Explicit, pre-computed ``(train_idx, valid_idx)`` pairs (e.g. a
    scaffold-based or UMAP-cluster-based fold manifest generated outside
    mlchem). Takes precedence over both ``cv_splitter`` and ``n_fold``
    when provided. This is the most generic API.

groups : array-like, optional
    Group labels propagated to ``cv_splitter.split`` (e.g. scaffold IDs
    for ``GroupKFold``). Ignored when ``cv_indices`` is provided.

Returns
-------
dict
    Dictionary with fold-level arrays and summary statistics:
    ``train_scores``, ``train_mean``, ``train_se``,
    ``cv_scores``, ``cv_mean``, ``cv_se``.

Examples
--------
Evaluate binary ROC-AUC using predicted probabilities:

>>> from sklearn.datasets import make_classification
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.metrics import roc_auc_score
>>> X, y = make_classification(
...     n_samples=40, n_features=4, n_informative=2,
...     n_redundant=0, random_state=0
... )
>>> scores = crossval(
...     LogisticRegression(max_iter=1000),
...     X,
...     y,
...     roc_auc_score,
...     n_fold=4,
...     shuffle=True,
...     random_state=0,
...     response_method="predict_proba",
... )
>>> cv_mean = scores["cv_mean"]
"""

    from sklearn.model_selection import cross_validate
    scorer = _make_metric_scorer(metric, response_method)

    # Resolve cross-validation indices if any of cv_indices,
    # cv_splitter, or groups are provided.

    if cv_indices is not None or cv_splitter is not None or groups is not None:
        resolved_pairs = generate_cv_indices(
            X,
            y=y,
            cv_iter=n_fold,
            cv_splitter=cv_splitter,
            cv_indices=cv_indices,
            groups=groups,
            task_type=task_type,
            shuffle=shuffle,
            random_state=random_state,
        )
        result = cross_validate(estimator,
                                X,
                                y,
                                cv=resolved_pairs,
                                scoring=scorer,
                                groups=groups,
                                return_train_score=True,)
    else:
        # ---- Legacy path: unchanged behaviour for n_fold. ----
        validate_task_type(task_type)

        cv_kwargs = {
            'n_splits': n_fold,
            'shuffle': shuffle,
        }
        if shuffle:
            cv_kwargs['random_state'] = random_state

        if task_type == 'classification':
            from sklearn.model_selection import StratifiedKFold
            result = cross_validate(estimator,
                                    X,
                                    y,
                                    cv=StratifiedKFold(**cv_kwargs),
                                    scoring=scorer,
                                    return_train_score=True,)
        else:
            from sklearn.model_selection import KFold
            result = cross_validate(estimator,
                                    X,
                                    y,
                                    cv=KFold(**cv_kwargs),
                                    scoring=scorer,
                                    return_train_score=True)
    
    train_summary = summarise_fold_scores(result['train_score'])
    cv_summary = summarise_fold_scores(result['test_score'])
    return {
        'train_scores': train_summary['scores'],
        'train_mean': train_summary['mean'],
        'train_se': train_summary['se'],
        'cv_scores': cv_summary['scores'],
        'cv_mean': cv_summary['mean'],
        'cv_se': cv_summary['se'],
    }


def y_scrambling(estimator,
                 train_set: np.ndarray | pd.DataFrame,
                 y_train: Iterable,
                 metric: Callable,
                 n_scrambles: int = 100,
                 n_fold: int = 5,
                 cv_splitter=None,
                 groups=None,
                 cv_indices: Iterable | None = None,
                 logic: Literal['lower', 'greater'] = 'greater',
                 task_type: Literal[
                     'classification', 'regression'] = 'classification',
                 plot: bool = True,
                 n_jobs: int = 1,
                 safety_multiplier: float = 2.3,
                 log_level: int | str | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = logging.INFO,
                 response_method: Literal['predict',
                                           'predict_proba',
                                           'decision_function'] = 'predict',
                 ) -> dict[str, float | np.ndarray]:
    """
    Perform y-scrambling to estimate chance-level model performance.

    The target vector is shuffled ``n_scrambles`` times, each shuffled target is
    cross-validated, and the null-score distribution is compared with the score
    obtained from the unshuffled labels.

    Parameters
    ----------
    estimator : object
        Scikit-learn compatible estimator.
    train_set : numpy.ndarray or pandas.DataFrame
        Training feature matrix.
    y_train : iterable
        Target values.
    metric : callable
        Scoring function ``metric(y_true, y_pred)``.
    n_scrambles : int, optional
        Number of shuffled-label iterations. Default is 100.
    n_fold : int, optional
        Number of CV folds when explicit folds are not provided.
    cv_splitter : object, optional
        CV splitter overriding ``n_fold``.
    groups : array-like, optional
        Group labels passed to group-aware splitters.
    cv_indices : iterable, optional
        Explicit fold definitions overriding ``cv_splitter`` and ``n_fold``.
    logic : {'lower', 'greater'}, optional
        Score direction used for comparisons.
    task_type : {'classification', 'regression'}, optional
        Task type used by default CV strategy.
    plot : bool, optional
        If True, display a histogram of scrambled scores.
    n_jobs : int, optional
        Number of parallel workers. Use -1 for all CPUs.
    safety_multiplier : float, optional
        Multiplier defining ``safety_margin = safety_multiplier * scrambled_std``.
    log_level : int or str, optional
        Logging threshold.
    response_method : {'predict', 'predict_proba', 'decision_function'}, optional
        Estimator method used to obtain metric inputs. Defaults to ``predict``.
        For ROC-AUC, use ``predict_proba`` or ``decision_function``.

    Returns
    -------
    dict
        Mapping with keys:

        - ``reference_score``
        - ``scrambled_scores``
        - ``scrambled_std``
        - ``best_random_score``
        - ``probability_better``
        - ``obtained_margin``
        - ``safety_margin``
        - ``safety_margin_ratio``
        - ``safety_multiplier``

    Examples
    --------
    Compare a binary ROC-AUC reference score with a small scrambled-label
    distribution:

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.metrics import roc_auc_score
    >>> X, y = make_classification(
    ...     n_samples=40, n_features=4, n_informative=2,
    ...     n_redundant=0, random_state=0
    ... )
    >>> result = y_scrambling(
    ...     estimator=LogisticRegression(max_iter=1000),
    ...     train_set=X,
    ...     y_train=y,
    ...     metric=roc_auc_score,
    ...     n_scrambles=10,
    ...     n_fold=4,
    ...     plot=False,
    ...     response_method="predict_proba",
    ... )
    >>> reference_score = result["reference_score"]
    """

    from sklearn.base import clone
    import random

    resolved_log_level = coerce_log_level(log_level)
    _configure_module_logging(resolved_log_level)
    n_jobs = resolve_n_jobs(n_jobs)
    if safety_multiplier <= 0:
        raise ValueError("'safety_multiplier' must be greater than 0.")

    estimator_copy = clone(estimator)
    y_train_copy = list(y_train) if not isinstance(y_train, list) else y_train.copy()
    if isinstance(train_set, pd.DataFrame):
        X_train = train_set.values
    else:
        X_train = train_set

    reference = crossval(
        estimator_copy,
        X_train,
        y_train_copy,
        metric,
        n_fold=n_fold,
        task_type=task_type,
        cv_splitter=cv_splitter,
        cv_indices=cv_indices,
        groups=groups,
        response_method=response_method,
    )
    ref_score = reference['cv_mean']

    def evaluate_scramble(seed_val):
        """Evaluate model performance on a single scrambled iteration."""
        rng = random.Random(seed_val)
        y_shuffled = y_train_copy.copy()
        rng.shuffle(y_shuffled)
        est = clone(estimator)
        disable_estimator_parallelization(est)  # Prevent nested parallelization warnings
        cv_result = crossval(
            est,
            X_train,
            y_shuffled,
            metric,
            n_fold=n_fold,
            task_type=task_type,
            cv_splitter=cv_splitter,
            cv_indices=cv_indices,
            groups=groups,
            response_method=response_method,
        )
        return cv_result['cv_mean']

    # Run scrambling iterations in parallel
    scores = []
    if n_jobs == 1:
        # Sequential execution
        for i in tqdm(range(n_scrambles), desc="Y-scrambling", disable=False):
            scores.append(evaluate_scramble(i))
    else:
        # Parallel execution with ThreadPoolExecutor
        max_workers = n_jobs if n_jobs > 0 else None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_scramble, i): i for i in range(n_scrambles)}
            for future in tqdm(as_completed(futures), total=n_scrambles, desc="Y-scrambling", disable=False):
                scores.append(future.result())

    scores = np.array(scores)
    ys_max = max(scores)
    ys_min = min(scores)
    ys_std = np.std(scores)
    safety_margin = safety_multiplier * ys_std

    if logic == 'greater':
        value = len(scores[scores >= (ref_score)])/len(scores)
        best_random_score = ys_max
        obtained_margin = max(ref_score - best_random_score, 0)
    else: # logic == 'lower'
        value = len(scores[scores <= (ref_score)])/len(scores)
        best_random_score = ys_min
        obtained_margin = max(best_random_score - ref_score, 0)

    # Rucker et al, https://doi.org/10.1021/ci700157b
    # (generalised with user-configurable safety_multiplier)
    safety_margin_ratio = obtained_margin / safety_margin if safety_margin > 0 else np.nan

    logger.log(
        resolved_log_level,
        f'Probability to obtain a better model by chance: {value:.3f}')
    logger.log(
        resolved_log_level,
        f'Safety margin: {safety_margin_ratio:.2f}')

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.histplot(scores)
        plt.axvline(ref_score,
                    color='red',
                    linestyle='--')
        if logic == 'greater':
            plt.axvspan(best_random_score,
                        best_random_score + safety_margin,
                        color='red',
                        alpha=0.1)
        else:
            plt.axvspan(best_random_score - safety_margin,
                        best_random_score,
                        color='red',
                        alpha=0.1)
        plt.show()

    return {'reference_score': ref_score,
            'scrambled_scores': scores,
            'scrambled_std': ys_std,
            'best_random_score': best_random_score,
            'probability_better': value,
            'obtained_margin': obtained_margin,
            'safety_margin': safety_margin,
            'safety_margin_ratio': safety_margin_ratio,
            'safety_multiplier': safety_multiplier,
            }

@dataclass
class _MajorityVoteCVCache:
    task_type: Literal['classification', 'regression']
    estimator_names: list[str]
    metrics: list[Callable]
    metric_names: list[str]
    probabilities: list[bool]
    logics: list[Literal['lower', 'greater']]
    desired_performance_score: Literal['train', 'cv', 'train_cv_average']
    folds: dict[str, tuple[
        list[tuple[dict[str, np.ndarray], np.ndarray]],
        list[tuple[dict[str, np.ndarray], np.ndarray]],
    ]]
    cv_indices: list[tuple[np.ndarray, np.ndarray]]
    token: str


class MajorityVote:
    """
    MajorityVote ensemble evaluation utility.

    ``evaluate_cv()`` fits estimators on CV folds and can score individual
    estimators and ensemble combinations in one call. Alternatively, use
    ``individual_only=True`` to screen individual estimators, then pass the
    filtered results to ``evaluate_consensus_cv()`` to score shortlisted
    combinations using the same fold predictions without refitting.
    ``fit()`` separately fits the final consensus on full train/test data.

    Parameters
    ----------
    train_set : pandas.DataFrame
        Training feature matrix.
    y_train : iterable
        Training labels/targets.
    task_type : {'classification', 'regression'}
        Task type used to select voting strategy.
    estimator_list : list
        List of scikit-learn compatible estimators.
        Classification estimators must provide ``predict_proba`` for soft
        voting.
    column_list : list[list[str]]
        Per-estimator feature subsets aligned with ``estimator_list``.
        For ``fit()``, each listed column must exist in both the training
        and hold-out DataFrames.
    test_set : pandas.DataFrame or None, optional
        Hold-out feature matrix used by ``fit()``.
    y_test : iterable or None, optional
        Hold-out labels/targets used by ``fit()``.
    estimator_names : list[str] or None, optional
        Optional user-defined estimator names.
    n_jobs : int, optional
        Number of workers used by parallel sections.
    log_level : int or str, optional
        Logging threshold.
    """

    def __init__(
        self,
        train_set: pd.DataFrame,
        y_train: Iterable,
        task_type: Literal['classification', 'regression'],
        estimator_list: list,
        column_list: list[list[str]],
        test_set: pd.DataFrame | None = None,
        y_test: Iterable | None = None,
        estimator_names: list[str] | None = None,
        n_jobs: int = 1,
        log_level: int | str | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = logging.INFO,
    ) -> None:

        self.task_type = validate_task_type(task_type)
        self.estimator_list = list(estimator_list)
        self.estimator_names = [] if estimator_names is None else list(estimator_names)
        self.column_list = list(column_list)
        self.train_set = train_set
        self.test_set = test_set
        self.y_train = y_train
        self.y_test = y_test
        self.n_jobs = resolve_n_jobs(n_jobs)
        self.log_level = coerce_log_level(log_level)
        self._cv_cache: _MajorityVoteCVCache | None = None
        self.cv_indices_: list[tuple[np.ndarray, np.ndarray]] | None = None
        _configure_module_logging(self.log_level)

    def _log(self, level: int, msg: str, *args) -> None:
        if level >= self.log_level:
            logger.log(level, msg, *args)

    @staticmethod
    def _mean_and_se(values: Iterable[float]) -> tuple[float, float]:
        arr = np.asarray(list(values), dtype=float)
        mean = float(np.mean(arr))
        if arr.size <= 1:
            return mean, 0.0
        se = float(np.std(arr) / np.sqrt(arr.size))
        return mean, se

    @staticmethod
    def _hard_vote(pred_matrix: np.ndarray) -> np.ndarray:
        from scipy.stats import mode
        voted = mode(pred_matrix, axis=1, keepdims=False)[0]
        return np.asarray(voted).reshape(-1)

    @staticmethod
    def _soft_vote(prob_matrix: np.ndarray) -> np.ndarray:
        return np.round(prob_matrix.mean(axis=1))

    @staticmethod
    def _mean_vote(pred_matrix: np.ndarray) -> np.ndarray:
        return pred_matrix.mean(axis=1)

    def _resolved_estimator_names(self) -> list[str]:
        names = []
        used = set()
        for i, estimator in enumerate(self.estimator_list):
            base_name = self.estimator_names[i] if i < len(self.estimator_names) else str(estimator)
            candidate = str(base_name)
            suffix = 2
            while candidate in used:
                candidate = f"{base_name}_{suffix}"
                suffix += 1
            used.add(candidate)
            names.append(candidate)
        return names

    def _resolve_selected_estimators(
        self,
        selected_estimators: Iterable[int | str] | None = None,
    ) -> list[tuple[int, object, list[str], str]]:
        """Resolve user-selected estimators to indexed estimator payloads.

        Parameters
        ----------
        selected_estimators : iterable of int or str, optional
            Estimator indices and/or resolved estimator names. If None,
            all configured estimators are selected.

        Returns
        -------
        list of tuple
            Tuples of ``(index, estimator, columns, estimator_name)``.
        """
        resolved_names = self._resolved_estimator_names()

        if selected_estimators is None:
            selected_idx = list(range(len(self.estimator_list)))
        else:
            selected_idx = []
            for item in selected_estimators:
                if isinstance(item, int):
                    idx = item
                    if idx < 0 or idx >= len(self.estimator_list):
                        raise ValueError(
                            f"Estimator index {idx} is out of bounds for "
                            f"{len(self.estimator_list)} estimator(s)."
                        )
                elif isinstance(item, str):
                    if item not in resolved_names:
                        raise ValueError(
                            f"Unknown estimator name '{item}'. Available names: "
                            f"{resolved_names}."
                        )
                    idx = resolved_names.index(item)
                else:
                    raise TypeError(
                        "'selected_estimators' items must be either int indices "
                        "or str estimator names."
                    )

                if idx not in selected_idx:
                    selected_idx.append(idx)

            if len(selected_idx) == 0:
                raise ValueError("'selected_estimators' must not be empty.")

        return [
            (idx, self.estimator_list[idx], self.column_list[idx], resolved_names[idx])
            for idx in selected_idx
        ]

    def fit(
        self,
        selected_estimators: Iterable[int | str] | None = None,
        update_active_estimators: bool = False,
        build_consensus: bool = True,
        return_prediction_dataframes: bool = False,
    ) -> dict[str, list[str] | int] | dict:
        """
        Hold-out fitting using train and test sets.

        Parameters
        ----------
        selected_estimators : iterable of int or str, optional
            Optional subset of estimators to fit, provided as indices
            and/or resolved estimator names. If None, all estimators are
            fitted.

        update_active_estimators : bool, optional (default=False)
            If True, permanently replace ``estimator_list`` / ``column_list`` /
            ``estimator_names`` with the selected subset after resolution.
            This is useful for intentionally narrowing the final
            consensus fit stage to shortlisted estimators.

        build_consensus : bool, optional (default=True)
            For classification tasks, build transparent hold-out prediction
            tables with per-estimator probability/class columns plus
            ``CONSENSUS_hard`` and ``CONSENSUS_soft``.

        return_prediction_dataframes : bool, optional (default=False)
            If True, return the generated prediction DataFrames together
            with the fit report.

        Returns
        -------
        dict
            Fit report with keys:
            ``requested_count``, ``successful_count``, ``failed_count``,
            ``successful_estimators``, ``failed_estimators``.
            When ``return_prediction_dataframes=True``, returns a dictionary
            containing the fit report and generated prediction DataFrames.

        Examples
        --------
        Fit one classifier on the training set and evaluate it on a hold-out set:

        >>> import pandas as pd
        >>> from sklearn.linear_model import LogisticRegression
        >>> train_set = pd.DataFrame({"signal": [0, 1, 2, 3]})
        >>> test_set = pd.DataFrame({"signal": [4, 5]})
        >>> vote = MajorityVote(
        ...     train_set=train_set,
        ...     y_train=[0, 0, 1, 1],
        ...     task_type="classification",
        ...     estimator_list=[LogisticRegression(max_iter=1000)],
        ...     column_list=[["signal"]],
        ...     test_set=test_set,
        ...     y_test=[1, 1],
        ...     estimator_names=["logreg"],
        ...     n_jobs=1,
        ... )
        >>> report = vote.fit(selected_estimators=["logreg"])
        >>> report["successful_count"]
        1
        """
        if self.test_set is None or self.y_test is None:
            raise ValueError(
                "MajorityVote.fit requires 'test_set' and 'y_test'. "
                "Use evaluate_cv() for cross-validation-centric evaluation."
            )

        selected_estimators_resolved = self._resolve_selected_estimators(selected_estimators)
        for _, _, columns, estimator_name in selected_estimators_resolved:
            validate_dataframe_columns(
                self.train_set,
                columns,
                dataframe_name=f"train_set for estimator '{estimator_name}'",
            )
            validate_dataframe_columns(
                self.test_set,
                columns,
                dataframe_name=f"test_set for estimator '{estimator_name}'",
            )

        if update_active_estimators:
            self.estimator_list = [payload[1] for payload in selected_estimators_resolved]
            self.column_list = [payload[2] for payload in selected_estimators_resolved]
            self.estimator_names = [payload[3] for payload in selected_estimators_resolved]
            selected_estimators_resolved = self._resolve_selected_estimators(None)

        self._log(
            logging.INFO,
            "MajorityVote.fit consensus stage: requested_estimators=%d",
            len(selected_estimators_resolved),
        )

        successful_estimators = []
        failed_estimators = []

        def fit_and_predict(est_data):
            """Fit a single estimator and return predictions."""
            _, estimator, columns, estimator_name = est_data
            X_train = self.train_set[columns]
            X_train = X_train.loc[:, ~X_train.columns.duplicated(keep='first')].copy()
            X_test = self.test_set[columns]
            X_test = X_test.loc[:, ~X_test.columns.duplicated(keep='first')].copy()

            try:
                from sklearn.base import clone
                estimator_copy = clone(estimator)
                disable_estimator_parallelization(estimator_copy)
                estimator_copy.fit(X_train, self.y_train)
                if self.task_type == 'classification':
                    y_train_hard = estimator_copy.predict(X_train)
                    y_test_hard = estimator_copy.predict(X_test)
                    y_train_soft = _second_class_probability(estimator_copy, X_train)
                    y_test_soft = _second_class_probability(estimator_copy, X_test)
                    return (
                        estimator_name,
                        'classification',
                        y_train_hard,
                        y_test_hard,
                        y_train_soft,
                        y_test_soft,
                    )
                y_train_pred = estimator_copy.predict(X_train)
                y_test_pred = estimator_copy.predict(X_test)
                return (estimator_name, 'regression', y_train_pred, y_test_pred)
            except Exception as ex:
                warnings.warn(
                    (
                        f"Skipping estimator '{estimator_name}' during "
                        f"MajorityVote.fit ({self.task_type}): {ex}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None

        if self.task_type == 'classification':
            self.df_train_predictions_hard = pd.DataFrame(index=self.train_set.index)
            self.df_test_predictions_hard = pd.DataFrame(index=self.test_set.index)

            self.df_train_predictions_soft = pd.DataFrame(index=self.train_set.index)
            self.df_test_predictions_soft = pd.DataFrame(index=self.test_set.index)
            train_hard_predictions = {}
            test_hard_predictions = {}
            train_soft_predictions = {}
            test_soft_predictions = {}

            est_data_list = selected_estimators_resolved

            if self.n_jobs == 1:
                results = [fit_and_predict(data) for data in tqdm(est_data_list, desc="Fitting estimators", disable=False)]
            else:
                max_workers = self.n_jobs if self.n_jobs > 0 else None
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(fit_and_predict, data): data for data in est_data_list}
                    results = []
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Fitting estimators", disable=False):
                        results.append(future.result())

            for result in results:
                if result is None:
                    continue
                estimator_name, _, y_train_hard, y_test_hard, y_train_soft, y_test_soft = result
                successful_estimators.append(estimator_name)
                train_hard_predictions[estimator_name] = y_train_hard
                test_hard_predictions[estimator_name] = y_test_hard
                train_soft_predictions[estimator_name] = y_train_soft
                test_soft_predictions[estimator_name] = y_test_soft

            failed_estimators = [
                payload[3] for payload in est_data_list
                if payload[3] not in successful_estimators
            ]

            if not train_hard_predictions:
                raise RuntimeError(
                    "No estimators were successfully fitted in "
                    "MajorityVote.fit for classification."
                )

            train_hard_predictions['Y'] = self.y_train
            test_hard_predictions['Y'] = self.y_test
            train_soft_predictions['Y'] = self.y_train
            test_soft_predictions['Y'] = self.y_test
            self.df_train_predictions_hard = pd.DataFrame(
                train_hard_predictions,
                index=self.train_set.index,
            )
            self.df_test_predictions_hard = pd.DataFrame(
                test_hard_predictions,
                index=self.test_set.index,
            )
            self.df_train_predictions_soft = pd.DataFrame(
                train_soft_predictions,
                index=self.train_set.index,
            )
            self.df_test_predictions_soft = pd.DataFrame(
                test_soft_predictions,
                index=self.test_set.index,
            )

            if build_consensus:
                estimator_cols = list(self.df_train_predictions_hard.columns[:-1])

                train_predictions = {'Y': self.y_train}
                test_predictions = {'Y': self.y_test}

                for est_name in estimator_cols:
                    train_predictions[f'{est_name}_proba'] = self.df_train_predictions_soft[est_name].values
                    train_predictions[f'{est_name}_CLASS'] = self.df_train_predictions_hard[est_name].values
                    test_predictions[f'{est_name}_proba'] = self.df_test_predictions_soft[est_name].values
                    test_predictions[f'{est_name}_CLASS'] = self.df_test_predictions_hard[est_name].values

                hard_train_matrix = self.df_train_predictions_hard[estimator_cols].to_numpy(copy=False)
                hard_test_matrix = self.df_test_predictions_hard[estimator_cols].to_numpy(copy=False)
                soft_train_matrix = self.df_train_predictions_soft[estimator_cols].to_numpy(copy=False)
                soft_test_matrix = self.df_test_predictions_soft[estimator_cols].to_numpy(copy=False)

                train_predictions['CONSENSUS_hard'] = self._hard_vote(hard_train_matrix)
                test_predictions['CONSENSUS_hard'] = self._hard_vote(hard_test_matrix)
                train_predictions['CONSENSUS_soft'] = self._soft_vote(soft_train_matrix)
                test_predictions['CONSENSUS_soft'] = self._soft_vote(soft_test_matrix)
                self.df_train_predictions = pd.DataFrame(
                    train_predictions,
                    index=self.train_set.index,
                )
                self.df_test_predictions = pd.DataFrame(
                    test_predictions,
                    index=self.test_set.index,
                )

        else:
            self.df_train_predictions = pd.DataFrame(index=self.train_set.index)
            self.df_test_predictions = pd.DataFrame(index=self.test_set.index)
            train_predictions = {}
            test_predictions = {}

            est_data_list = selected_estimators_resolved

            if self.n_jobs == 1:
                results = [fit_and_predict(data) for data in tqdm(est_data_list, desc="Fitting estimators", disable=False)]
            else:
                max_workers = self.n_jobs if self.n_jobs > 0 else None
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(fit_and_predict, data): data for data in est_data_list}
                    results = []
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Fitting estimators", disable=False):
                        results.append(future.result())

            for result in results:
                if result is None:
                    continue
                estimator_name, _, y_train_pred, y_test_pred = result
                successful_estimators.append(estimator_name)
                train_predictions[estimator_name] = y_train_pred
                test_predictions[estimator_name] = y_test_pred

            failed_estimators = [
                payload[3] for payload in est_data_list
                if payload[3] not in successful_estimators
            ]

            if not train_predictions:
                raise RuntimeError(
                    "No estimators were successfully fitted in "
                    "MajorityVote.fit for regression."
                )

            train_predictions['Y'] = self.y_train
            test_predictions['Y'] = self.y_test
            self.df_train_predictions = pd.DataFrame(
                train_predictions,
                index=self.train_set.index,
            )
            self.df_test_predictions = pd.DataFrame(
                test_predictions,
                index=self.test_set.index,
            )

        self.fit_report_ = {
            'requested_count': len(selected_estimators_resolved),
            'successful_count': len(successful_estimators),
            'failed_count': len(failed_estimators),
            'successful_estimators': successful_estimators,
            'failed_estimators': failed_estimators,
        }
        self._log(
            logging.INFO,
            f"MajorityVote.fit consensus stage completed: "
            f"successful={self.fit_report_['successful_count']} "
            f"failed={self.fit_report_['failed_count']}",
        )

        if return_prediction_dataframes:
            response = {'fit_report': self.fit_report_}
            if self.task_type == 'classification':
                response['train_predictions_hard'] = self.df_train_predictions_hard
                response['test_predictions_hard'] = self.df_test_predictions_hard
                response['train_predictions_soft'] = self.df_train_predictions_soft
                response['test_predictions_soft'] = self.df_test_predictions_soft
                if build_consensus:
                    response['train_predictions'] = self.df_train_predictions
                    response['test_predictions'] = self.df_test_predictions
            else:
                response['train_predictions'] = self.df_train_predictions
                response['test_predictions'] = self.df_test_predictions
            return response

        return self.fit_report_

    def evaluate_cv(
        self,
        metric: Callable | Sequence[Callable],
        metric_name: str | Sequence[str],
        use_continuous_probabilities: bool | Sequence[bool] = False,
        cv_iter: int = 5,
        cv_splitter=None,
        cv_indices: Iterable | None = None,
        groups: np.ndarray | pd.Series | None = None,
        n_estimators_max: int = 5,
        logic: Literal['lower', 'greater'] | Sequence[Literal['lower', 'greater']] = 'greater',
        desired_performance_score: Literal['train', 'cv', 'train_cv_average'] = 'train_cv_average',
        shuffle: bool = False,
        random_state: int | None = None,
        individual_only: bool = False,
    ) -> pd.DataFrame:
        """
Evaluate ensemble combinations using fold-level cross-validation.

The same explicit fold definitions are reused for every estimator and
combination, ensuring fair fold-wise comparison. By default this method fits
the estimators and scores both individual estimators and their combinations
in one call. For a screening step before choosing combinations, set
``individual_only=True`` and pass the filtered results to
``evaluate_consensus_cv``. Calling ``evaluate_cv`` again starts a new fit
and replaces the cached screening predictions.

Parameters
----------
metric : callable or sequence of callable
    Scoring functions accepting ``(y_true, y_pred)``. Multiple metrics
    share the same fitted estimators and fold predictions.

metric_name : str or sequence of str
    Unique labels aligned with ``metric``, appended to combination identifiers.

use_continuous_probabilities : bool or sequence of bool, optional
    For classification soft-vote strategies, use mean predicted
    probabilities directly instead of rounding them to class labels. Enable
    this for binary ROC-AUC; the probability for ``classes_[1]`` is used.
    A sequence supplies one setting per metric; a scalar applies to all.
    Does not affect regression or hard-vote predictions, which use predicted
    class labels for classification. Values are still validated.

cv_iter : int, optional (default=5)
    Number of folds used when ``cv_indices`` and ``cv_splitter`` are not
    supplied. Ignored if either is supplied.

cv_splitter : object, optional
    Scikit-learn compatible splitter exposing ``split(X, y, groups)``.
    Ignored when ``cv_indices`` is supplied.

cv_indices : iterable of (train_idx, valid_idx), optional
    Explicit precomputed fold manifest. When supplied, all combinations
    are evaluated on exactly these folds, and ``cv_iter``, ``cv_splitter``,
    ``groups``, ``shuffle`` and ``random_state`` are ignored.

groups : array-like, optional
    Group labels forwarded to the splitter unless ``cv_indices`` is supplied.
    Default splitters do not use groups; a custom splitter determines whether
    to use them.

n_estimators_max : int, optional (default=5)
    Maximum combination size passed to ``generate_combination_cascade``.
    Ignored when ``individual_only=True``.

logic : {'lower', 'greater'} or sequence, optional (default='greater')
    Direction of optimization for reliability computation. A sequence supplies
    one direction per metric; a scalar applies to all.

desired_performance_score : {'train', 'cv', 'train_cv_average'}, optional
    Performance component used inside
    ``get_reliability_score_components``. Also used in individual-only mode
    and reused by ``evaluate_consensus_cv``.

shuffle : bool, optional (default=False)
    Whether generated CV folds are shuffled when using ``cv_iter``.
    Ignored when ``cv_splitter`` or ``cv_indices`` is supplied.

random_state : int or None, optional
    Random seed used with generated shuffled folds. Ignored when
    ``shuffle=False``, or when ``cv_splitter`` or ``cv_indices`` is supplied.
    Configure a custom splitter's randomness on the splitter itself.

individual_only : bool, optional (default=False)
    Score only individual estimators and retain their fold predictions for
    ``evaluate_consensus_cv``. The returned DataFrame includes
    ``estimator`` and ``strategy`` columns and per-fold train and validation
    prediction arrays.
    The corresponding row positions are available in ``cv_indices_``.

Returns
-------
pandas.DataFrame
    One row per evaluated ensemble variant. Columns include:
    ``combination``, ``train_mean``, ``train_se``, ``train_min``, ``train_max``,
    ``validation_mean``, ``validation_se``, ``validation_min``, ``validation_max``,
    ``reliability_mean``, ``reliability_se``, ``reliability_min``, ``reliability_max``,
    ``train_folds``, ``validation_folds``, ``reliability_folds``.
    Minima and maxima are calculated across the corresponding fold values.
    Results are ranked by ``reliability_mean`` descending. With multiple
    metrics, a ``metric`` column identifies each row and results are grouped
    in metric input order, ranked separately within each group.
    Train/validation columns contain raw metric values. Reliability is
    computed per fold, then averaged: for ``logic='lower'`` it is the
    negative of the selected loss plus ``abs(train - validation)``;
    for ``logic='greater'`` the existing performance divided by
    ``1 + abs(train - validation)`` is unchanged. Reliability is always
    maximized, is not raw RMSE, and is nonpositive for nonnegative losses.
    In individual-only mode, ``train_predictions_folds`` and
    ``validation_predictions_folds`` contain the predictions used by each
    row's metric and strategy, in the order of ``cv_indices_``.

Examples
--------
Rank classifier combinations by cross-validated ROC-AUC. The continuous
probability option keeps soft-vote scores suitable for ROC-AUC:

>>> import pandas as pd
>>> from sklearn.datasets import make_classification
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.metrics import roc_auc_score
>>> raw_X, y = make_classification(
...     n_samples=60, n_features=4, n_informative=2,
...     n_redundant=0, random_state=0
... )
>>> X = pd.DataFrame(raw_X, columns=["f1", "f2", "f3", "f4"])
>>> columns = X.columns.tolist()
>>> ensemble = MajorityVote(
...     train_set=X,
...     y_train=y,
...     task_type="classification",
...     estimator_list=[
...         LogisticRegression(max_iter=1000),
...         RandomForestClassifier(n_estimators=20, random_state=0),
...         LogisticRegression(C=0.5, max_iter=1000),
...     ],
...     column_list=[columns, columns, columns],
...     estimator_names=["logistic", "forest", "logistic_alt"],
... )
>>> results = ensemble.evaluate_cv(
...     metric=roc_auc_score,
...     metric_name="AUC",
...     cv_iter=3,
...     n_estimators_max=2,
...     use_continuous_probabilities=True,
...     shuffle=True,
...     random_state=0,
... )
>>> results_summary = results[["combination", "validation_mean"]].head()

Multiple metrics can use different optimization directions and prediction
types without refitting:

>>> from sklearn.metrics import accuracy_score
>>> results = ensemble.evaluate_cv(
...     metric=[accuracy_score, roc_auc_score],
...     metric_name=["accuracy", "AUC"],
...     use_continuous_probabilities=[False, True],
...     cv_iter=3,
...     n_estimators_max=2,
... )

Shortlist individual models before evaluating ensemble combinations:

>>> individual = ensemble.evaluate_cv(
...     metric=accuracy_score, metric_name="accuracy", cv_iter=3,
...     individual_only=True,
... )
>>> best = individual.groupby("estimator")["validation_mean"].max().nlargest(3).index
>>> shortlist = individual.loc[individual["estimator"].isin(best)].copy()
>>> consensus = ensemble.evaluate_consensus_cv(shortlist, n_estimators_max=3)
"""

        from sklearn.base import clone

        self._cv_cache = None
        self.cv_indices_ = None

        if not isinstance(individual_only, bool):
            raise ValueError("'individual_only' must be a bool.")

        metrics = list(metric) if isinstance(metric, Sequence) else [metric]
        metric_names = (
            list(metric_name)
            if isinstance(metric_name, Sequence) and not isinstance(metric_name, str)
            else [metric_name]
        )
        if not metrics or any(not callable(item) for item in metrics):
            raise ValueError("'metric' must contain at least one callable.")
        if len(metric_names) != len(metrics):
            raise ValueError("'metric_name' must contain one name per metric.")
        if any(not isinstance(name, str) or not name for name in metric_names):
            raise ValueError("Metric names must be non-empty strings.")
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("Metric names must be unique.")

        logics = (
            list(logic)
            if isinstance(logic, Sequence) and not isinstance(logic, str)
            else [logic] * len(metrics)
        )
        probabilities = (
            list(use_continuous_probabilities)
            if isinstance(use_continuous_probabilities, Sequence)
            and not isinstance(use_continuous_probabilities, (str, bytes))
            else [use_continuous_probabilities] * len(metrics)
        )
        if len(logics) != len(metrics) or any(item not in ('lower', 'greater') for item in logics):
            raise ValueError("'logic' must specify 'lower' or 'greater' for each metric.")
        if len(probabilities) != len(metrics) or any(
            not isinstance(item, bool) for item in probabilities
        ):
            raise ValueError("'use_continuous_probabilities' must specify a bool for each metric.")

        y_all = np.asarray(self.y_train)
        resolved_pairs = generate_cv_indices(
            self.train_set.values,
            y=y_all,
            cv_iter=cv_iter,
            cv_splitter=cv_splitter,
            cv_indices=cv_indices,
            groups=groups,
            task_type=self.task_type,
            shuffle=shuffle,
            random_state=random_state,
        )

        estimator_names = self._resolved_estimator_names()

        fold_train_hard = []
        fold_valid_hard = []
        fold_train_soft = []
        fold_valid_soft = []
        fold_train_reg = []
        fold_valid_reg = []

        valid_estimators = set(estimator_names)

        for fold_number, (train_idx, valid_idx) in enumerate(
            tqdm(resolved_pairs, desc="CV folds", disable=False),
            start=1,
        ):
            y_train_fold = y_all[train_idx]
            y_valid_fold = y_all[valid_idx]

            fold_data_train_hard = {}
            fold_data_valid_hard = {}
            fold_data_train_soft = {}
            fold_data_valid_soft = {}
            fold_data_train_reg = {}
            fold_data_valid_reg = {}

            for i, (estimator, columns) in enumerate(
                tqdm(
                    zip(self.estimator_list, self.column_list),
                    total=len(self.estimator_list),
                    desc=f"Fitting estimators (fold {fold_number}/{len(resolved_pairs)})",
                    leave=False,
                    disable=False,
                )
            ):
                estimator_name = estimator_names[i]
                if estimator_name not in valid_estimators:
                    continue

                X_cols = self.train_set[columns]
                X_cols = X_cols.loc[:, ~X_cols.columns.duplicated(keep='first')]
                X_train_fold = X_cols.iloc[train_idx]
                X_valid_fold = X_cols.iloc[valid_idx]

                try:
                    est = clone(estimator)
                    disable_estimator_parallelization(est)
                    est.fit(X_train_fold, y_train_fold)

                    if self.task_type == 'classification':
                        fold_data_train_hard[estimator_name] = np.asarray(est.predict(X_train_fold))
                        fold_data_valid_hard[estimator_name] = np.asarray(est.predict(X_valid_fold))
                        fold_data_train_soft[estimator_name] = _second_class_probability(
                            est, X_train_fold
                        )
                        fold_data_valid_soft[estimator_name] = _second_class_probability(
                            est, X_valid_fold
                        )
                    else:
                        fold_data_train_reg[estimator_name] = np.asarray(est.predict(X_train_fold))
                        fold_data_valid_reg[estimator_name] = np.asarray(est.predict(X_valid_fold))
                except Exception as ex:
                    valid_estimators.discard(estimator_name)
                    warnings.warn(
                        (
                            f"Skipping estimator '{estimator_name}' during "
                            f"MajorityVote.evaluate_cv ({self.task_type}): {ex}"
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )

            if self.task_type == 'classification':
                fold_train_hard.append((fold_data_train_hard, y_train_fold))
                fold_valid_hard.append((fold_data_valid_hard, y_valid_fold))
                fold_train_soft.append((fold_data_train_soft, y_train_fold))
                fold_valid_soft.append((fold_data_valid_soft, y_valid_fold))
            else:
                fold_train_reg.append((fold_data_train_reg, y_train_fold))
                fold_valid_reg.append((fold_data_valid_reg, y_valid_fold))

        valid_estimators = [name for name in estimator_names if name in valid_estimators]
        if len(valid_estimators) == 0:
            raise RuntimeError(
                "No estimators were successfully fitted in MajorityVote.evaluate_cv."
            )

        cache = _MajorityVoteCVCache(
            task_type=self.task_type,
            estimator_names=valid_estimators,
            metrics=metrics,
            metric_names=metric_names,
            probabilities=probabilities,
            logics=logics,
            desired_performance_score=desired_performance_score,
            folds={
                'hard': (fold_train_hard, fold_valid_hard),
                'soft': (fold_train_soft, fold_valid_soft),
                'mean': (fold_train_reg, fold_valid_reg),
            },
            cv_indices=resolved_pairs,
            token=uuid4().hex,
        )
        if individual_only:
            combinations = [[name] for name in valid_estimators]
        else:
            combinations = self._cv_combinations(cache, valid_estimators, n_estimators_max)
        results = self._score_cv_combinations(cache, combinations, individual_only)
        if individual_only:
            results.attrs['majority_vote_cv_token'] = cache.token
            self._cv_cache = cache
            self.cv_indices_ = [(train.copy(), valid.copy()) for train, valid in resolved_pairs]
        self.final_results_cv = results
        return results

    @staticmethod
    def _cv_combinations(
        cache: _MajorityVoteCVCache,
        estimator_names: list[str],
        n_estimators_max: int,
    ) -> list[list[str]]:
        from mlchem.helper import generate_combination_cascade

        combinations = generate_combination_cascade(estimator_names, n_estimators_max)
        if cache.task_type == 'classification':
            combinations = [comb for comb in combinations if len(comb) % 2 != 0]
        return combinations

    def evaluate_consensus_cv(
        self,
        estimator_results: pd.DataFrame,
        n_estimators_max: int = 5,
    ) -> pd.DataFrame:
        """
        Evaluate combinations of shortlisted estimators without refitting.

        This is the second stage of ``evaluate_cv(individual_only=True)``.
        For a one-call evaluation without screening, use ``evaluate_cv``
        with its default ``individual_only=False`` instead.

        Parameters
        ----------
        estimator_results : pandas.DataFrame
            The DataFrame returned by the latest ``evaluate_cv`` call with
            ``individual_only=True``, optionally filtered to retain desired
            rows. Each retained ``estimator`` selects that model once, even
            when several metric or strategy rows remain. Filtering or editing
            score and prediction columns does not alter cached predictions.
            DataFrame attributes must be preserved when filtering or copying.
        n_estimators_max : int, optional (default=5)
            Maximum ensemble size. Classification uses only odd sizes.

        Returns
        -------
        pandas.DataFrame
            Fold metrics for shortlisted combinations, including individual
            estimators. Metrics, folds, and prediction settings are inherited
            from the individual evaluation; settings used to generate those
            folds cannot be overridden in this stage.

        Raises
        ------
        ValueError
            If no current individual evaluation exists, the table is empty,
            contains unknown estimators, or came from a different evaluation.
        TypeError
            If ``estimator_results`` is not a DataFrame.
        """
        cache = self._cv_cache
        if cache is None:
            raise ValueError("Run evaluate_cv(individual_only=True) before evaluating consensus.")
        if not isinstance(estimator_results, pd.DataFrame):
            raise TypeError("'estimator_results' must be a pandas DataFrame.")
        if estimator_results.attrs.get('majority_vote_cv_token') != cache.token:
            raise ValueError("'estimator_results' must come from the latest individual evaluation.")
        if 'estimator' not in estimator_results.columns:
            raise ValueError("'estimator_results' must contain the 'estimator' column.")
        selected = estimator_results['estimator'].tolist()
        if not selected:
            raise ValueError("'estimator_results' must contain at least one estimator.")
        if any(not isinstance(name, str) or name not in cache.estimator_names for name in selected):
            raise ValueError("'estimator_results' contains an unknown estimator.")
        selected_names = set(selected)
        names = [name for name in cache.estimator_names if name in selected_names]
        combinations = self._cv_combinations(cache, names, n_estimators_max)
        self.final_results_cv = self._score_cv_combinations(cache, combinations)
        return self.final_results_cv

    def _score_cv_combinations(
        self,
        cache: _MajorityVoteCVCache,
        combinations: list[list[str]],
        individual_only: bool = False,
    ) -> pd.DataFrame:
        from mlchem.metrics import get_reliability_score_components

        metrics = cache.metrics
        metric_names = cache.metric_names
        probabilities = cache.probabilities
        logics = cache.logics
        rows_by_metric = [[] for _ in metrics]

        for combination in tqdm(
            combinations, desc="Evaluating combinations", disable=False
        ):
            if cache.task_type == 'classification':
                strategies = ('hard', 'soft')
            else:
                strategies = ('mean',)

            for strategy in strategies:
                train_scores = [[] for _ in metrics]
                validation_scores = [[] for _ in metrics]
                train_predictions = [[] for _ in metrics]
                validation_predictions = [[] for _ in metrics]

                fold_train, fold_valid = cache.folds[strategy]
                for fold_idx in range(len(cache.cv_indices)):
                    train_dict, y_train_fold = fold_train[fold_idx]
                    valid_dict, y_valid_fold = fold_valid[fold_idx]

                    if any(name not in train_dict for name in combination):
                        train_scores = [[] for _ in metrics]
                        break

                    train_matrix = np.column_stack([train_dict[name] for name in combination])
                    valid_matrix = np.column_stack([valid_dict[name] for name in combination])

                    if strategy == 'hard':
                        y_train_pred = self._hard_vote(train_matrix)
                        y_valid_pred = self._hard_vote(valid_matrix)
                    elif strategy == 'soft':
                        y_train_pred = train_matrix.mean(axis=1)
                        y_valid_pred = valid_matrix.mean(axis=1)
                        if not all(probabilities):
                            y_train_rounded = self._soft_vote(train_matrix)
                            y_valid_rounded = self._soft_vote(valid_matrix)
                    else:
                        y_train_pred = self._mean_vote(train_matrix)
                        y_valid_pred = self._mean_vote(valid_matrix)

                    for metric_idx, metric_function in enumerate(metrics):
                        metric_train_pred = y_train_pred
                        metric_valid_pred = y_valid_pred
                        if strategy == 'soft' and not probabilities[metric_idx]:
                            metric_train_pred = y_train_rounded
                            metric_valid_pred = y_valid_rounded
                        train_scores[metric_idx].append(
                            float(metric_function(y_train_fold, metric_train_pred))
                        )
                        validation_scores[metric_idx].append(
                            float(metric_function(y_valid_fold, metric_valid_pred))
                        )
                        if individual_only:
                            train_predictions[metric_idx].append(metric_train_pred.copy())
                            validation_predictions[metric_idx].append(metric_valid_pred.copy())

                for metric_idx, name in enumerate(metric_names):
                    if not train_scores[metric_idx]:
                        continue

                    reliability_folds = []
                    for train_score, validation_score in zip(
                        train_scores[metric_idx], validation_scores[metric_idx]
                    ):
                        comp = get_reliability_score_components(
                            train_score=train_score,
                            cv_score=validation_score,
                            logic=logics[metric_idx],
                            desired_performance_score=cache.desired_performance_score,
                        )
                        reliability_folds.append(float(comp['reliability_score']))

                    train_mean, train_se = self._mean_and_se(train_scores[metric_idx])
                    validation_mean, validation_se = self._mean_and_se(validation_scores[metric_idx])
                    reliability_mean, reliability_se = self._mean_and_se(reliability_folds)

                    if cache.task_type == 'classification':
                        comb_label = f"{combination}_{strategy}_{name}"
                    else:
                        comb_label = f"{combination}_{name}"

                    row = {
                        'combination': comb_label,
                        'train_mean': train_mean,
                        'train_se': train_se,
                        'train_min': float(np.min(train_scores[metric_idx])),
                        'train_max': float(np.max(train_scores[metric_idx])),
                        'validation_mean': validation_mean,
                        'validation_se': validation_se,
                        'validation_min': float(np.min(validation_scores[metric_idx])),
                        'validation_max': float(np.max(validation_scores[metric_idx])),
                        'reliability_mean': reliability_mean,
                        'reliability_se': reliability_se,
                        'reliability_min': float(np.min(reliability_folds)),
                        'reliability_max': float(np.max(reliability_folds)),
                        'train_folds': np.asarray(train_scores[metric_idx]),
                        'validation_folds': np.asarray(validation_scores[metric_idx]),
                        'reliability_folds': np.asarray(reliability_folds),
                    }
                    if len(metrics) > 1:
                        row['metric'] = name
                    if individual_only:
                        row['estimator'] = combination[0]
                        row['strategy'] = strategy
                        row['train_predictions_folds'] = train_predictions[metric_idx]
                        row['validation_predictions_folds'] = validation_predictions[metric_idx]
                    rows_by_metric[metric_idx].append(row)

        if not any(rows_by_metric):
            raise RuntimeError(
                "No ensemble combinations were successfully evaluated in "
                "MajorityVote.evaluate_cv."
            )

        return pd.concat(
            [
                pd.DataFrame(rows).sort_values(by='reliability_mean', ascending=False)
                for rows in rows_by_metric
            ],
            ignore_index=True,
        )


class ApplicabilityDomain:
    """
    A class to calculate the leverage of data points in a dataset for
    applicability domain analysis.

    The leverage is a measure of the influence of a data point in a
    regression model. It helps identify
    data points that have a significant impact on the model's
    predictions. This class provides a method to calculate the leverage
    values for a given dataset and determine whether each data point is
    within the applicability domain based on a threshold.

    Methods:
    ---------
    leverage(X: np.ndarray):
        Calculates the leverage values for the 
        given dataset and determines whether each data point is within the
        applicability domain based on a threshold.

    """

    @staticmethod
    def leverage(X: np.ndarray) -> dict[str, list[float] | list[bool] | float]:
        """
Calculate leverage values for a dataset and determine applicability domain.

Parameters
----------
X : numpy.ndarray
    Feature matrix of shape (n_samples, n_features).

Returns
-------
dict of str to list or float
    Dictionary containing:

    - ``'leverages'`` (list of float): leverage values for each data
      point.
    - ``'results'`` (list of bool): boolean flags indicating whether each
      point is within the domain.
    - ``'threshold'`` (float): threshold used to determine domain
      inclusion.

Examples
--------
Calculate leverage for four rows and two features:

>>> import numpy as np
>>> X = np.array([
...     [1.0, 0.0],
...     [1.0, 1.0],
...     [1.0, 2.0],
...     [1.0, 3.0],
... ])
>>> result = ApplicabilityDomain.leverage(X)
>>> result["threshold"]
1.5
>>> len(result["leverages"])
4
"""

        threshold = 3 * X.shape[1] / X.shape[0]
        # Precompute the inverse of X.T @ X
        b = np.linalg.pinv(np.dot(X.T, X))

        # Calculate leverages using matrix operations
        leverages = np.einsum('ij,jk,ik->i', X, b, X)

        dict_results = {
            'leverages': leverages.tolist(),
            'results': [lev < threshold for lev in leverages],
            'threshold': threshold
        }
        return dict_results
