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

from typing import Literal, Callable, Iterable
import pandas as pd
import numpy as np
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from mlchem.helper import coerce_log_level, validate_task_type, resolve_n_jobs, disable_estimator_parallelization


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
        (e.g. for stratified splitters).

    cv_iter : int, optional (default=5)
        Number of folds to generate when neither ``cv_splitter`` nor
        ``cv_indices`` is supplied. Mirrors the legacy behaviour.

    cv_splitter : object, optional
        A scikit-learn compatible cross-validation splitter (e.g.
        ``GroupKFold``, ``StratifiedGroupKFold``, ``PredefinedSplit``)
        exposing a ``split(X, y, groups)`` method.

    cv_indices : iterable of (array-like, array-like), optional
        Explicit, pre-computed ``(train_idx, valid_idx)`` pairs. Takes
        precedence over ``cv_splitter`` and ``cv_iter`` when provided.
        This is the most generic API and is suitable for externally
        generated fold manifests (scaffold-based folds, UMAP-cluster
        folds, etc.).

    groups : array-like, optional
        Group labels used by group-aware splitters such as
        ``GroupKFold`` or ``StratifiedGroupKFold``. Ignored unless
        ``cv_splitter`` is provided and supports grouping.

    task_type : {'classification', 'regression'}, optional (default='classification')
        Used only to pick the default splitter (``StratifiedKFold`` or
        ``KFold``) when neither ``cv_splitter`` nor ``cv_indices`` is given.

    shuffle : bool, optional (default=False)
        Whether to shuffle samples before splitting, for the default
        ``cv_iter``-based splitter.

    random_state : int or None, optional (default=None)
        Random seed for the default ``cv_iter``-based splitter when
        ``shuffle=True``.

    strict_coverage : bool, optional (default=False)
        Forwarded to :func:`validate_cv_indices`.

    Returns
    -------
    list of (numpy.ndarray, numpy.ndarray)
        Validated ``(train_idx, valid_idx)`` pairs.
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
"""

    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer

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
                                scoring=make_scorer(metric),
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
                                    scoring=make_scorer(metric),
                                    return_train_score=True,)
        else:
            from sklearn.model_selection import KFold
            result = cross_validate(estimator,
                                    X,
                                    y,
                                    cv=KFold(**cv_kwargs),
                                    scoring=make_scorer(metric),
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
                 ) -> dict[str, float | np.ndarray]:
    """
Perform y-scrambling to assess model performance due to chance.

This function evaluates the robustness of a model by randomly shuffling
the target variable multiple times and measuring performance on the validation set.
It compares the distribution of scores from scrambled targets to the actual
model performance. More explained at https://doi.org/10.1021/ci700157b.

Parameters
----------
estimator : object
    A scikit-learn compatible estimator.

train_set : numpy.ndarray or pandas.DataFrame
    Training feature matrix.

y_train : iterable
    Target values for training.

metric : callable
    A scoring function that accepts (y_true, y_pred) as arguments.

n_scrambles : int, optional (default=100)
    Number of random target permutations used to build the null
    performance distribution.

n_fold : int, optional
    Number of cross-validation folds. Default is 5. Ignored when
    ``cv_splitter`` or ``cv_indices`` is provided.

cv_splitter : object, optional
    Cross-validation splitter. If provided, it overrides ``n_fold``.

groups : array-like, optional
    Group labels forwarded to group-aware splitters.

cv_indices : iterable or None, optional
    Predefined cross-validation index pairs. If provided, it overrides
    ``n_fold`` and ``cv_splitter``.

logic : {'lower', 'greater'}, optional
    Logic to determine if a score is better. 'greater' means higher is better, 'lower' means lower is better.

task_type : {'classification', 'regression'}, optional
    Type of task. Determines the default behavior of certain metrics.

plot : bool, optional (default=True)
    Whether to display a histogram of the scrambled scores.

n_jobs : int, optional (default=1)
    Number of parallel workers for iterations. -1 uses all available CPUs.

safety_multiplier : float, optional (default=2.3)
    Multiplier applied to the standard deviation of scrambled scores to
    define the absolute safety band:
    ``safety_margin = safety_multiplier * scrambled_std``.
    Use this parameter to apply a custom z-score-like thresholding
    convention.

log_level : int or str, optional (default=logging.INFO)
    Logging level for diagnostics.

Returns
-------
dict
        Dictionary with:
        - ``reference_score``: float, CV score on true labels.
        - ``scrambled_scores``: numpy.ndarray, CV scores on scrambled labels.
        - ``scrambled_std``: float, standard deviation of scrambled scores.
        - ``best_random_score``: float, best scrambled score according to
            ``logic`` (max for ``'greater'``, min for ``'lower'``).
        - ``probability_better``: float, empirical probability that scrambled
            performance is at least as good as reference (or at most as good for
            ``'lower'``).
        - ``obtained_margin``: float, gap between reference score and
            ``best_random_score`` in the direction of improvement.
        - ``safety_margin``: float, absolute safety band width
            (``safety_multiplier * scrambled_std``).
        - ``safety_margin_ratio``: float, ``obtained_margin / safety_margin``.
        - ``safety_multiplier``: float, multiplier used to define
            ``safety_margin``.
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
            estimator_copy,
            X_train,
            y_shuffled,
            metric,
            n_fold=n_fold,
            task_type=task_type,
            cv_splitter=cv_splitter,
            cv_indices=cv_indices,
            groups=groups,
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

class MajorityVote:
    """
MajorityVote ensemble evaluation utility.

This class supports two workflows:
1) Recommended workflow: ``evaluate_cv()`` using fold-level
    cross-validation, fold-level reliability, and reliability-based
    ranking.
2) Final consensus workflow: ``fit()`` to refit the selected estimator
    set on the full training set and generate transparent train/test
    prediction tables and consensus outputs.

Parameters
----------
train_set : pandas.DataFrame
    Training feature matrix.

y_train : iterable
    Training labels/targets.

task_type : {'classification', 'regression'}
    Task type used to select the voting strategy.

estimator_list : list
    List of scikit-learn compatible estimators.

column_list : list[list[str]]
    Per-estimator feature subsets aligned with ``estimator_list``.

test_set : pandas.DataFrame or None, optional
    Hold-out feature matrix used only by ``fit()``.

y_test : iterable or None, optional
    Hold-out labels/targets used only by ``fit()``.

estimator_names : list[str] or None, optional
    Optional user-defined names for estimators.

n_jobs : int, optional (default=1)
    Number of workers used where parallel execution is supported.

log_level : int or str, optional (default=logging.INFO)
    Logging level used by this module.
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
        """
        if self.test_set is None or self.y_test is None:
            raise ValueError(
                "MajorityVote.fit requires 'test_set' and 'y_test'. "
                "Use evaluate_cv() for cross-validation-centric evaluation."
            )

        selected_estimators_resolved = self._resolve_selected_estimators(selected_estimators)
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
                    y_train_soft = estimator_copy.predict_proba(X_train)[:, 1]
                    y_test_soft = estimator_copy.predict_proba(X_test)[:, 1]
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
                self.df_train_predictions_hard[estimator_name] = y_train_hard
                self.df_test_predictions_hard[estimator_name] = y_test_hard
                self.df_train_predictions_soft[estimator_name] = y_train_soft
                self.df_test_predictions_soft[estimator_name] = y_test_soft

            failed_estimators = [
                payload[3] for payload in est_data_list
                if payload[3] not in successful_estimators
            ]

            if self.df_train_predictions_hard.shape[1] == 0:
                raise RuntimeError(
                    "No estimators were successfully fitted in "
                    "MajorityVote.fit for classification."
                )

            self.df_train_predictions_hard['Y'] = self.y_train
            self.df_test_predictions_hard['Y'] = self.y_test

            self.df_train_predictions_soft['Y'] = self.y_train
            self.df_test_predictions_soft['Y'] = self.y_test

            if build_consensus:
                estimator_cols = list(self.df_train_predictions_hard.columns[:-1])

                self.df_train_predictions = pd.DataFrame(index=self.train_set.index)
                self.df_train_predictions['Y'] = self.y_train
                self.df_test_predictions = pd.DataFrame(index=self.test_set.index)
                self.df_test_predictions['Y'] = self.y_test

                for est_name in estimator_cols:
                    self.df_train_predictions[f'{est_name}_proba'] = self.df_train_predictions_soft[est_name].values
                    self.df_train_predictions[f'{est_name}_CLASS'] = self.df_train_predictions_hard[est_name].values
                    self.df_test_predictions[f'{est_name}_proba'] = self.df_test_predictions_soft[est_name].values
                    self.df_test_predictions[f'{est_name}_CLASS'] = self.df_test_predictions_hard[est_name].values

                hard_train_matrix = self.df_train_predictions_hard[estimator_cols].to_numpy(copy=False)
                hard_test_matrix = self.df_test_predictions_hard[estimator_cols].to_numpy(copy=False)
                soft_train_matrix = self.df_train_predictions_soft[estimator_cols].to_numpy(copy=False)
                soft_test_matrix = self.df_test_predictions_soft[estimator_cols].to_numpy(copy=False)

                self.df_train_predictions['CONSENSUS_hard'] = self._hard_vote(hard_train_matrix)
                self.df_test_predictions['CONSENSUS_hard'] = self._hard_vote(hard_test_matrix)
                self.df_train_predictions['CONSENSUS_soft'] = self._soft_vote(soft_train_matrix)
                self.df_test_predictions['CONSENSUS_soft'] = self._soft_vote(soft_test_matrix)

        else:
            self.df_train_predictions = pd.DataFrame(index=self.train_set.index)
            self.df_test_predictions = pd.DataFrame(index=self.test_set.index)

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
                self.df_train_predictions[estimator_name] = y_train_pred
                self.df_test_predictions[estimator_name] = y_test_pred

            failed_estimators = [
                payload[3] for payload in est_data_list
                if payload[3] not in successful_estimators
            ]

            if self.df_train_predictions.shape[1] == 0:
                raise RuntimeError(
                    "No estimators were successfully fitted in "
                    "MajorityVote.fit for regression."
                )

            self.df_train_predictions['Y'] = self.y_train
            self.df_test_predictions['Y'] = self.y_test

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
        metric: Callable,
        metric_name: str,
        cv_iter: int = 5,
        cv_splitter=None,
        cv_indices: Iterable | None = None,
        groups: np.ndarray | pd.Series | None = None,
        n_estimators_max: int = 5,
        logic: Literal['lower', 'greater'] = 'greater',
        desired_performance_score: Literal['train', 'cv', 'train_cv_average'] = 'train_cv_average',
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """
Evaluate ensemble combinations using fold-level cross-validation.

The same explicit fold definitions are reused for every estimator and
combination, ensuring fair fold-wise comparison.

Parameters
----------
metric : callable
    Scoring function accepting ``(y_true, y_pred)``.

metric_name : str
    Label appended to generated combination identifiers.

cv_iter : int, optional (default=5)
    Number of folds used when ``cv_indices`` and ``cv_splitter`` are not
    supplied.

cv_splitter : object, optional
    Scikit-learn compatible splitter exposing ``split(X, y, groups)``.

cv_indices : iterable of (train_idx, valid_idx), optional
    Explicit precomputed fold manifest. When supplied, all combinations
    are evaluated on exactly these folds.

groups : array-like, optional
    Group labels forwarded to group-aware ``cv_splitter`` objects.

n_estimators_max : int, optional (default=5)
    Maximum combination size passed to ``generate_combination_cascade``.

logic : {'lower', 'greater'}, optional (default='greater')
    Direction of optimization for reliability computation.

desired_performance_score : {'train', 'cv', 'train_cv_average'}, optional
    Performance component used inside
    ``get_reliability_score_components``.

shuffle : bool, optional (default=False)
    Whether generated CV folds are shuffled when using ``cv_iter``.

random_state : int or None, optional
    Random seed used with generated shuffled folds.

Returns
-------
pandas.DataFrame
    One row per evaluated ensemble variant. Columns include:
    ``combination``, ``train_mean``, ``train_se``,
    ``validation_mean``, ``validation_se``,
    ``reliability_mean``, ``reliability_se``,
    ``train_folds``, ``validation_folds``, ``reliability_folds``.
    Results are ranked by ``reliability_mean`` descending.
"""

        from sklearn.base import clone
        from mlchem.helper import generate_combination_cascade
        from mlchem.metrics import get_reliability_score_components

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

        for train_idx, valid_idx in resolved_pairs:
            y_train_fold = y_all[train_idx]
            y_valid_fold = y_all[valid_idx]

            fold_data_train_hard = {}
            fold_data_valid_hard = {}
            fold_data_train_soft = {}
            fold_data_valid_soft = {}
            fold_data_train_reg = {}
            fold_data_valid_reg = {}

            for i, (estimator, columns) in enumerate(zip(self.estimator_list, self.column_list)):
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
                        fold_data_train_soft[estimator_name] = np.asarray(est.predict_proba(X_train_fold)[:, 1])
                        fold_data_valid_soft[estimator_name] = np.asarray(est.predict_proba(X_valid_fold)[:, 1])
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

        combinations = generate_combination_cascade(valid_estimators, n_estimators_max)
        if self.task_type == 'classification':
            combinations = [comb for comb in combinations if len(comb) % 2 != 0]

        rows = []

        for combination in combinations:
            if self.task_type == 'classification':
                strategies = ('hard', 'soft')
            else:
                strategies = ('mean',)

            for strategy in strategies:
                train_scores = []
                validation_scores = []

                for fold_idx in range(len(resolved_pairs)):
                    if self.task_type == 'classification':
                        train_dict, y_train_fold = fold_train_hard[fold_idx]
                        valid_dict, y_valid_fold = fold_valid_hard[fold_idx]
                        if strategy == 'soft':
                            train_dict, y_train_fold = fold_train_soft[fold_idx]
                            valid_dict, y_valid_fold = fold_valid_soft[fold_idx]
                    else:
                        train_dict, y_train_fold = fold_train_reg[fold_idx]
                        valid_dict, y_valid_fold = fold_valid_reg[fold_idx]

                    if any(name not in train_dict for name in combination):
                        train_scores = []
                        validation_scores = []
                        break

                    train_matrix = np.column_stack([train_dict[name] for name in combination])
                    valid_matrix = np.column_stack([valid_dict[name] for name in combination])

                    if strategy == 'hard':
                        y_train_pred = self._hard_vote(train_matrix)
                        y_valid_pred = self._hard_vote(valid_matrix)
                    elif strategy == 'soft':
                        y_train_pred = self._soft_vote(train_matrix)
                        y_valid_pred = self._soft_vote(valid_matrix)
                    else:
                        y_train_pred = self._mean_vote(train_matrix)
                        y_valid_pred = self._mean_vote(valid_matrix)

                    train_scores.append(float(metric(y_train_fold, y_train_pred)))
                    validation_scores.append(float(metric(y_valid_fold, y_valid_pred)))

                if len(train_scores) == 0:
                    continue

                reliability_folds = []
                for train_score, validation_score in zip(train_scores, validation_scores):
                    comp = get_reliability_score_components(
                        train_score=train_score,
                        cv_score=validation_score,
                        logic=logic,
                        desired_performance_score=desired_performance_score,
                    )
                    reliability_folds.append(float(comp['reliability_score']))

                train_mean, train_se = self._mean_and_se(train_scores)
                validation_mean, validation_se = self._mean_and_se(validation_scores)
                reliability_mean, reliability_se = self._mean_and_se(reliability_folds)

                if self.task_type == 'classification':
                    comb_label = f"{combination}_{strategy}_{metric_name}"
                else:
                    comb_label = f"{combination}_{metric_name}"

                rows.append({
                    'combination': comb_label,
                    'train_mean': train_mean,
                    'train_se': train_se,
                    'validation_mean': validation_mean,
                    'validation_se': validation_se,
                    'reliability_mean': reliability_mean,
                    'reliability_se': reliability_se,
                    'train_folds': np.asarray(train_scores),
                    'validation_folds': np.asarray(validation_scores),
                    'reliability_folds': np.asarray(reliability_folds),
                })

        if len(rows) == 0:
            raise RuntimeError(
                "No ensemble combinations were successfully evaluated in "
                "MajorityVote.evaluate_cv."
            )

        self.final_results_cv = pd.DataFrame(rows)
        self.final_results_cv = self.final_results_cv.sort_values(
            by='reliability_mean', ascending=False
        ).reset_index(drop=True)
        return self.final_results_cv


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
