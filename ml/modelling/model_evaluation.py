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
    return {
        'scores': scores,
        'mean': np.mean(scores),
        'stderr': np.std(scores) / np.sqrt(len(scores))
    }

def crossval(estimator,
             X: np.ndarray | pd.DataFrame,
             y: np.ndarray | pd.DataFrame,
             metric_function: Callable,
             n_fold: int = 5,
             task_type: Literal['classification',
                                'regression'] = 'classification',
             random_state: int | None = None,
             shuffle: bool = False,
             cv_splitter=None,
             cv_indices: Iterable | None = None,
             groups: np.ndarray | pd.Series | None = None,
             ) -> np.ndarray:
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

metric_function : callable
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
numpy.ndarray
    An array of train and cross-validation scores.
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
                                scoring=make_scorer(metric_function),
                                groups=resolved_pairs,
                                return_train_score=True,)

    # ---- Legacy path: unchanged behaviour for cv_iter=n_fold. ----
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
                                scoring=make_scorer(metric_function),
                                return_train_score=True,)
    else:
        from sklearn.model_selection import KFold
    result = cross_validate(estimator,
                            X,
                            y,
                            cv=KFold(**cv_kwargs),
                            scoring=make_scorer(metric_function),
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
                 cv_iter: int = 5,
                 cv_splitter=None,
                 groups=None,
                 cv_indices: Iterable | None = None,
                 logic: Literal['lower', 'greater'] = 'greater',
                 task_type: Literal[
                     'classification', 'regression'] = 'classification',
                 desired_performance_score: Literal['train','cv','train_cv_average'] = 'train_cv_average',
                 plot: bool = True,
                 n_jobs: int = 1,
                 log_level: int | str | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = logging.INFO,
                 ) -> None:
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

cv_iter : int, optional
    Number of cross-validation iterations. Default is 5. Ignored when
    ``cv_splitter`` or ``cv_indices`` is provided.

cv_splitter : object, optional
    Cross-validation splitter. If provided, it overrides ``cv_iter``.

groups : array-like, optional
    Group labels for the samples used while splitting the dataset into train/test set.

cv_indices : iterable or None, optional
    Predefined cross-validation indices. If provided, it overrides ``cv_iter`` and ``cv_splitter``.

logic : {'lower', 'greater'}, optional
    Logic to determine if a score is better. 'greater' means higher is better, 'lower' means lower is better.

task_type : {'classification', 'regression'}, optional
    Type of task. Determines the default behavior of certain metrics.

desired_performance_score : {'train','cv','train_cv_average'}, optional
    Which performance score to prioritize when evaluating reliability.

plot : bool, optional (default=True)
    Whether to display a histogram of the scrambled scores.

n_jobs : int, optional (default=1)
    Number of parallel workers for iterations. -1 uses all available CPUs.

log_level : int or str, optional (default=logging.INFO)
    Logging level for diagnostics.

Returns
-------
None
"""

    from sklearn.base import clone
    import random

    resolved_log_level = coerce_log_level(log_level)
    _configure_module_logging(resolved_log_level)
    n_jobs = resolve_n_jobs(n_jobs)

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
        cv_iter,
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
            cv_iter,
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
    ys_std = np.std(scores)

    if logic == 'greater':
        value = len(scores[scores >= (ref_score)])/len(scores)
        obtained_margin = max(ref_score-ys_max, 0)
    else: # logic == 'lower'
        value = len(scores[scores <= (ref_score)])/len(scores)
        obtained_margin = max(ys_max-ref_score, 0)

    # Rucker et al, https://doi.org/10.1021/ci700157b
    safety_margin = 2.3 * ys_std

    logger.log(
        resolved_log_level,
        f'Probability to obtain a better model by chance: {value:.3f}')
    logger.log(
        resolved_log_level,
        f'Safety margin: {obtained_margin / (2.3 * ys_std):.2f}')

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.histplot(scores)
        plt.axvline(ref_score,
                    color='red',
                    linestyle='--')
        plt.axvspan(ys_max,
                    ys_max + safety_margin,
                    color='red',
                    alpha=0.1)
        plt.show()

    return {'reference_score': ref_score,
            'scrambled_scores': scores,
            'probability_better': value,
            'obtained_margin': obtained_margin,
            'safety_margin': safety_margin
            }

class MajorityVote:
    """
MajorityVote(train_set, test_set, y_train, y_test, task_type, 
estimator_list, column_list, estimator_names=None, n_jobs=1, log_level=logging.INFO)

Ensemble model using majority voting (for classification) or averaging (for regression).

This class combines predictions from multiple estimators to 
improve model performance and robustness by leveraging the strengths of
different models. Supports parallel execution of estimator fitting and evaluation.

Parameters
----------
train_set : pandas.DataFrame
    The training dataset.

test_set : pandas.DataFrame
    The testing dataset.

y_train : iterable
    Target values for the training dataset.

y_test : iterable
    Target values for the testing dataset.

task_type : {'classification', 'regression'}
    The type of task to perform.

estimator_list : list
    A list of fitted scikit-learn estimators.

column_list : list of str
    A list of feature columns for each estimator.

estimator_names : list of str, optional
    A list of names for the estimators. Defaults to an empty list.

n_jobs : int, optional (default=1)
    Number of parallel workers for fitting and evaluation. -1 uses all available CPUs.

log_level : int or str, optional (default=logging.INFO)
    Logging level for diagnostics.
"""

    def __init__(
        self,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame,
        y_train: Iterable,
        y_test: Iterable,
        task_type: Literal['classification', 'regression'],
        estimator_list: list,
        column_list: list[str],
        estimator_names: list[str] | None = None,
        n_jobs: int = 1,
        log_level: int | str | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = logging.INFO
         ) -> None:

        self.task_type = validate_task_type(task_type)
        self.estimator_list = estimator_list
        self.estimator_names = [] if estimator_names is None else list(estimator_names)
        self.column_list = column_list
        self.train_set = train_set
        self.test_set = test_set
        self.y_train = y_train
        self.y_test = y_test
        self.n_jobs = resolve_n_jobs(n_jobs)
        self.log_level = coerce_log_level(log_level)
        _configure_module_logging(self.log_level)

    def fit(self) -> None:
        """
Fit the estimators on the training data and store predictions.

For classification tasks, both hard (class labels) and soft (probabilities)
predictions are stored. For regression tasks, predicted values are stored.
Supports parallel execution via n_jobs parameter.

Returns
-------
None
"""

        def fit_and_predict(est_data):
            """Fit a single estimator and return predictions."""
            i, estimator, columns = est_data
            X_train = self.train_set[columns]
            X_train = X_train.loc[:, ~X_train.columns.
                                  duplicated(keep='first')].copy()
            X_test = self.test_set[columns]
            X_test = X_test.loc[:, ~X_test.columns.
                                duplicated(keep='first')].copy()
            if len(self.estimator_names) > 0:
                estimator_name = self.estimator_names[i]
            else:
                estimator_name = str(estimator)
            
            try:
                disable_estimator_parallelization(estimator)  # Prevent nested parallelization warnings
                estimator.fit(X_train, self.y_train)
                if self.task_type == 'classification':
                    y_train_hard = estimator.predict(X_train)
                    y_test_hard = estimator.predict(X_test)
                    y_train_soft = estimator.predict_proba(X_train)[:, 1]
                    y_test_soft = estimator.predict_proba(X_test)[:, 1]
                    return (estimator_name, 'classification',
                            y_train_hard, y_test_hard, y_train_soft, y_test_soft)
                else:  # regression
                    y_train_pred = estimator.predict(X_train)
                    y_test_pred = estimator.predict(X_test)
                    return (estimator_name, 'regression',
                            y_train_pred, y_test_pred)
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
            self.df_train_predictions_hard = pd.\
                DataFrame(index=self.train_set.index)
            self.df_test_predictions_hard = pd.\
                DataFrame(index=self.test_set.index)

            self.df_train_predictions_soft = pd.\
                DataFrame(index=self.train_set.index)
            self.df_test_predictions_soft = pd.\
                DataFrame(index=self.test_set.index)

            # Parallel execution
            est_data_list = list(enumerate(
                zip(self.estimator_list, self.column_list)
            ))
            est_data_list = [(i, est, cols) for i, (est, cols) in est_data_list]
            
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
                estimator_name, task_type, y_train_hard, y_test_hard, y_train_soft, y_test_soft = result
                self.df_train_predictions_hard[estimator_name] = y_train_hard
                self.df_test_predictions_hard[estimator_name] = y_test_hard
                self.df_train_predictions_soft[estimator_name] = y_train_soft
                self.df_test_predictions_soft[estimator_name] = y_test_soft

            if self.df_train_predictions_hard.shape[1] == 0:
                raise RuntimeError(
                    "No estimators were successfully fitted in "
                    "MajorityVote.fit for classification."
                )

            self.df_train_predictions_hard['Y'] = self.y_train
            self.df_test_predictions_hard['Y'] = self.y_test

            self.df_train_predictions_soft['Y'] = self.y_train
            self.df_test_predictions_soft['Y'] = self.y_test

        else:     # if regression
            self.df_train_predictions = pd.DataFrame(index=self.train_set.index)
            self.df_test_predictions = pd.DataFrame(index=self.test_set.index)

            # Parallel execution
            est_data_list = list(enumerate(
                zip(self.estimator_list, self.column_list)
            ))
            est_data_list = [(i, est, cols) for i, (est, cols) in est_data_list]
            
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
                estimator_name, task_type, y_train_pred, y_test_pred = result
                self.df_train_predictions[estimator_name] = y_train_pred
                self.df_test_predictions[estimator_name] = y_test_pred

            if self.df_train_predictions.shape[1] == 0:
                raise RuntimeError(
                    "No estimators were successfully fitted in "
                    "MajorityVote.fit for regression."
                )

            self.df_train_predictions['Y'] = self.y_train
            self.df_test_predictions['Y'] = self.y_test

    def predict(self,
                metric,
                metric_name: str,
                n_estimators_max: int = 5) -> None:
        """
Generate ensemble predictions and evaluate performance using a 
specified metric.

For classification, both hard and soft voting are evaluated. 
For regression, predictions are averaged. Results are stored for each 
combination of estimators up to a specified maximum. Supports parallel 
evaluation via n_jobs parameter.

Parameters
----------
metric : callable
    A scoring function that takes (y_true, y_pred) as input and returns a float.

metric_name : str
    Name of the metric used for evaluation.

n_estimators_max : int, optional
    Maximum number of estimators to consider in combinations. Default is 5.

Returns
-------
None
"""

        from mlchem.helper import generate_combination_cascade
        self.n_estimators_max = n_estimators_max

        def majority_vote(dataframe,
                          individual_ys,
                          hard: bool) -> np.ndarray:
            """
            Perform majority voting or averaging on the predictions.
            """
            dataframe_probe = dataframe[individual_ys].to_numpy(copy=False)
            if hard:
                from scipy.stats import mode
                return mode(dataframe_probe, axis=1)[0]
            else:
                return np.round(dataframe_probe.mean(axis=1))

        def evaluate_combination(comb_data):
            """Evaluate a single combination and return results."""
            combination, is_classification = comb_data
            train_results = {}
            test_results = {}
            
            if is_classification:
                # Soft predictions
                key_soft = f'{combination}_soft_{metric_name}'
                train_results[key_soft] = metric(
                    self.df_train_predictions_soft.Y.values,
                    majority_vote(self.df_train_predictions_soft, combination, hard=False)
                )
                test_results[key_soft] = metric(
                    self.df_test_predictions_soft.Y.values,
                    majority_vote(self.df_test_predictions_soft, combination, hard=False)
                )
                
                # Hard predictions
                key_hard = f'{combination}_hard_{metric_name}'
                train_results[key_hard] = metric(
                    self.df_train_predictions_hard.Y.values,
                    majority_vote(self.df_train_predictions_hard, combination, hard=True)
                )
                test_results[key_hard] = metric(
                    self.df_test_predictions_hard.Y.values,
                    majority_vote(self.df_test_predictions_hard, combination, hard=True)
                )
            else:  # regression
                key = f'{combination}_{metric_name}'
                train_results[key] = metric(
                    self.df_train_predictions.Y.values,
                    majority_vote(self.df_train_predictions, combination, hard=False)
                )
                test_results[key] = metric(
                    self.df_test_predictions.Y.values,
                    majority_vote(self.df_test_predictions, combination, hard=False)
                )
            
            return {'train': train_results, 'test': test_results}

        if self.task_type == 'classification':
            self.combinations = generate_combination_cascade(
                self.df_train_predictions_hard.columns[:-1],
                self.n_estimators_max
                )
            # exclude even number of estimators for classification
            self.combinations = [x for
                                 x in
                                 self.combinations if
                                 len(x) % 2 != 0]
        else:
            self.combinations = generate_combination_cascade(
                self.df_train_predictions.columns[:-1],
                self.n_estimators_max
                )

        # Prepare combination data for parallel processing
        comb_data_list = [(comb, self.task_type == 'classification') for comb in self.combinations]

        # Parallel execution of combinations
        if self.n_jobs == 1:
            all_results = [evaluate_combination(data) for data in tqdm(comb_data_list, desc="Evaluating combinations", disable=False)]
        else:
            max_workers = self.n_jobs if self.n_jobs > 0 else None
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(evaluate_combination, data): data for data in comb_data_list}
                all_results = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating combinations", disable=False):
                    all_results.append(future.result())

        # Merge all results
        dict_results_train = {}
        dict_results_test = {}
        for result_pair in all_results:
            dict_results_train.update(result_pair['train'])
            dict_results_test.update(result_pair['test'])

        def extract_from(results, models):
            """
            Extract scores from the results dictionary for the given
            models.
            """
            return [results[model] for model in models]

        models = [a for a in
                  dict_results_train.keys()]
        self.final_results_train = \
            pd.DataFrame(index=models,
                         data=extract_from(results=dict_results_train,
                                           models=models),
                         columns=[f'{metric_name}_train'])
        self.final_results_test = \
            pd.DataFrame(index=models,
                         data=extract_from(results=dict_results_test,
                                           models=models),
                         columns=[f'{metric_name}_test'])

        self.final_results = \
            pd.DataFrame(index=[c[:-(len(metric_name) + 1)]
                                for c in self.final_results_train.index])
        self.final_results[f'{metric_name}_train'] = \
            self.final_results_train[f'{metric_name}_train'].values
        self.final_results[f'{metric_name}_test'] = \
            self.final_results_test[f'{metric_name}_test'].values


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
