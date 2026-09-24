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

import pandas as pd
import numpy as np
from typing import Literal, Iterable, Callable, Optional
from math import comb
import logging
import warnings
import contextlib
from sklearn.base import clone
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
from mlchem.helper import (
    generate_combination_cascade,
    coerce_log_level,
    resolve_n_jobs,
    validate_task_type,
)
from mlchem.metrics import calculate_reliability_components
from mlchem.ml.modelling.model_evaluation import crossval, generate_cv_indices


logger = logging.getLogger(__name__)


def _configure_wrapper_logging(level: int) -> None:
    """Configure wrapper logger to emit at the specified level."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(level)


@contextlib.contextmanager
def _suppress_parallel_delayed_warning():
    """
    Silence the benign "sklearn.utils.parallel.delayed should be used with
    sklearn.utils.parallel.Parallel..." UserWarning while candidate subsets
    are evaluated concurrently across raw worker threads.

    ``warnings.filters`` is a single, process-wide list. sklearn's own
    nested ``cross_val_score`` internally enters/exits ``warnings.catch_warnings()``
    for every fold, and when many threads do this at the same time (as
    happens here via ``ThreadPoolExecutor``), those enter/exit cycles race
    on the shared filters list. As a result ``warnings.filterwarnings('ignore', ...)``
    does not reliably suppress the warning under real concurrent load.

    Overriding ``warnings.showwarning`` instead is safe here: it is only
    ever replaced by this function, so concurrent ``catch_warnings``
    save/restore cycles from other threads just save and restore the same
    override rather than racing to replace it with something else.
    """
    previous_showwarning = warnings.showwarning

    def _filtered_showwarning(message, category, filename, lineno, file=None, line=None):
        if category is UserWarning and 'sklearn.utils.parallel.delayed' in str(message):
            return
        previous_showwarning(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = _filtered_showwarning
    try:
        yield
    finally:
        warnings.showwarning = previous_showwarning


def _clone_for_search(estimator, outer_n_jobs: int):
    """
    Clone estimator and avoid nested parallelism when the outer wrapper
    already parallelizes candidate evaluation.
    """
    estimator_copy = clone(estimator)
    if outer_n_jobs == 1:
        return estimator_copy

    if hasattr(estimator_copy, 'get_params') and hasattr(estimator_copy, 'set_params'):
        params = estimator_copy.get_params(deep=False)
        if 'n_jobs' in params and params.get('n_jobs') not in (None, 1):
            try:
                estimator_copy.set_params(n_jobs=1)
            except Exception:
                # Not all estimators with n_jobs support runtime rewrites.
                pass

    return estimator_copy


def _orient_scores_to_utility(
    scores: np.ndarray,
    logic: Literal['lower', 'greater'],
) -> np.ndarray:
    """
    Orient metric scores so that higher utility is always better.

    For 'greater' metrics (e.g., ROC-AUC, MCC, accuracy), utility = score.
    For 'lower' metrics (e.g., RMSE, loss), utility = -score to flip orientation.

    Parameters
    ----------
    scores : numpy.ndarray
        Array of metric scores (shape: (n_subsets,) or (n_subsets, n_folds)).
    logic : {'lower', 'greater'}
        Whether the metric is minimized or maximized.

    Returns
    -------
    numpy.ndarray
        Oriented scores where higher is always better.
    """
    if logic == 'greater':
        return scores
    else:  # logic == 'lower'
        return -scores


def _compute_fold_degradation(
    cv_folds_kstar: np.ndarray,
    cv_folds_k: np.ndarray,
) -> dict[str, float]:
    """
    Compute paired fold-level degradation d from subset k* to subset k.

    For each fold r, computes d[r] = u[k*,r] - u[k,r], where u represents
    utility (oriented scores). Then computes mean and standard error.

    Parameters
    ----------
    cv_folds_kstar : numpy.ndarray
        Fold-level utility scores for reference subset k* (shape: (n_folds,)).
    cv_folds_k : numpy.ndarray
        Fold-level utility scores for subset k (shape: (n_folds,)).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'degradation': numpy array of fold-level degradation scores
        - 'mean_degradation': float, mean degradation across folds
        - 'se_degradation': float, standard error of degradation
    """
    if len(cv_folds_kstar) != len(cv_folds_k):
        raise ValueError("cv_folds_kstar and cv_folds_k must have the same length.")

    degradation = cv_folds_kstar - cv_folds_k
    mean_degradation = float(np.mean(degradation))
    se_degradation = float(np.std(degradation, ddof=1) / np.sqrt(len(degradation)))

    return {
        'degradation': degradation,
        'mean_degradation': mean_degradation,
        'se_degradation': se_degradation,
    }


def _select_parsimonious_subset(
    cv_folds_kstar: np.ndarray,
    cv_folds_kstar_index: int,
    all_cv_folds: list[np.ndarray],
    parsimony_mode: Literal['best', 'tolerance', 'standard_error', 'uncertainty'],
    tolerance: float = 0.0,
    se_multiplier: float = 1.0,
) -> dict:
    """
    Select a parsimonious subset based on fold-level degradation criteria.

    This function compares all subsets k < k* against the reference k* using
    paired fold-level CV scores, selecting the smallest subset meeting the
    parsimony criterion.

    Parameters
    ----------
    cv_folds_kstar : numpy.ndarray
        Fold-level utility scores for the reference subset k*.
    cv_folds_kstar_index : int
        Index (1-based) of the reference subset k*.
    all_cv_folds : list[numpy.ndarray]
        List of fold-level utility arrays for all subsets, indexed 0 to max.
    parsimony_mode : {'best', 'tolerance', 'standard_error', 'uncertainty'}
        Selection mode:
        - 'best': return k* (existing behavior)
        - 'tolerance': select smallest k where mean(d[k]) <= tolerance
        - 'standard_error': select smallest k where mean(d[k]) <= se_multiplier * SE(d[k])
        - 'uncertainty': select smallest k where mean(d[k]) <= tolerance + se_multiplier * SE(d[k])
    tolerance : float, optional
        Tolerance threshold for 'tolerance' and 'uncertainty' modes. Default 0.0.
    se_multiplier : float, optional
        Multiplier for standard error in 'standard_error' and 'uncertainty' modes. Default 1.0.

    Returns
    -------
    dict
        Dictionary containing:
        - 'selected_index': int, 1-based index of selected subset
        - 'parsimony_mode': str
        - 'degradation_info': dict with degradation stats for each subset
        - 'acceptable_subsets': list of 1-based indices meeting criteria
        - 'reference_index': int, the k* reference index
    """
    if parsimony_mode == 'best':
        return {
            'selected_index': cv_folds_kstar_index,
            'parsimony_mode': 'best',
            'degradation_info': {},
            'acceptable_subsets': [cv_folds_kstar_index],
            'reference_index': cv_folds_kstar_index,
        }

    degradation_info = {}
    acceptable_subsets = []

    # Evaluate subsets k from 1 to k*-1
    for k_idx in range(1, cv_folds_kstar_index):
        if k_idx >= len(all_cv_folds):
            break  # Safety check

        cv_folds_k = all_cv_folds[k_idx]
        deg_result = _compute_fold_degradation(cv_folds_kstar, cv_folds_k)
        mean_deg = deg_result['mean_degradation']
        se_deg = deg_result['se_degradation']

        degradation_info[k_idx] = {
            'mean_degradation': mean_deg,
            'se_degradation': se_deg,
        }

        # Determine if this subset meets the criterion
        criterion_met = False
        if parsimony_mode == 'tolerance':
            criterion_met = mean_deg <= tolerance
        elif parsimony_mode == 'standard_error':
            criterion_met = mean_deg <= se_multiplier * se_deg
        elif parsimony_mode == 'uncertainty':
            criterion_met = mean_deg <= (tolerance + se_multiplier * se_deg)

        if criterion_met:
            acceptable_subsets.append(k_idx)

    # Select the smallest acceptable subset, or fall back to k* if none qualify
    if acceptable_subsets:
        selected_index = acceptable_subsets[0]  # Smallest
    else:
        selected_index = cv_folds_kstar_index  # Fallback to k*

    return {
        'selected_index': selected_index,
        'parsimony_mode': parsimony_mode,
        'degradation_info': degradation_info,
        'acceptable_subsets': acceptable_subsets,
        'reference_index': cv_folds_kstar_index,
        'tolerance': tolerance,
        'se_multiplier': se_multiplier,
    }


def _resolve_wrapper_cv_indices(
    X,
    y,
    cv_iter: int,
    cv_splitter,
    cv_indices,
    groups,
    task_type: str,
):
    """
    Materialise explicit (train_idx, valid_idx) folds for a wrapper.

    Returns None when neither `cv_splitter` nor `cv_indices` is supplied,
    signalling that legacy `cv_iter`-only behaviour should be used
    unchanged. Otherwise, folds are resolved once (via
    `generate_cv_indices`) so every candidate feature/subset is evaluated
    against exactly the same folds.
    """
    if cv_splitter is None and cv_indices is None:
        return None

    return generate_cv_indices(
        X,
        y=y,
        cv_iter=cv_iter,
        cv_splitter=cv_splitter,
        cv_indices=cv_indices,
        groups=groups,
        task_type=task_type,
    )


def _safe_abs_corr(x: np.ndarray, y: np.ndarray, method: str = 'pearson') -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    if method == 'pearson':
        corr = np.corrcoef(x, y)[0, 1]
    elif method == 'spearman':
        corr = pd.Series(x).corr(pd.Series(y), method='spearman')
    else:
        raise ValueError("'method' must be either 'pearson' or 'spearman'.")

    if np.isnan(corr):
        return 0.0
    return float(abs(corr))


def _add_reliability_columns(
    dataframe: pd.DataFrame,
    logic: Literal['lower', 'greater'],
    selection_strategy: Literal['legacy', 'cv_only'] = 'legacy',
) -> pd.DataFrame:
    score_columns = [
        'geometric_mean',
        'performance_score',
        'instability_score',
        'reliability_score',
    ]
    if dataframe.empty:
        for column in score_columns:
            dataframe[column] = pd.Series(dtype=float)
        return dataframe

    reliability_scores = dataframe.apply(
        lambda row: calculate_reliability_components(
            train_score=row.training_score,
            cv_score=row.cv_score,
            test_score=None if selection_strategy == 'cv_only' else row.test_score,
            logic=logic,
        ),
        axis=1,
        result_type='expand',
    )
    return pd.concat([dataframe, reliability_scores], axis=1)


class SequentialForwardSelection:
    """
  Sequential Forward Feature Selection wrapper.

  This class performs Sequential Forward Feature Selection by iteratively
  adding features that yield the highest gain in cross-validation score.
  Best feature set can be selected:
  - calculating a helper metric that takes train/cv(/test) score instability
  into account
  - through the application of the parsimony principle (read further for its
  scientific rationale)
  - via a combination of both approaches.
  

  Public Methods
  ----------
  `__init__()`: Initialises the SequentialForwardSelection class.

  `set_log_level()`: Set the logging level for wrapper diagnostics.

  `fit()`: Fit the Sequential Forward Selection model.

  `find_best()`: Find the best feature subset based on reliability score,
  optionally applying a parsimony rule. Read the method's docstring for
  more details

  `plot()`: Plot the performance of the Sequential Forward Selection process.

Attributes
  ----------
  estimator : object
      The scikit-learn estimator used for feature selection.
  estimator_string : str, optional
      A string representation of the estimator. If None, it is inferred from the estimator.
  metric : callable
      A function to evaluate model performance.
  max_features : int, optional
      Maximum number of features to select. Default is 25.
  cv_iter : int, optional
      Number of cross-validation iterations. Default is 5. Ignored when
      ``cv_splitter`` or ``cv_indices`` is provided.
  cv_splitter : object, optional
      A scikit-learn compatible cross-validation splitter (e.g.
      ``GroupKFold(5)``, ``StratifiedGroupKFold(...)``,
      ``PredefinedSplit(...)``). Combine with ``groups`` for group-aware
      splitters. Takes precedence over ``cv_iter``.
  groups : array-like, optional
      Group labels (e.g. scaffold IDs) propagated to ``cv_splitter``.
      Ignored when ``cv_indices`` is provided.
  cv_indices : iterable of (array-like, array-like), optional
      Explicit, pre-computed ``(train_idx, valid_idx)`` pairs, e.g. a
      scaffold-based or UMAP-cluster-based fold manifest generated
      outside mlchem. Takes precedence over both ``cv_splitter`` and
      ``cv_iter``. This is the most generic API.
  logic : {'lower', 'greater'}, optional
      Whether to minimize or maximize the cross-validation score. Default is 'greater'.
  task_type : {'classification', 'regression'}, optional
      Type of task. Default is 'classification'.
  selection_strategy : {'legacy', 'cv_only'}, optional
      Strategy used by :meth:`find_best` to select the winning feature
      subset. ``'legacy'`` (default, kept for backward compatibility)
      uses train/CV/test scores, exactly as in previous releases.
      ``'cv_only'`` is a leakage-free mode that selects using only
      train/CV information; test scores are still computed and stored
      for reporting, but never influence selection. ``'cv_only'`` is the
      recommended choice for new projects.
  parsimony_mode : {'none', 'best', 'tolerance', 'standard_error', 'uncertainty'}, optional
      Parsimony selection mode for downstream feature reduction (default: 'none').
      - 'none': Parsimony disabled (existing behavior).
      - 'best': Alias for 'none'; select k* (highest CV performance).
      - 'tolerance': Select smallest k where mean(degradation[k]) <= parsimony_tolerance.
      - 'standard_error': Select smallest k where mean(degradation[k]) <= parsimony_se_multiplier * SE(degradation[k]).
      - 'uncertainty': Select smallest k where mean(degradation[k]) <= parsimony_tolerance + parsimony_se_multiplier * SE(degradation[k]).
  parsimony_tolerance : float, optional
      Tolerance threshold (in utility units) for 'tolerance' and 'uncertainty' modes.
      Default is 0.0.
  parsimony_se_multiplier : float, optional
      Multiplier for standard error in 'standard_error' and 'uncertainty' modes.
      Default is 1.0.
  log_level : {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'} or int, optional
      Logging level threshold. Use 'DEBUG' for detailed diagnostics,
      'INFO' for standard output, 'WARNING' to suppress most output.
      Default is logging.INFO.

    Notes
    -----
    Automatic best-subset selection uses a reliability score. With
    ``selection_strategy='legacy'``, for each selected prefix,
    ``performance_score = (train * cv * test) ** (1/3)``,
    ``instability_score = |train-cv| + |train-test| + |cv-test|``, and for
    higher-is-better metrics ``reliability_score = performance_score /
    (1 + instability_score)``. For lower-is-better metrics, the geometric
    mean is inverted first so the same reliability score can be maximised.
    With ``selection_strategy='cv_only'``, the same formulas are used but
    with only ``train`` and ``cv`` scores (no test score involved).
  
  Examples
  --------

  >>> import pandas as pd
  >>> import numpy as np
  >>> from sklearn.linear_model import LogisticRegression
  >>> from sklearn.datasets import make_classification
  >>> from mlchem.metrics import get_geometric_S

  Standard cross-validation (existing behaviour):

  >>> sfs = SequentialForwardSelection(estimator=LogisticRegression(),
  ...                                  metric=get_geometric_S,
  ...                                  max_features=5,
  ...                                  cv_iter=3,
  ...                                  logic='greater')

  GroupKFold with scaffold groups:

  >>> from sklearn.model_selection import GroupKFold
  >>> sfs = SequentialForwardSelection(estimator=LogisticRegression(),
  ...                                  metric=get_geometric_S,
  ...                                  cv_splitter=GroupKFold(5),
  ...                                  groups=scaffold_ids)

  Precomputed scaffold or cluster folds:

  >>> sfs = SequentialForwardSelection(estimator=LogisticRegression(),
  ...                                  metric=get_geometric_S,
  ...                                  cv_indices=scaffold_fold_manifest)

  >>> X, y = make_classification(300, 10, n_informative=5)
  >>> train_size = 0.8
  >>> train_samples = int(train_size * len(X))

  >>> X_train, y_train = X[:train_samples], y[:train_samples]
  >>> X_test, y_test = X[train_samples:], y[train_samples:]

  >>> train_set = pd.DataFrame(X_train, columns=np.arange(X_train.shape[1]))
  >>> test_set = pd.DataFrame(X_test, columns=np.arange(X_test.shape[1]))

  >>> sfs.fit(train_set, y_train, test_set, y_test)
  >>> sfs.plot(best_feature='None')


Scientific Rationale
-------------------
Parsimony selects the smallest previously visited feature subset whose
paired cross-validation degradation relative to the selected reference
subset is no greater than a configurable empirical uncertainty margin.
The margin is inspired by the one-standard-error rule but is computed
from matched fold-wise differences. Because cross-validation folds are
dependent, the criterion is intended for model selection and should not
be interpreted as a formal equivalence test.

### Tolerance

An optional absolute tolerance may be used to regard predictive
improvements or degradations smaller than a user-defined amount as
practically negligible. This is analogous to minimum-improvement
stopping rules used in sequential feature selection, such as the `tol`
parameter of scikit-learn's SequentialFeatureSelector [1].

### Paired uncertainty margin

The reference and candidate subsets are evaluated using identical
cross-validation splits. After orienting the scoring metric so that
larger values are better, the degradation on resample r is

    d_r = score_reference,r - score_candidate,r.

The reported uncertainty is the descriptive standard error of the mean
paired degradation,

    SE_d = SD(d_r) / sqrt(R),

where R is the number of matched resamples and SD uses the sample
standard deviation. A candidate is acceptable when its mean degradation
does not exceed `se_multiplier * SE_d`, optionally combined with an
absolute tolerance.

### Relationship to the one-standard-error rule

This criterion is inspired by the one-standard-error principle, which
prefers a simpler model when its cross-validated performance lies within
an uncertainty margin of the best-performing model [2,3]. Unlike the
classical rule, this implementation estimates uncertainty from paired
reference-minus-candidate differences rather than from the reference
model's cross-validation error alone.

### Statistical interpretation

The criterion is an uncertainty-aware model-selection heuristic, not a
hypothesis test or an equivalence/non-inferiority procedure. Cross-
validation results are dependent because training sets overlap; therefore
SD(d_r) / sqrt(R) can understate or otherwise misrepresent the sampling
uncertainty [4-6]. Acceptance means only that the observed mean degradation
falls within the configured empirical margin on the supplied resamples.

### References:

[1] scikit-learn developers, SequentialFeatureSelector documentation.
[2] Breiman et al. (1984), Classification and Regression Trees.
[3] Hastie, Tibshirani & Friedman (2009), Elements of Statistical Learning.
[4] Dietterich (1998), doi:10.1162/089976698300017197
[5] Nadeau & Bengio (2003), doi:10.1023/A:1024068626366
[6] Bengio & Grandvalet (2004), JMLR 5:1089-1105
  """

    def __init__(self,
                 estimator,
                 estimator_string: Optional[str],
                 metric: Callable,
                 max_features: int = 25,
                 cv_iter: int = 5,
                 cv_splitter=None,
                 cv_indices: Iterable | None = None,
                 groups=None,
                 logic: Literal['lower', 'greater'] = 'greater',
                 task_type: Literal[
                     'classification', 'regression'] = 'classification',
                 selection_strategy: Literal['legacy', 'cv_only'] = 'legacy',
                 log_level: int | str | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = logging.INFO,
                 ) -> None:
        """
  Initialise the SequentialForwardSelection object.

Attributes
  ----------
  estimator : object
      The scikit-learn estimator used for feature selection.
  estimator_string : str, optional
      A string representation of the estimator. If None, it is inferred from the estimator.
  metric : callable
      A function to evaluate model performance.
  max_features : int, optional
      Maximum number of features to select. Default is 25.
  cv_iter : int, optional
      Number of cross-validation iterations. Default is 5. Ignored when
      ``cv_splitter`` or ``cv_indices`` is provided.
  cv_splitter : object, optional
      A scikit-learn compatible cross-validation splitter (e.g.
      ``GroupKFold(5)``, ``StratifiedGroupKFold(...)``,
      ``PredefinedSplit(...)``). Combine with ``groups`` for group-aware
      splitters. Takes precedence over ``cv_iter``.
  groups : array-like, optional
      Group labels (e.g. scaffold IDs) propagated to ``cv_splitter``.
      Ignored when ``cv_indices`` is provided.
  cv_indices : iterable of (array-like, array-like), optional
      Explicit, pre-computed ``(train_idx, valid_idx)`` pairs, e.g. a
      scaffold-based or UMAP-cluster-based fold manifest generated
      outside mlchem. Takes precedence over both ``cv_splitter`` and
      ``cv_iter``. This is the most generic API.
  logic : {'lower', 'greater'}, optional
      Whether to minimize or maximize the cross-validation score. Default is 'greater'.
  task_type : {'classification', 'regression'}, optional
      Type of task. Default is 'classification'.
  selection_strategy : {'legacy', 'cv_only'}, optional
      Strategy used by :meth:`find_best` to select the winning feature
      subset. ``'legacy'`` (default, kept for backward compatibility)
      uses train/CV/test scores, exactly as in previous releases.
      ``'cv_only'`` is a leakage-free mode that selects using only
      train/CV information; test scores are still computed and stored
      for reporting, but never influence selection. ``'cv_only'`` is the
      recommended choice for new projects.
  log_level : {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'} or int, optional
      Logging level threshold. Use 'DEBUG' for detailed diagnostics,
      'INFO' for standard output, 'WARNING' to suppress most output.
      Default is logging.INFO.

    Notes
    -----
    Automatic best-subset selection uses a reliability score. With
    ``selection_strategy='legacy'``, for each selected prefix,
    ``performance_score = (train * cv * test) ** (1/3)``,
    ``instability_score = |train-cv| + |train-test| + |cv-test|``, and for
    higher-is-better metrics ``reliability_score = performance_score /
    (1 + instability_score)``. For lower-is-better metrics, the geometric
    mean is inverted first so the same reliability score can be maximised.
    With ``selection_strategy='cv_only'``, the same formulas are used but
    with only ``train`` and ``cv`` scores (no test score involved).
  """

        self.estimator = estimator
        if not estimator_string:
            estimator_string = str(estimator)
        self.estimator_string = estimator_string
        self.metric = metric
        self.max_features = max_features
        self.cv_iter = cv_iter
        self.cv_splitter = cv_splitter
        self.cv_indices = cv_indices
        self.groups = groups
        if selection_strategy not in ('legacy', 'cv_only'):
            raise ValueError("'selection_strategy' must be either 'legacy' or 'cv_only'.")
        self.selection_strategy = selection_strategy
        
        # Warn if using legacy selection_strategy (backward-compatible default)
        if selection_strategy == 'legacy':
            warnings.warn(
                "selection_strategy='legacy' is deprecated and will be removed in a future version. "
                "The test score leak it entails is problematic. "
                "Please explicitly set selection_strategy='cv_only' (recommended) or 'legacy' (if needed). "
                "See docstring for details.",
                FutureWarning,
                stacklevel=2,
            )

        self.logic = logic
        self.task_type = validate_task_type(task_type)
        self.log_level = coerce_log_level(log_level)
        _configure_wrapper_logging(self.log_level)

        # Where to store the temporarily best feature set at each iteration
        self.extending_features = []

        # Where to store all training scores obtained from the model
        # using the accepted features
        self.train_scores = []

        # Where to store all cross-validation scores obtained from the
        # model using the accepted features
        self.cv_scores = []

        # Where to store the standard error of the cv scores
        self.cv_se = []

        # Where to store test scores
        self.unseen_scores = []

        # NEW: Storage for fold-level CV scores (for parsimony)
        # Structure: list of numpy arrays, one per subset
        # (indexed by subset size: 0-based index = subset size - 1)
        self.cv_folds = []

        # NEW: Storage for parsimony diagnostics
        self.parsimony_diagnostics = {}

    def set_log_level(self, log_level: int | str | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']) -> None:
        """Set the logging level for wrapper diagnostics.

        Parameters
        ----------
        log_level : {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'} or int
            Logging level threshold. Use 'DEBUG' for detailed diagnostics,
            'INFO' for standard output, 'WARNING' to suppress most output.
        """
        self.log_level = coerce_log_level(log_level)
        _configure_wrapper_logging(self.log_level)

    def _log(self, level: int, msg: str, *args) -> None:
        if level >= self.log_level:
            logger.log(level, msg, *args)

    def fit(
        self,
        train_set: pd.DataFrame,
        y_train: Iterable,
        test_set: pd.DataFrame,
        y_test: Iterable,
        n_jobs: int = 1,
    ) -> None:
        """
        Fit the Sequential Forward Selection model.

        Parameters
        ----------
        train_set : pandas.DataFrame
            Training dataset.
        y_train : iterable
            Target values for the training set.
        test_set : pandas.DataFrame
            Test dataset.
        y_test : iterable
            Target values for the test set.
        n_jobs : int, optional
            Number of parallel workers used to evaluate candidate
            features at each SFS cycle. Use -1 to use all available CPUs.
            Default is 1.

        Returns
        -------
        None
        """

        self.train_set = train_set
        self.y_train = y_train
        self.test_set = test_set
        self.y_test = y_test
        self.feature_labels = self.train_set.columns
        self.n_jobs = resolve_n_jobs(n_jobs)

        # Resolve fold definitions once so every candidate feature is
        # evaluated against exactly the same folds. Returns None (legacy
        # behaviour) unless cv_splitter/cv_indices was supplied.
        self._resolved_cv_indices = _resolve_wrapper_cv_indices(
            self.train_set,
            self.y_train,
            self.cv_iter,
            self.cv_splitter,
            self.cv_indices,
            self.groups,
            self.task_type,
        )

        self._log(
            logging.INFO,
            "SFS start: samples=%d, features=%d, max_features=%d, cv_iter=%d, n_jobs=%d",
            len(self.train_set),
            len(self.feature_labels),
            self.max_features,
            self.cv_iter,
            self.n_jobs,
        )

        # sklearn's wrapper propagates the caller's sklearn config to worker
        # threads, avoiding an upstream UserWarning that plain joblib.delayed
        # does not.
        from sklearn.utils.parallel import Parallel, delayed

        def evaluate_feature(feat):
            features_to_test = self.extending_features + [feat]
            train_set_temp = self.train_set[features_to_test]
            estimator_copy = _clone_for_search(self.estimator, self.n_jobs)
            estimator_copy.fit(train_set_temp, self.y_train)
            cv_kwargs = (
                {'cv_indices': self._resolved_cv_indices}
                if self._resolved_cv_indices is not None else {}
            )
            cvscores = crossval(
                estimator_copy,
                train_set_temp.values,
                y_train,
                self.metric,
                self.cv_iter,
                self.task_type,
                **cv_kwargs,
            )
            self._log(
                logging.DEBUG,
                "SFS candidate=%s | subset_size=%d | cv_mean=%.4f | cv_se=%.4f",
                feat,
                len(features_to_test),
                float(np.mean(cvscores)),
                float(np.std(cvscores)) / np.sqrt(len(cvscores)),
            )
            return np.mean(cvscores), np.std(cvscores) / np.sqrt(len(cvscores)), cvscores

        for cycle in tqdm(range(self.max_features), desc="SFS", disable=False):

            # Temporary lists where to store cross-validation scores,
            # standard errors, and fold-level scores.
            cv_scores_storage = []
            cv_se_storage = []
            cv_folds_storage = []

            # List of features to be assessed
            self.list_available_features = [feat for feat in
                                            self.feature_labels if
                                            feat not in self.extending_features
                                            ]
            self._log(
                logging.DEBUG,
                "SFS cycle=%d | available_features=%d",
                cycle + 1,
                len(self.list_available_features),
            )

            # Hypothetically assess model if an extra feature is added.
            # Do it for all unexplored features.
            if self.n_jobs == 1:
                for feat in self.list_available_features:
                    cv_mean, cv_se, cv_folds = evaluate_feature(feat)
                    cv_scores_storage.append(cv_mean)
                    cv_se_storage.append(cv_se)
                    cv_folds_storage.append(cv_folds)
            else:
                scored = Parallel(n_jobs=self.n_jobs, prefer='threads')(
                    delayed(evaluate_feature)(feat)
                    for feat in self.list_available_features
                )
                cv_scores_storage = [score for score, _, _ in scored]
                cv_se_storage = [se for _, se, _ in scored]
                cv_folds_storage = [folds for _, _, folds in scored]

            # Include in the model the feature with best CV gains.
            if self.logic == 'greater':
                index = np.argmax(cv_scores_storage)
            else:
                index = np.argmin(cv_scores_storage)

            self.cv_scores.append(cv_scores_storage[index])
            self.cv_se.append(cv_se_storage[index])
            # Store fold-level scores (oriented to utility: higher is better)
            fold_level_utility = _orient_scores_to_utility(
                cv_folds_storage[index],
                self.logic,
            )
            self.cv_folds.append(fold_level_utility)
            feature_to_add = self.list_available_features[index]
            self.extending_features.append(feature_to_add)

            self._log(
                logging.INFO,
                "SFS accepted cycle=%d | feature=%s | cv=%.4f +- %.4f",
                cycle + 1,
                feature_to_add,
                self.cv_scores[-1],
                self.cv_se[-1],
            )

            # Get score on unseen test data
            train_set_temp = self.train_set[self.extending_features]
            test_set_temp = self.test_set[self.extending_features]
            self.estimator.fit(train_set_temp, y_train)
            y_train_pred = self.estimator.predict(train_set_temp)
            y_test_pred = self.estimator.predict(test_set_temp)
            self.train_scores.append(self.metric(self.y_train, y_train_pred))
            self.unseen_scores.append(self.metric(self.y_test, y_test_pred))
            self._log(
                logging.DEBUG,
                "SFS cycle=%d scores | train=%.4f | test=%.4f",
                cycle + 1,
                self.train_scores[-1],
                self.unseen_scores[-1],
            )

        self._log(
            logging.INFO,
            "SFS completed: selected_features=%d",
            len(self.extending_features),
        )

    def _apply_parsimony_selection(
        self,
        parsimony_mode: Literal['best', 'tolerance', 'standard_error', 'uncertainty'],
        tolerance: float = 0.0,
        se_multiplier: float = 1.0,
    ) -> dict:
        """
        Apply parsimony-based subset selection using fold-level degradation.

        This helper is called by :meth:`find_best` when parsimony selection is
        requested. It computes fold-level degradation d[k,r] for each subset k
        compared to the reference k* (best reliability score), then applies
        the requested parsimony criterion to select a final subset.

        Stores results in self.parsimony_diagnostics for inspection.
        """
        # Find the reference subset k* using the same reliability baseline as
        # the non-parsimony path so the reference is stable across selection
        # modes and independent from the parsimony rule itself.
        scores = [
            calculate_reliability_components(
                train_score=train_score,
                cv_score=cv_score,
                test_score=None if self.selection_strategy == 'cv_only' else test_score,
                logic=self.logic,
            )
            for train_score, cv_score, test_score in zip(
                self.train_scores,
                self.cv_scores,
                self.unseen_scores,
            )
        ]
        kstar_index_zero = int(np.argmax([
            score['reliability_score'] for score in scores
        ]))

        kstar_index = kstar_index_zero + 1  # Convert to 1-based
        cv_folds_kstar = self.cv_folds[kstar_index_zero]

        # Apply parsimony selection rule
        parsimony_result = _select_parsimonious_subset(
            cv_folds_kstar=cv_folds_kstar,
            cv_folds_kstar_index=kstar_index,
            all_cv_folds=self.cv_folds,
            parsimony_mode=parsimony_mode,
            tolerance=tolerance,
            se_multiplier=se_multiplier,
        )

        self.parsimony_diagnostics = parsimony_result

        self._log(
            logging.INFO,
            "SFS parsimony: mode=%s | reference_k*=%d | selected_k=%d",
            parsimony_mode,
            kstar_index,
            parsimony_result['selected_index'],
        )

        return parsimony_result

    def find_best(
        self,
        which: Optional[int] = None,
        parsimony_mode: Literal['none', 'best', 'tolerance', 'standard_error', 'uncertainty'] = 'none',
        parsimony_tolerance: float = 0.0,
        parsimony_se_multiplier: float = 1.0,
    ) -> dict:
        """
        Find the best feature subset based on reliability score and,
        optionally, on parsimony.

        Parameters
        ----------
        which : int, optional
            If specified, returns the feature subset at the given index.
            If None, the best subset is determined automatically using the
            reliability score.
        parsimony_mode : {'none', 'best', 'tolerance', 'standard_error', 'uncertainty'}, optional
            Parsimony selection mode used when ``which`` is ``None``.
            ``'none'`` keeps the original reliability-score selection.
            ``'best'`` returns the reference subset ``k*``.
            ``'tolerance'``, ``'standard_error'``, and ``'uncertainty'``
            select a smaller subset when the fold-level degradation meets
            the corresponding criterion.
        parsimony_tolerance : float, optional
            Absolute tolerance used by ``'tolerance'`` and ``'uncertainty'``.
            Default is 0.0.
        parsimony_se_multiplier : float, optional
            Standard-error multiplier used by ``'standard_error'`` and
            ``'uncertainty'``. Default is 1.0.

        Returns
        -------
        dict
            Dictionary containing the selected subset and reliability scores.

            ``best_index`` : int
                Number of selected features in the winning prefix.
            ``features`` : list
                Selected feature names.
            ``performance_score`` : float
                Performance contribution for the winning prefix.
            ``instability_score`` : float
                Sum of train/CV/test score gaps for the winning prefix.
            ``reliability_score`` : float
                Reliability score used for automatic selection.
            ``best_score`` : float
                Alias of ``reliability_score`` retained for backwards
                compatibility.

        Notes
        -----
        For each feature subset, the automatic algorithm computes:

        ``reliability_score = performance_score / (1 + instability_score)``

        With ``selection_strategy='legacy'`` (default, for backward
        compatibility):

        ``instability_score = |train-cv| + |train-test| + |cv-test|``

        and, for higher-is-better metrics:

        ``performance_score = (train_score + cv_score + test_score) / 3``

        For lower-is-better metrics, such as RMSE, lower performance scores
        are better, so the performance score has its sign flipped before
        reliability calculation is applied:

        ``performance_score = -(train_score + cv_score + test_score) / 3``

        For this selection strategy ('legacy'), the test score is
        intentionally included in the calculation.

        With ``selection_strategy='cv_only'`` (leakage-free, recommended for
        new projects), the same formulas are used but with only
        ``train_score`` and ``cv_score`` (no test score involved), so the
        test score can never influence which subset is selected. Test scores
        are still stored in ``self.unseen_scores`` for reporting/plotting.

        With parsimony selection enabled (via ``parsimony_mode``), the best
        subset is selected based on fold-level degradation criteria rather
        than the reliability score. The reference subset is the one with the
        highest reliability score, and it is compared against all smaller
        subsets.
        """

        if which is None:
            if len(self.cv_scores) == 0:
                raise ValueError("No feature subsets have been evaluated. Run fit() before find_best().")

            if parsimony_mode not in ('none', 'best', 'tolerance', 'standard_error', 'uncertainty'):
                raise ValueError(
                    "'parsimony_mode' must be one of ('none', 'best', 'tolerance', 'standard_error', 'uncertainty'), "
                    f"got '{parsimony_mode}'."
                )

            if parsimony_mode != 'none':
                parsimony_result = self._apply_parsimony_selection(
                    parsimony_mode='best' if parsimony_mode == 'best' else parsimony_mode,
                    tolerance=parsimony_tolerance,
                    se_multiplier=parsimony_se_multiplier,
                )
                best_index = parsimony_result['selected_index']
                dictionary = {
                    'best_index': best_index,
                    'features': self.extending_features[:best_index],
                    'parsimony_mode': parsimony_mode,
                    'parsimony_diagnostics': parsimony_result,
                }
            else:
                self.parsimony_diagnostics = {}
                # Original behaviour: use reliability score
                scores = [
                    calculate_reliability_components(
                        train_score=train_score,
                        cv_score=cv_score,
                        test_score=None if self.selection_strategy == 'cv_only' else test_score,
                        logic=self.logic,
                    )
                    for train_score, cv_score, test_score in zip(
                        self.train_scores,
                        self.cv_scores,
                        self.unseen_scores,
                    )
                ]

                best_index_zero_based = int(np.argmax([
                    score['reliability_score'] for score in scores
                ]))
                best_index = best_index_zero_based + 1
                winning_scores = scores[best_index_zero_based]
                dictionary = {
                    'best_index': best_index,
                    'features': self.extending_features[:best_index],
                    'performance_score': winning_scores['performance_score'],
                    'instability_score': winning_scores['instability_score'],
                    'reliability_score': winning_scores['reliability_score'],
                    'best_score': winning_scores['reliability_score'],
                }
        else:     # if which == int
            best_index = which
            dictionary = {'best_index': best_index,
                          'features': self.extending_features[:best_index]
                          }
        return dictionary

    def plot(
        self,
        best_feature: int | Literal['auto'] | None = 'auto',
        figsize: tuple[int, int] = (10, 6),
        colours: list[str] = ['steelblue', 'orange', 'green'],
        title: str | None = None,
        title_size: int = 20,
        xlabel: str = '# of features',
        ylabel: str = 'Score',
        fontsize: int = 14,
        legendsize: int = 13,
        save: bool = False
         ) -> None:
        """
        Plot the performance of the Sequential Forward Selection process.
        The interval of 1 standard error is shown as a shaded region
        around the cv_score curve. 

        Parameters
        ----------
        best_feature : int, 'auto', or None, optional
            Index of the best feature subset to highlight. If 'auto' or
            None, it is determined automatically using reliability-score
            selection. Default is 'auto'.
        figsize : tuple of int, optional
            Size of the plot. Default is (10, 6).
        colours : list of str, optional
            Colours for training, validation, and test scores. Default is
            ['steelblue', 'orange', 'green'].
        title : str, optional
            Title of the plot.
        title_size : int, optional
            Font size of the title. Default is 20.
        xlabel : str, optional
            Label for the x-axis. Default is '# of features'.
        ylabel : str, optional
            Label for the y-axis. Default is 'Score'.
        fontsize : int, optional
            Font size for axis labels. Default is 14.
        legendsize : int, optional
            Font size for the legend. Default is 13.
        save : bool, optional
            Whether to save the plot. Default is False.

        Returns
        -------
        None

        Notes
        -----
        The automatic algorithm for determining the best feature subset
        is the same as described in `find_best`: ``performance_score =
        (train + cv + test) / 3``, ``instability_score = |train-cv| +
        |train-test| + |cv-test|``, and for higher-is-better metrics
        ``reliability_score = performance_score / (1 + instability_score)``.
        For lower-is-better metrics, the performance score has its sign flipped.
        The subset with the highest reliability score is highlighted.
        """

        assert best_feature in ('auto', None) or isinstance(best_feature, int), \
            "'best_feature' must be an integer, 'auto', or None."

        # Capture estimator name
        if not self.estimator_string:
            self.estimator_string = str(self.estimator)[
                :str(self.estimator).find('(')
                ]

        plt.figure(figsize=figsize)

        if not title:
            title_text = f'SFS - model'
        else:
            title_text = title

        plt.title(title_text,fontsize=title_size)
        plt.grid(axis='y')
        plt.xlabel(xlabel, size=fontsize)
        plt.ylabel(ylabel, size=fontsize)


        # Plot training scores
        plt.plot(range(1, len(self.train_scores)+1),
                 self.train_scores,
                 label='training score',
                 color=colours[0])
        # Plot cross-validation scores
        plt.plot(range(1, len(self.train_scores)+1),
                 self.cv_scores,
                 label='validation score',
                 color=colours[1])
        # Show standard deviation of cross-validation performance
        plt.fill_between(range(1, len(self.train_scores)+1),
                         np.array(self.cv_scores) - 0.5 * np.array(self.cv_se),
                         np.array(self.cv_scores) + 0.5 * np.array(self.cv_se),
                         alpha=0.2, color=colours[1])
        # Plot test scores
        plt.plot(range(1, len(self.train_scores)+1),
                 self.unseen_scores,
                 label='test score',
                 color=colours[2])

        plt.legend(fontsize=legendsize, loc='best')

        which = None if best_feature in ('auto', None) else best_feature
        ind = self.find_best(which=which)['best_index']

        # Draw a vertical line corresponding to the best iteration
        # returning the optimal scores.
        plt.axvline(ind,
                    ls='--',
                    c='r',
                    lw=1)

        if save:     # save estimator, all columns, best columns.

            import joblib

            plt.savefig('SFS_%s.png' % (self.estimator_string),
                        dpi=500)
            joblib.dump(self.estimator,
                        self.estimator_string)
            joblib.dump(self.extending_features,
                        self.estimator_string+'_allcols')
            joblib.dump(self.extending_features[:ind],
                        self.estimator_string+'_best')

        plt.show()

        self._log(logging.INFO, 'SFS summary: number_of_features=%d', ind)
        self._log(logging.INFO, 'SFS summary: winner_subset=%s', self.extending_features[:ind])
        self._log(logging.INFO, 'SFS summary: train_score=%.3f', self.train_scores[ind - 1])
        self._log(
            logging.INFO,
            'SFS summary: cv_score=%.3f +- %.3f',
            self.cv_scores[ind - 1],
            self.cv_se[ind - 1],
        )
        self._log(logging.INFO, 'SFS summary: test_score=%.3f', self.unseen_scores[ind - 1])


class CombinatorialSelection:
    """
    Combinatorial feature selection wrapper.

    This class performs a two-stage combinatorial search over feature
    subsets and ranks the surviving subsets with the same reliability
    score logic used by :class:`SequentialForwardSelection`: the
    performance term is the arithmetic mean of the available train/CV/
    test scores, and lower-is-better metrics are sign-flipped before the
    reliability score is computed. Results are then ranked by
    ``reliability_score = performance_score / (1 + instability_score)``,
    where ``instability_score`` is the sum of pairwise gaps between the
    available scores.

    Public Methods
    --------------
    ``__init__()``: Initialise the combinatorial selector.

    ``set_log_level()``: Set the logging level for diagnostics.

    ``rank_features_by_relevance_redundancy()``: Optional pre-ranking of
    features before subset generation.

    ``fit_stage_1()``: Evaluate and filter candidate subsets in stage 1.

    ``fit_stage_2()``: Refine the stage 1 winners in a second combinatorial
    pass.

    ``display_best()``: Fit and report the best subset from stage 2.

    Attributes
    ----------
    estimator : object
        The machine learning estimator used to fit the data.
    metric : callable
        A metric function to evaluate estimator performance. Must accept
        ``(y_true, y_pred)``.
    logic : {'greater', 'lower'}
        Determines whether a higher or lower score is considered better.
    task_type : {'classification', 'regression'}
        Specifies the type of task.
    cv_splitter : object, optional
        A scikit-learn compatible cross-validation splitter.
    cv_indices : iterable of (array-like, array-like), optional
        Explicit, pre-computed ``(train_idx, valid_idx)`` pairs.
    groups : array-like, optional
        Group labels propagated to ``cv_splitter``.
    selection_strategy : {'legacy', 'cv_only'}
        Controls whether the test score participates in reliability
        scoring.
    log_level : int or str
        Logging threshold used by the wrapper logger.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> from mlchem.metrics import get_geometric_S

    Standard cross-validation (existing behaviour):

    >>> cs = CombinatorialSelection(estimator=LogisticRegression(),
    ...                              metric=get_geometric_S,
    ...                              logic='greater')

    GroupKFold with scaffold groups:

    >>> from sklearn.model_selection import GroupKFold
    >>> cs = CombinatorialSelection(estimator=LogisticRegression(),
    ...                              metric=get_geometric_S,
    ...                              cv_splitter=GroupKFold(5),
    ...                              groups=scaffold_ids)  # doctest: +SKIP

    Precomputed scaffold or UMAP-cluster folds:

    >>> cs = CombinatorialSelection(estimator=LogisticRegression(),
    ...                              metric=get_geometric_S,
    ...                              cv_indices=scaffold_fold_manifest)  # doctest: +SKIP
    """

    def __init__(self,
                 estimator,
                 metric,
                 logic: Literal['lower', 'greater'] = 'greater',
                 task_type: Literal[
                     'classification', 'regression'
                     ] = 'classification',
                 cv_splitter=None,
                 cv_indices: Iterable | None = None,
                 groups=None,
                 selection_strategy: Literal['legacy', 'cv_only'] = 'legacy',
                 log_level: int | str | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = logging.INFO,
                 ) -> None:
        """
        Initialise the CombinatorialSelection object.

        Parameters
        ----------
        estimator : object
            The machine learning estimator used to fit the data.
        metric : callable
            A metric function to evaluate estimator performance.
        logic : {'greater', 'lower'}, optional
            Determines whether a higher or lower score is considered better.
            Default is 'greater'.
        task_type : {'classification', 'regression'}, optional
            Specifies the type of task. Default is 'classification'.
        cv_splitter : object, optional
            A scikit-learn compatible cross-validation splitter. Takes
            precedence over the ``cv_iter`` passed to
            ``fit_stage_1``/``fit_stage_2``.
        cv_indices : iterable of (array-like, array-like), optional
            Explicit ``(train_idx, valid_idx)`` fold pairs. Takes
            precedence over both ``cv_splitter`` and ``cv_iter``.
        groups : array-like, optional
            Group labels forwarded to ``cv_splitter``. Ignored when
            ``cv_indices`` is provided.
        selection_strategy : {'legacy', 'cv_only'}, optional
            Whether the ``reliability_score`` used for ranking includes
            the test score (``'legacy'``, default) or not (``'cv_only'``,
            leakage-free). Default is ``'legacy'`` for backward
            compatibility.
        log_level : {{'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}} or int, optional
            Logging level threshold. Use 'DEBUG' for detailed diagnostics,
            'INFO' for standard output, 'WARNING' to suppress most output.
            Default is logging.INFO.
        """

        self.estimator = estimator
        self.metric = metric
        self.logic = logic
        self.task_type = validate_task_type(task_type)
        self.log_level = coerce_log_level(log_level)
        self.cv_splitter = cv_splitter
        self.cv_indices = cv_indices
        self.groups = groups
        if selection_strategy not in ('legacy', 'cv_only'):
            raise ValueError("'selection_strategy' must be either 'legacy' or 'cv_only'.")
        self.selection_strategy = selection_strategy
        
        # Warn if using legacy selection_strategy (backward-compatible default)
        if selection_strategy == 'legacy':
            warnings.warn(
                "selection_strategy='legacy' is deprecated and will be removed in a future version. "
                "The test score leak it entails is problematic. "
                "Please explicitly set selection_strategy='cv_only' (recommended) or 'legacy' (if needed). "
                "See docstring for details.",
                FutureWarning,
                stacklevel=2,
            )
        
        _configure_wrapper_logging(self.log_level)

    def set_log_level(self, log_level: int | str | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']) -> None:
        """Set the logging level for wrapper diagnostics.

        Parameters
        ----------
        log_level : {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'} or int
            Logging level threshold. Use 'DEBUG' for detailed diagnostics,
            'INFO' for standard output, 'WARNING' to suppress most output.
        """
        self.log_level = coerce_log_level(log_level)
        _configure_wrapper_logging(self.log_level)

    def _log(self, level: int, msg: str, *args) -> None:
        if level >= self.log_level:
            logger.log(level, msg, *args)

    @staticmethod
    def _validate_subset_limit(
        n_features: int,
        subset_size: int,
        max_subsets: int | None,
        stage_name: str,
    ) -> None:
        if max_subsets is None:
            return

        if max_subsets < 1:
            raise ValueError("'max_subsets' must be at least 1.")

        effective_subset_size = min(n_features, subset_size)
        total_subsets = sum(
            comb(n_features, current_subset_size)
            for current_subset_size in range(1, effective_subset_size + 1)
        )
        if total_subsets > max_subsets:
            raise ValueError(
                (
                    f"{stage_name} would generate {total_subsets} subsets, "
                    f"which exceeds max_subsets={max_subsets}. "
                    "Increase max_subsets or reduce search space "
                    "(fewer features, smaller subset size, or tighter thresholds)."
                )
            )

    def rank_features_by_relevance_redundancy(
        self,
        dataframe: pd.DataFrame,
        target: Iterable,
        features: list[str] | None = None,
        alpha: float = 1.0,
        beta: float = 0.2,
        top_features: int | None = None,
        relevance_metric: Literal['mutual_info', 'pearson', 'spearman'] = 'mutual_info',
        redundancy_metric: Literal['pearson', 'spearman'] = 'pearson',
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Rank features with a global relevance-redundancy criterion.

        Higher `alpha` increases target relevance importance.
        Higher `beta` increases redundancy penalty importance.
        """

        if alpha < 0 or beta < 0:
            raise ValueError("'alpha' and 'beta' must be non-negative.")
        if redundancy_metric not in ('pearson', 'spearman'):
            raise ValueError("'redundancy_metric' must be either 'pearson' or 'spearman'.")

        feature_pool = list(features) if features is not None else list(dataframe.columns)
        if len(feature_pool) == 0:
            raise ValueError("No features available for ranking.")

        y = np.asarray(target)
        if y.ndim > 1:
            y = y.ravel()
        if len(y) != len(dataframe):
            raise ValueError("'target' length must match dataframe rows.")

        if top_features is None:
            top_features = len(feature_pool)
        if top_features < 1:
            raise ValueError("'top_features' must be at least 1.")
        top_features = min(top_features, len(feature_pool))

        if relevance_metric == 'mutual_info':
            if self.task_type == 'classification':
                from sklearn.feature_selection import mutual_info_classif
                relevance_array = mutual_info_classif(
                    dataframe[feature_pool].values,
                    y,
                    random_state=random_state,
                )
            else:
                from sklearn.feature_selection import mutual_info_regression
                relevance_array = mutual_info_regression(
                    dataframe[feature_pool].values,
                    y,
                    random_state=random_state,
                )
            relevance_scores = {
                feat: float(score)
                for feat, score in zip(feature_pool, relevance_array)
            }
        else:
            relevance_scores = {
                feat: _safe_abs_corr(
                    dataframe[feat].values,
                    y,
                    method=relevance_metric,
                )
                for feat in feature_pool
            }

        if len(feature_pool) == 1:
            redundancy_scores = {feature_pool[0]: 0.0}
        else:
            corr_matrix = dataframe[feature_pool].corr(method=redundancy_metric).abs()
            diagonal_mask = np.eye(len(corr_matrix), dtype=bool)
            corr_matrix = corr_matrix.mask(diagonal_mask)
            redundancy_series = corr_matrix.mean(axis=1, skipna=True).fillna(0.0)
            redundancy_scores = {
                feat: float(redundancy_series.loc[feat])
                for feat in feature_pool
            }

        records = []
        for feat in feature_pool:
            relevance = relevance_scores[feat]
            redundancy = redundancy_scores[feat]
            score = alpha * relevance - beta * redundancy
            records.append({
                'feature': feat,
                'relevance': relevance,
                'redundancy': redundancy,
                'score': score,
                'alpha': alpha,
                'beta': beta,
            })

        df_ranking = pd.DataFrame(records)
        df_ranking.sort_values(by='score', ascending=False, inplace=True)
        df_ranking = df_ranking.head(top_features).copy()
        df_ranking.insert(0, 'rank', np.arange(1, len(df_ranking) + 1))
        df_ranking.reset_index(drop=True, inplace=True)
        return df_ranking

    def fit_stage_1(
        self,
        train_set: pd.DataFrame,
        y_train: Iterable,
        test_set: pd.DataFrame,
        y_test: Iterable,
        features: list[str] | None = None,
        k: int = 2,
        training_threshold: float = 0.25,
        cv_train_ratio: float = 0.7,
        cv_iter: int = 5,
        max_subsets: int | None = None,
        n_jobs: int = 1,
        ranking_target: Iterable | None = None,
        alpha: float = 1.0,
        beta: float = 0.2,
        top_ranked_features: int | None = None,
        relevance_metric: Literal['mutual_info', 'pearson', 'spearman'] = 'mutual_info',
        redundancy_metric: Literal['pearson', 'spearman'] = 'pearson',
        ranking_random_state: int = 1,
    ) -> pd.DataFrame:
        """
        Perform the first stage of combinatorial feature selection.

        Parameters
        ----------
        train_set : pandas.DataFrame
            The training dataset.
        y_train : iterable
            Target values for the training dataset.
        test_set : pandas.DataFrame
            The testing dataset.
        y_test : iterable
            Target values for the testing dataset.
        features : list of str, optional
            List of features to consider. Default is an empty list.
        k : int, optional
            Number of features to combine. Default is 2.
        training_threshold : float, optional
            Minimum training score required to consider a subset.
            Default is 0.25.
        cv_train_ratio : float, optional
            Minimum ratio of cross-validation to training score. Default
            is 0.7.
        cv_iter : int, optional
            Number of cross-validation iterations. Default is 5.
        max_subsets : int or None, optional
            Hard cap on the number of generated feature subsets.
            If None, no hard cap is applied. Default is None.
        n_jobs : int, optional
            Number of parallel workers used to evaluate candidate
            feature subsets. Use -1 to use all available CPUs.
            Default is 1.
        ranking_target : iterable or None, optional
            Target variable used by relevance-redundancy feature ranking.
            If None, no pre-ranking is applied unless `top_ranked_features`
            is provided (which then raises an error).
        alpha : float, optional
            Relevance coefficient in ranking score.
        beta : float, optional
            Redundancy penalty coefficient in ranking score.
        top_ranked_features : int or None, optional
            Number of top ranked features to keep before combinatorial
            subset generation. If None and ranking is enabled, all ranked
            features are kept.
        relevance_metric : {'mutual_info', 'pearson', 'spearman'}, optional
            Relevance metric used in ranking.
        redundancy_metric : {'pearson', 'spearman'}, optional
            Redundancy metric used in ranking.
        ranking_random_state : int, optional
            Random seed used by mutual information estimators.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the results of the first stage of
            feature selection.

        Notes
        -----
        Generates all possible feature subsets of size ``k`` and evaluates
        each subset using training, cross-validation, and test scores.
        Results are filtered using training/CV thresholds and ranked by
        ``reliability_score``. The score uses the same arithmetic-mean
        performance term and lower-is-better orientation as SFS. The legacy
        ``geometric_mean`` formulation is no longer used.
        """


        def is_better(a: float | int, b: float | int) -> bool:
            return a > b if self.logic == 'greater' else a < b

        self.train_set = train_set
        self.y_train = y_train
        self.test_set = test_set
        self.y_test = y_test

        self.features = [] if features is None else list(features)
        self.k = k
        self.training_threshold = training_threshold
        self.cv_train_ratio = cv_train_ratio
        self.cv_iter = cv_iter
        self.max_subsets = max_subsets
        self.n_jobs = resolve_n_jobs(n_jobs)

        # Resolve fold definitions once so every candidate subset is
        # evaluated against exactly the same folds. Returns None (legacy
        # behaviour) unless cv_splitter/cv_indices was supplied.
        self._resolved_cv_indices = _resolve_wrapper_cv_indices(
            self.train_set,
            self.y_train,
            self.cv_iter,
            self.cv_splitter,
            self.cv_indices,
            self.groups,
            self.task_type,
        )

        self._log(
            logging.INFO,
            "Combinatorial stage 1 start: samples=%d, features=%d, k=%d, n_jobs=%d",
            len(self.train_set),
            len(self.features),
            self.k,
            self.n_jobs,
        )

        if not 0 <= self.cv_train_ratio <= 1:
            raise ValueError("'cv_train_ratio' must be between 0 and 1.")
        if self.logic == 'lower' and self.cv_train_ratio == 0:
            raise ValueError(
                "'cv_train_ratio' must be greater than 0 when logic='lower'."
            )

        # Set cv threshold based on the desired cv/train ratio
        self.cv_threshold = self.training_threshold * self.cv_train_ratio \
            if self.logic == 'greater' else \
            self.training_threshold / self.cv_train_ratio

        self.ascending_decision = False if self.logic == 'greater' else \
        True

        if top_ranked_features is not None and ranking_target is None:
            raise ValueError(
                "'ranking_target' must be provided when 'top_ranked_features' is set."
            )

        if ranking_target is not None:
            self.df_feature_ranking = self.rank_features_by_relevance_redundancy(
                dataframe=self.train_set,
                target=ranking_target,
                features=list(self.features),
                alpha=alpha,
                beta=beta,
                top_features=top_ranked_features,
                relevance_metric=relevance_metric,
                redundancy_metric=redundancy_metric,
                random_state=ranking_random_state,
            )
            self.features = self.df_feature_ranking.feature.tolist()

        self._validate_subset_limit(
            n_features=len(self.features),
            subset_size=self.k,
            max_subsets=self.max_subsets,
            stage_name='fit_stage_1',
        )

        self.feature_subsets = generate_combination_cascade(self.features,
                                                            self.k)

        self.dict_results = {
            'feature_subsets': [],
            'training_score': [],
            'cv_score': [],
            'test_score': []
            }

        def evaluate_subset(subset):
            x = self.train_set[subset]
            estimator_copy = _clone_for_search(self.estimator, self.n_jobs)
            estimator_copy.fit(x.values, self.y_train)
            y_train_pred = estimator_copy.predict(x.values)
            train_score = self.metric(self.y_train, y_train_pred)
            if not is_better(train_score, self.training_threshold):
                self._log(
                    logging.DEBUG,
                    "Stage 1 rejected subset=%s by train threshold: %.4f",
                    subset,
                    train_score,
                )
                return None

            cv_score = crossval(
                estimator_copy,
                x,
                y_train,
                self.metric,
                self.cv_iter,
                self.task_type,
                **({'cv_indices': self._resolved_cv_indices} if self._resolved_cv_indices is not None else {}),
            ).mean()
            if not is_better(cv_score, self.cv_threshold):
                self._log(
                    logging.DEBUG,
                    "Stage 1 rejected subset=%s by cv threshold: %.4f",
                    subset,
                    cv_score,
                )
                return None

            y_test_pred = estimator_copy.predict(self.test_set[subset].values)
            test_score = self.metric(self.y_test, y_test_pred)
            self._log(
                logging.DEBUG,
                "Stage 1 accepted subset=%s | train=%.4f | cv=%.4f | test=%.4f",
                subset,
                train_score,
                cv_score,
                test_score,
            )
            return subset, train_score, cv_score, test_score

        if self.n_jobs == 1:
            for i, subset in enumerate(tqdm(self.feature_subsets, desc="Stage 1", disable=False)):
                result = evaluate_subset(subset)
                if result is None:
                    continue
                subset, train_score, cv_score, test_score = result
                self.dict_results['feature_subsets'].append(subset)
                self.dict_results['training_score'].append(train_score)
                self.dict_results['cv_score'].append(cv_score)
                self.dict_results['test_score'].append(test_score)
        else:
            max_workers = self.n_jobs if self.n_jobs > 0 else None
            # Benign: sklearn's own nested cross_val_score Parallel/delayed
            # can misreport propagation when dispatched from raw threads.
            with _suppress_parallel_delayed_warning():
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(evaluate_subset, subset): subset for subset in self.feature_subsets}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Stage 1", disable=False):
                        result = future.result()
                        if result is None:
                            continue
                        subset, train_score, cv_score, test_score = result
                        self.dict_results['feature_subsets'].append(subset)
                        self.dict_results['training_score'].append(train_score)
                        self.dict_results['cv_score'].append(cv_score)
                        self.dict_results['test_score'].append(test_score)
        self.df_results_stage1 = pd.DataFrame(
            self.dict_results,
            columns=self.dict_results.keys()
            )
        self.df_results_stage1 = _add_reliability_columns(
            self.df_results_stage1,
            self.logic,
            self.selection_strategy,
        )
        self.df_results_stage1.sort_values(
            by='reliability_score',
            ascending=False,
            inplace=True)
        self._log(
            logging.INFO,
            "Combinatorial stage 1 completed: kept_subsets=%d",
            len(self.df_results_stage1),
        )
        return self.df_results_stage1

    def fit_stage_2(self,
                    top_n_subsets: int = 10,
                    cv_iter: int = 5,
                    max_subsets: int | None = None,
                    n_jobs: int = 1) -> pd.DataFrame:
        """
        Perform the second stage of combinatorial feature selection.

        Parameters
        ----------
        top_n_subsets : int, optional
            Number of top feature subsets from stage 1 to consider.
            Default is 10.
        cv_iter : int, optional
            Number of cross-validation iterations. Default is 5.
        max_subsets : int or None, optional
            Hard cap on the number of generated feature subsets.
            If None, no hard cap is applied. Default is None.
        n_jobs : int, optional
            Number of parallel workers used to evaluate candidate
            feature subsets. Use -1 to use all available CPUs.
            Default is 1.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the results of the second stage of
            feature selection.
        """

        def is_better(a: float | int, b: float | int) -> bool:
            return a > b if self.logic == 'greater' else a < b

        self.cv_iter = cv_iter
        self.n_jobs = resolve_n_jobs(n_jobs)

        # Resolve fold definitions once so every candidate subset is
        # evaluated against exactly the same folds. Returns None (legacy
        # behaviour) unless cv_splitter/cv_indices was supplied.
        self._resolved_cv_indices = _resolve_wrapper_cv_indices(
            self.train_set,
            self.y_train,
            self.cv_iter,
            self.cv_splitter,
            self.cv_indices,
            self.groups,
            self.task_type,
        )

        self.best_recurrent = np.unique(
            np.hstack(
                self.df_results_stage1.head(top_n_subsets).
                feature_subsets.values)
                )

        self._validate_subset_limit(
            n_features=len(self.best_recurrent),
            subset_size=top_n_subsets,
            max_subsets=max_subsets,
            stage_name='fit_stage_2',
        )

        self.feature_subsets = generate_combination_cascade(
            self.best_recurrent, top_n_subsets
            )
        self._log(
            logging.INFO,
            "Combinatorial stage 2 start: recurrent_features=%d, subset_size=%d, n_jobs=%d",
            len(self.best_recurrent),
            top_n_subsets,
            self.n_jobs,
        )

        # Set cv threshold based on the desirede cv/train ratio
        if self.logic == 'greater':
            self.training_threshold_2 = self.df_results_stage1.\
                training_score.head(top_n_subsets).min()
            self.cv_threshold_2 = self.training_threshold_2 *\
                self.cv_train_ratio
        else:
            self.training_threshold_2 = self.df_results_stage1.\
                training_score.head(top_n_subsets).max()
            self.cv_threshold_2 = self.\
                training_threshold_2/self.cv_train_ratio

        self.dict_results_2 = {
            'feature_subsets': [],
            'training_score': [],
            'cv_score': [],
            'test_score': [],
            }

        def evaluate_subset(subset):
            x = self.train_set[subset]
            estimator_copy = _clone_for_search(self.estimator, self.n_jobs)
            estimator_copy.fit(x.values, self.y_train)
            y_train_pred = estimator_copy.predict(x.values)
            train_score = self.metric(self.y_train, y_train_pred)
            if not is_better(train_score, self.training_threshold_2):
                self._log(
                    logging.DEBUG,
                    "Stage 2 rejected subset=%s by train threshold: %.4f",
                    subset,
                    train_score,
                )
                return None

            cv_score = crossval(
                estimator_copy,
                x,
                self.y_train,
                self.metric,
                self.cv_iter,
                self.task_type,
                **({'cv_indices': self._resolved_cv_indices} if self._resolved_cv_indices is not None else {}),
            ).mean()
            if not is_better(cv_score, self.cv_threshold_2):
                self._log(
                    logging.DEBUG,
                    "Stage 2 rejected subset=%s by cv threshold: %.4f",
                    subset,
                    cv_score,
                )
                return None

            y_test_pred = estimator_copy.predict(self.test_set[subset].values)
            test_score = self.metric(self.y_test, y_test_pred)
            self._log(
                logging.DEBUG,
                "Stage 2 accepted subset=%s | train=%.4f | cv=%.4f | test=%.4f",
                subset,
                train_score,
                cv_score,
                test_score,
            )
            return subset, train_score, cv_score, test_score

        if self.n_jobs == 1:
            for i, subset in enumerate(tqdm(self.feature_subsets, desc="Stage 2", disable=False)):
                result = evaluate_subset(subset)
                if result is None:
                    continue
                subset, train_score, cv_score, test_score = result
                self.dict_results_2['feature_subsets'].append(subset)
                self.dict_results_2['training_score'].append(train_score)
                self.dict_results_2['cv_score'].append(cv_score)
                self.dict_results_2['test_score'].append(test_score)
        else:
            max_workers = self.n_jobs if self.n_jobs > 0 else None
            # Benign: sklearn's own nested cross_val_score Parallel/delayed
            # can misreport propagation when dispatched from raw threads.
            with _suppress_parallel_delayed_warning():
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(evaluate_subset, subset): subset for subset in self.feature_subsets}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Stage 2", disable=False):
                        result = future.result()
                        if result is None:
                            continue
                        subset, train_score, cv_score, test_score = result
                        self.dict_results_2['feature_subsets'].append(subset)
                        self.dict_results_2['training_score'].append(train_score)
                        self.dict_results_2['cv_score'].append(cv_score)
                        self.dict_results_2['test_score'].append(test_score)
        self.df_results_stage2 = pd.DataFrame(
            self.dict_results_2, columns=self.dict_results_2.keys()
            )
        self.df_results_stage2 = _add_reliability_columns(
            self.df_results_stage2,
            self.logic,
            self.selection_strategy,
        )
        self.df_results_stage2.sort_values(
            by='reliability_score',
            ascending=False,
            inplace=True)
        self._log(
            logging.INFO,
            "Combinatorial stage 2 completed: kept_subsets=%d",
            len(self.df_results_stage2),
        )
        return self.df_results_stage2

    def display_best(self, row: int = 1) -> None:
        """
        Display the best feature subset based on the specified row.

        Parameters
        ----------
        row : int, optional
            Row index of the best feature subset to display. Default is 1.

        Returns
        -------
        None

        Notes
        -----
        - Fits the estimator on the selected subset.
        - Displays training, cross-validation, and test scores.
        """

        self.record = self.df_results_stage2.iloc[row - 1]
        self.best_cols = self.record['feature_subsets']

        # Fit the estimator on the best feature subset
        self.estimator.fit(self.train_set[self.best_cols], self.y_train)
        self.y_train_pred = self.estimator.predict(
            self.train_set[self.best_cols]
            )
        self.y_test_pred = self.estimator.predict(
            self.test_set[self.best_cols]
            )

        # Perform cross-validation
        self.cv_performance = crossval(
            self.estimator,
            self.train_set[self.best_cols],
            self.y_train,
            self.metric,
            5,
            self.task_type,
            **({'cv_indices': self._resolved_cv_indices} if getattr(self, '_resolved_cv_indices', None) is not None else {}),
        )

        # Display results through logger
        self._log(logging.INFO, '# of Features: %d', len(self.best_cols))
        self._log(logging.INFO, 'Best Features: %s', self.best_cols)
        self._log(
            logging.INFO,
            'Train Score: %.3f',
            self.metric(self.y_train, self.y_train_pred),
        )
        self._log(
            logging.INFO,
            'CV Score: %.3f +- %.3f',
            self.cv_performance.mean(),
            self.cv_performance.std(),
        )
        self._log(
            logging.INFO,
            'Test Score: %.3f',
            self.metric(self.y_test, self.y_test_pred),
        )
