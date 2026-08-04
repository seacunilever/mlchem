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


def crossval(estimator,
             X: np.ndarray | pd.DataFrame,
             y: np.ndarray | pd.DataFrame,
             metric_function: Callable,
             n_fold: int = 5,
             task_type: Literal['classification',
                                'regression'] = 'classification',
             random_state: int | None = None,
             shuffle: bool = False
             ) -> np.ndarray:
    """
Evaluate an estimator using cross-validation.

This function performs K-fold cross-validation on the given dataset using
the specified estimator and metric function. It supports both classification
and regression tasks.

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
    leave-one-out cross-validation.

task_type : {'classification', 'regression'}, optional (default='classification')
    Type of task to determine the cross-validation strategy.

random_state : int or None, optional (default=None)
    Random seed for reproducibility.

shuffle : bool, optional (default=False)
    Whether to shuffle samples before splitting into batches.

Returns
-------
numpy.ndarray
    An array of cross-validation scores.
"""

    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer

    validate_task_type(task_type)

    cv_kwargs = {
        'n_splits': n_fold,
        'shuffle': shuffle,
    }
    if shuffle:
        cv_kwargs['random_state'] = random_state

    if task_type == 'classification':

        from sklearn.model_selection import StratifiedKFold
        return cross_val_score(estimator,
                                X,
                                y,
                                cv=StratifiedKFold(**cv_kwargs),
                                scoring=make_scorer(metric_function))
    else:
        from sklearn.model_selection import KFold
    return cross_val_score(estimator,
                            X,
                            y,
                            cv=KFold(**cv_kwargs),
                            scoring=make_scorer(metric_function))


def y_scrambling(estimator,
                 train_set: np.ndarray | pd.DataFrame,
                 y_train: Iterable,
                 test_set: np.ndarray | pd.DataFrame,
                 y_test: Iterable,
                 metric_function: Callable,
                 n_iter: int,
                 plot: bool = True,
                 n_jobs: int = 1,
                 log_level: int | str | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = logging.INFO) -> None:
    """
Perform y-scrambling to assess model performance due to chance.

This function evaluates the robustness of a model by randomly shuffling
the target variable multiple times and measuring performance on the test set.
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

test_set : numpy.ndarray or pandas.DataFrame
    Testing feature matrix.

y_test : iterable
    Target values for testing.

metric_function : callable
    A scoring function that accepts (y_true, y_pred) as arguments.

n_iter : int
    Number of shuffling iterations.

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
    if isinstance(test_set, pd.DataFrame):
        X_test = test_set.values
    else:
        X_test = test_set

    estimator_copy.fit(X_train, y_train_copy)
    ref_score = metric_function(y_test,
                                estimator_copy.predict(X_test))

    def evaluate_scramble(seed_val):
        """Evaluate model performance on a single scrambled iteration."""
        rng = random.Random(seed_val)
        y_shuffled = y_train_copy.copy()
        rng.shuffle(y_shuffled)
        est = clone(estimator)
        disable_estimator_parallelization(est)  # Prevent nested parallelization warnings
        est.fit(X_train, y_shuffled)
        y_pred = est.predict(X_test)
        return metric_function(y_true=y_test, y_pred=y_pred)

    # Run scrambling iterations in parallel
    scores = []
    if n_jobs == 1:
        # Sequential execution
        for i in tqdm(range(n_iter), desc="Y-scrambling", disable=False):
            scores.append(evaluate_scramble(i))
    else:
        # Parallel execution with ThreadPoolExecutor
        max_workers = n_jobs if n_jobs > 0 else None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_scramble, i): i for i in range(n_iter)}
            for future in tqdm(as_completed(futures), total=n_iter, desc="Y-scrambling", disable=False):
                scores.append(future.result())

    scores = np.array(scores)
    ys_max = max(scores)
    ys_std = np.std(scores)

    value = len(
       scores[scores >= (ref_score)]
       )/len(scores)

    obtained_margin = ref_score-ys_max

    # Rucker et al, https://doi.org/10.1021/ci700157b
    safety_margin = 2.3 * ys_std

    logger.log(
        resolved_log_level,
        'Probability to obtain a better model by chance: %.3f',
        value,
    )
    logger.log(
        resolved_log_level,
        'Safety margin: %.2f',
        (obtained_margin / (2.3 * ys_std)),
    )

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
