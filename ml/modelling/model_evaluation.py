from typing import Literal, Callable, Iterable
import pandas as pd
import numpy as np


def crossval(estimator,
             X: np.ndarray | pd.DataFrame,
             y: np.ndarray | pd.DataFrame,
             metric_function: Callable,
             n_fold: int = 5,
             task_type: Literal['classification',
                                'regression'] = 'classification',
             random_state: int | None = None
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

Returns
-------
numpy.ndarray
    An array of cross-validation scores.
"""

    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer

    if task_type == 'classification':

        from sklearn.model_selection import StratifiedKFold
        return cross_val_score(estimator,
                                X,
                                y,
                                cv=StratifiedKFold(n_fold,
                                                random_state=random_state),
                                scoring=make_scorer(metric_function))
    else:
        from sklearn.model_selection import KFold
    return cross_val_score(estimator,
                            X,
                            y,
                            cv=KFold(n_fold,
                                     random_state=random_state),
                            scoring=make_scorer(metric_function))


def y_scrambling(estimator,
                 train_set: np.ndarray | pd.DataFrame,
                 y_train: Iterable,
                 test_set: np.ndarray | pd.DataFrame,
                 y_test: Iterable,
                 metric_function: Callable,
                 n_iter: int,
                 plot: bool = True) -> None:
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

Returns
-------
None
"""

    from sklearn.base import clone
    import random
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    estimator_copy = clone(estimator)
    y_train_copy = y_train.copy()
    if isinstance(train_set, pd.DataFrame):
        X_train = train_set.values
    else:
        X_train = train_set
    if isinstance(test_set, pd.DataFrame):
        X_test = test_set.values
    else:
        X_test = test_set

    estimator_copy.fit(X_train, y_train)
    ref_score = metric_function(y_test,
                                estimator_copy.predict(X_test))

    scores = []
    for i in range(n_iter):
        random.seed(i)
        random.shuffle(y_train_copy)
        estimator_copy.fit(X_train, y_train_copy)
        y_pred_bootstrap = estimator_copy.predict(X_test)
        scores.append(
           metric_function(y_true=y_test,
                           y_pred=y_pred_bootstrap)
                           )

    scores = np.array(scores)
    ys_max = max(scores)
    ys_std = np.std(scores)

    value = len(
       scores[scores >= (ref_score)]
       )/len(scores)

    obtained_margin = ref_score-ys_max

    # Rucker et al, https://doi.org/10.1021/ci700157b
    safety_margin = 2.3 * ys_std

    print(f'Probability to obtain a better model by chance: {value:.3f}')
    print(f'Safety margin: {(obtained_margin / (2.3 * ys_std)):.2f}')

    if plot:
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
estimator_list, column_list, estimator_names=[])

Ensemble model using majority voting (for classification) or averaging (for regression).

This class combines predictions from multiple estimators to 
improve model performance and robustness by leveraging the strengths of
different models.

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
        estimator_names: list[str] = []
         ) -> None:

        self.task_type = task_type
        self.estimator_list = estimator_list
        self.estimator_names = estimator_names
        self.column_list = column_list
        self.train_set = train_set
        self.test_set = test_set
        self.y_train = y_train
        self.y_test = y_test

    def fit(self) -> None:
        """
Fit the estimators on the training data and store predictions.

For classification tasks, both hard (class labels) and soft (probabilities)
predictions are stored. For regression tasks, predicted values are stored.

Returns
-------
None
"""

        if self.task_type == 'classification':
            self.df_train_predictions_hard = pd.\
                DataFrame(index=self.train_set.index)
            self.df_test_predictions_hard = pd.\
                DataFrame(index=self.test_set.index)

            self.df_train_predictions_soft = pd.\
                DataFrame(index=self.train_set.index)
            self.df_test_predictions_soft = pd.\
                DataFrame(index=self.test_set.index)

            for estimator, columns in zip(self.estimator_list,
                            self.column_list):
                X_train = self.train_set[columns]
                X_train = X_train.loc[:, ~X_train.columns.
                                      duplicated(keep='first')].copy()
                X_test = self.test_set[columns]
                X_test = X_test.loc[:, ~X_test.columns.
                                    duplicated(keep='first')].copy()
                if len(self.estimator_names) > 0:
                    i = self.estimator_list.index(estimator)
                    estimator_name = self.estimator_names[i]
                else:
                    estimator_name = str(estimator)
                try:
                    estimator.fit(X_train, self.y_train)
                except Exception as ex:
                    print
                    (f"Problem encountered with the estimator {estimator}: {ex}")

                self.df_train_predictions_hard[estimator_name] = estimator.\
                    predict(X_train)
                self.df_test_predictions_hard[estimator_name] = estimator.\
                    predict(X_test)

                self.df_train_predictions_soft[estimator_name] = estimator.\
                    predict_proba(X_train)[:, 1]
                self.df_test_predictions_soft[estimator_name] = estimator.\
                    predict_proba(X_test)[:, 1]

            self.df_train_predictions_hard['Y'] = self.y_train
            self.df_test_predictions_hard['Y'] = self.y_test

            self.df_train_predictions_soft['Y'] = self.y_train
            self.df_test_predictions_soft['Y'] = self.y_test

        else:     # if regression
            self.df_train_predictions = pd.DataFrame(index=self.train_set.index)
            self.df_test_predictions = pd.DataFrame(index=self.test_set.index)

            for estimator, columns in zip(self.estimator_list, self.column_list):
                X_train = self.train_set[columns]
                X_train = X_train.loc[:, ~X_train.columns.
                                      duplicated(keep='first')].copy()
                X_test = self.test_set[columns]
                X_test = X_test.loc[:, ~X_test.columns.
                                    duplicated(keep='first')].copy()
                if len(self.estimator_names) > 0:
                    i = self.estimator_list.index(estimator)
                    estimator_name = self.estimator_names[i]
                else:
                    estimator_name = str(estimator)
                try:
                    estimator.fit(X_train, self.y_train)
                except Exception as exc:
                    (f"Problem encountered with the estimator {estimator}: {exc}")

                self.df_train_predictions[estimator_name] = estimator.\
                    predict(X_train)
                self.df_test_predictions[estimator_name] = estimator.\
                    predict(X_test)

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
combination of estimators up to a specified maximum.

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
            dataframe_probe = dataframe[individual_ys].values
            if hard:
                from scipy.stats import mode
                return mode(dataframe_probe, axis=1)[0]
            else:
                return np.round(dataframe_probe.mean(axis=1))

        def extract_from(results, models):
            """
            Extract scores from the results dictionary for the given
            models.
            """
            return [results[model] for model in models]

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

        dict_results_train = {}
        dict_results_test = {}
        for combination in self.combinations:
            if self.task_type == 'classification':
                dict_results_train[f'{combination}_soft_{metric_name}'] = \
                    metric(self.df_train_predictions_soft.Y.values,
                           majority_vote(self.df_train_predictions_soft,
                                         combination,
                                         hard=False)
                           )
                dict_results_train[f'{combination}_hard_{metric_name}'] = \
                    metric(self.df_train_predictions_hard.Y.values,
                           majority_vote(
                               self.df_train_predictions_hard,
                               combination,
                               hard=True)
                           )

                dict_results_test[f'{combination}_soft_{metric_name}'] = \
                    metric(self.df_test_predictions_soft.Y.values,
                           majority_vote(
                               self.df_test_predictions_soft,
                               combination,
                               hard=False)
                           )
                dict_results_test[f'{combination}_hard_{metric_name}'] = \
                    metric(self.df_test_predictions_hard.Y.values,
                           majority_vote(
                               self.df_test_predictions_hard,
                               combination,
                               hard=True)
                           )
            else:
                dict_results_train[f'{combination}_{metric_name}'] = \
                    metric(self.df_train_predictions.Y.values,
                           majority_vote(self.df_train_predictions,
                                         combination,
                                         hard=False)
                           )
                dict_results_test[f'{combination}_{metric_name}'] = \
                    metric(self.df_test_predictions.Y.values,
                           majority_vote(self.df_test_predictions,
                                         combination,
                                         hard=False)
                           )

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
    - 'leverages': list of float
        Leverage values for each data point.
    - 'results': list of bool
        Boolean flags indicating whether each point is within the domain.
    - 'threshold': float
        Threshold used to determine domain inclusion.
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
