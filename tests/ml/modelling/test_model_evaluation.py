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
import logging
import warnings
from collections import UserList
from itertools import combinations
from typing import Literal
from unittest.mock import patch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from mlchem.ml.modelling.model_evaluation import (crossval,
                                                  y_scrambling,
                                                  ApplicabilityDomain,
                                                  MajorityVote,
                                                  validate_cv_indices,
                                                  generate_cv_indices)
from mlchem.metrics import get_geometric_S, get_rmse


class FailingEstimator:
    def fit(self, X, y):
        raise RuntimeError("Intentional fit failure")


class FailingRegressor:
    def fit(self, X, y):
        raise RuntimeError("Intentional fit failure")


class NoProbaEstimator:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class _FastPredictionEstimator(BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.25, 0.75], (len(X), 1))


@pytest.fixture
def sample_data():
    X, y = make_classification(100, 10, n_informative=5)
    train_size = 0.8
    train_samples = int(train_size * len(X))

    X_train, y_train = X[:train_samples], y[:train_samples]
    X_test, y_test = X[train_samples:], y[train_samples:]

    train_set = pd.DataFrame(X_train, columns=np.arange(X_train.shape[1]))
    test_set = pd.DataFrame(X_test, columns=np.arange(X_test.shape[1]))

    return train_set, y_train, test_set, y_test

def test_crossval_classification(sample_data):
    train_set, y_train, _, _ = sample_data
    estimator = LogisticRegression()
    metric_function = lambda y_true, y_pred: (y_true == y_pred).mean()
    
    scores = crossval(estimator, train_set.values, y_train, metric_function, n_fold=5, task_type='classification')
    assert isinstance(scores, dict)
    assert set(['train_scores', 'train_mean', 'train_se', 'cv_scores', 'cv_mean', 'cv_se']).issubset(scores.keys())
    assert len(scores['cv_scores']) == 5


def test_crossval_default_response_method_matches_predict(sample_data):
    train_set, y_train, _, _ = sample_data
    metric_function = lambda y_true, y_pred: (y_true == y_pred).mean()

    implicit_default = crossval(
        LogisticRegression(),
        train_set.values,
        y_train,
        metric_function,
        n_fold=4,
    )
    explicit_predict = crossval(
        LogisticRegression(),
        train_set.values,
        y_train,
        metric_function,
        n_fold=4,
        response_method='predict',
    )

    np.testing.assert_array_equal(
        implicit_default['train_scores'], explicit_predict['train_scores']
    )
    np.testing.assert_array_equal(
        implicit_default['cv_scores'], explicit_predict['cv_scores']
    )


@pytest.mark.parametrize(
    'response_method',
    ['predict_proba', 'decision_function'],
)
def test_crossval_response_method_supports_binary_auc(
    sample_data,
    response_method,
):
    train_set, y_train, _, _ = sample_data
    string_labels = np.where(y_train == 1, 'positive', 'negative')

    scores = crossval(
        LogisticRegression(max_iter=500),
        train_set.values,
        string_labels,
        roc_auc_score,
        n_fold=4,
        response_method=response_method,
    )

    assert 0.5 <= scores['cv_mean'] <= 1.0
    assert np.isfinite(scores['train_scores']).all()


def test_crossval_probability_response_method_preserves_multiclass_scores():
    X, y = make_classification(
        n_samples=90,
        n_features=6,
        n_informative=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=5,
    )
    multiclass_auc = lambda y_true, y_score: roc_auc_score(
        y_true, y_score, multi_class='ovr'
    )

    scores = crossval(
        LogisticRegression(max_iter=500),
        X,
        y,
        multiclass_auc,
        n_fold=3,
        response_method='predict_proba',
    )

    assert 0.0 <= scores['cv_mean'] <= 1.0

def test_crossval_regression(sample_data):
    train_set, y_train, _, _ = sample_data
    estimator = LinearRegression()
    y_train_reg = y_train.astype(float)
    metric_function = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))

    scores = crossval(estimator, train_set.values, y_train_reg, metric_function, n_fold=5, task_type='regression')
    assert isinstance(scores, dict)
    assert len(scores['cv_scores']) == 5


def test_crossval_invalid_task_type_raises(sample_data):
    train_set, y_train, _, _ = sample_data
    estimator = LogisticRegression()
    metric_function = lambda y_true, y_pred: (y_true == y_pred).mean()

    with pytest.raises(ValueError, match="must be either 'classification' or 'regression'"):
        crossval(
            estimator,
            train_set.values,
            y_train,
            metric_function,
            n_fold=5,
            task_type='invalid',
        )


@pytest.mark.parametrize('task_type, estimator, y_values', [
    ('classification', LogisticRegression(), 'classification'),
    ('regression', LinearRegression(), 'regression'),
])
def test_crossval_shuffle_false_ignores_random_state(sample_data, task_type, estimator, y_values):
    train_set, y_train, _, _ = sample_data
    metric_function = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    y_input = y_train if y_values == 'classification' else y_train.astype(float)

    # Must not raise when random_state is provided with shuffle disabled.
    scores = crossval(
        estimator,
        train_set.values,
        y_input,
        metric_function,
        n_fold=5,
        task_type=task_type,
        random_state=123,
        shuffle=False,
    )

    assert isinstance(scores, dict)
    assert len(scores['cv_scores']) == 5


@pytest.mark.parametrize('task_type, estimator, y_values', [
    ('classification', LogisticRegression(), 'classification'),
    ('regression', LinearRegression(), 'regression'),
])
def test_crossval_shuffle_true_propagates_random_state(sample_data, task_type, estimator, y_values):
    train_set, y_train, _, _ = sample_data
    metric_function = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    y_input = y_train if y_values == 'classification' else y_train.astype(float)

    fake_result = {
        'train_score': np.array([1.0]),
        'test_score': np.array([1.0]),
    }
    with patch('sklearn.model_selection.cross_validate', return_value=fake_result) as mock_cross_validate:
        crossval(
            estimator,
            train_set.values,
            y_input,
            metric_function,
            n_fold=5,
            task_type=task_type,
            random_state=77,
            shuffle=True,
        )

    cv_splitter = mock_cross_validate.call_args.kwargs['cv']
    assert cv_splitter.shuffle is True
    assert cv_splitter.random_state == 77


# ---------------------------------------------------------------------------
# Index-driven cross-validation: validate_cv_indices / generate_cv_indices
# ---------------------------------------------------------------------------

def test_validate_cv_indices_accepts_well_formed_folds():
    folds = [
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
        (np.array([4, 5, 2, 3]), np.array([0, 1])),
    ]
    normalised = validate_cv_indices(folds, n_samples=6)
    assert len(normalised) == 2
    for train_idx, valid_idx in normalised:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(valid_idx, np.ndarray)


def test_validate_cv_indices_rejects_overlap():
    folds = [(np.array([0, 1, 2]), np.array([2, 3]))]
    with pytest.raises(ValueError, match='overlap'):
        validate_cv_indices(folds, n_samples=4)


def test_validate_cv_indices_rejects_out_of_range():
    folds = [(np.array([0, 1]), np.array([10]))]
    with pytest.raises(ValueError, match='out of range'):
        validate_cv_indices(folds, n_samples=4)


def test_validate_cv_indices_rejects_empty_fold():
    folds = [(np.array([], dtype=int), np.array([0, 1]))]
    with pytest.raises(ValueError, match='empty'):
        validate_cv_indices(folds, n_samples=4)


def test_validate_cv_indices_rejects_empty_manifest():
    with pytest.raises(ValueError, match='at least one'):
        validate_cv_indices([], n_samples=4)


def test_validate_cv_indices_warns_on_missing_coverage():
    folds = [(np.array([0, 1]), np.array([2]))]
    with pytest.warns(RuntimeWarning, match='never use'):
        validate_cv_indices(folds, n_samples=4)


def test_validate_cv_indices_strict_coverage_raises():
    folds = [(np.array([0, 1]), np.array([2]))]
    with pytest.raises(ValueError, match='never use'):
        validate_cv_indices(folds, n_samples=4, strict_coverage=True)


def test_generate_cv_indices_legacy_cv_iter(sample_data):
    train_set, y_train, _, _ = sample_data
    pairs = generate_cv_indices(
        train_set.values, y=y_train, cv_iter=5, task_type='classification',
    )
    assert len(pairs) == 5
    for train_idx, valid_idx in pairs:
        assert set(train_idx.tolist()).isdisjoint(valid_idx.tolist())


def test_generate_cv_indices_with_group_kfold(sample_data):
    from sklearn.model_selection import GroupKFold

    train_set, y_train, _, _ = sample_data
    groups = np.arange(len(train_set)) % 4  # 4 distinct groups

    pairs = generate_cv_indices(
        train_set.values,
        y=y_train,
        cv_splitter=GroupKFold(n_splits=4),
        groups=groups,
    )
    assert len(pairs) == 4
    for train_idx, valid_idx in pairs:
        train_groups = set(groups[train_idx].tolist())
        valid_groups = set(groups[valid_idx].tolist())
        assert train_groups.isdisjoint(valid_groups)


def test_generate_cv_indices_with_explicit_manifest(sample_data):
    train_set, y_train, _, _ = sample_data
    n = len(train_set)
    manifest = [
        (np.arange(n // 2, n), np.arange(0, n // 2)),
        (np.arange(0, n // 2), np.arange(n // 2, n)),
    ]
    pairs = generate_cv_indices(train_set.values, cv_indices=manifest)
    assert len(pairs) == 2


def test_crossval_cv_splitter_matches_cv_indices_manifest(sample_data):
    train_set, y_train, _, _ = sample_data
    estimator = LogisticRegression()
    metric_function = lambda y_true, y_pred: (y_true == y_pred).mean()

    pairs = generate_cv_indices(
        train_set.values, y=y_train, cv_iter=5, task_type='classification',
    )

    scores_from_indices = crossval(
        estimator, train_set.values, y_train, metric_function,
        cv_indices=pairs,
    )

    from sklearn.model_selection import StratifiedKFold
    scores_from_splitter = crossval(
        estimator, train_set.values, y_train, metric_function,
        cv_splitter=StratifiedKFold(n_splits=5),
    )

    np.testing.assert_allclose(scores_from_indices['cv_scores'], scores_from_splitter['cv_scores'])


def test_crossval_cv_iter_and_cv_indices_produce_identical_scores(sample_data):
    train_set, y_train, _, _ = sample_data
    estimator = LogisticRegression()
    metric_function = lambda y_true, y_pred: (y_true == y_pred).mean()

    legacy_scores = crossval(
        estimator, train_set.values, y_train, metric_function,
        n_fold=5, task_type='classification',
    )

    pairs = generate_cv_indices(
        train_set.values, y=y_train, cv_iter=5, task_type='classification',
    )
    explicit_scores = crossval(
        estimator, train_set.values, y_train, metric_function,
        cv_indices=pairs,
    )

    np.testing.assert_allclose(legacy_scores['cv_scores'], explicit_scores['cv_scores'])


def test_crossval_with_group_kfold_and_groups(sample_data):
    train_set, y_train, _, _ = sample_data
    estimator = LogisticRegression()
    metric_function = lambda y_true, y_pred: (y_true == y_pred).mean()
    groups = np.arange(len(train_set)) % 4

    from sklearn.model_selection import GroupKFold
    scores = crossval(
        estimator, train_set.values, y_train, metric_function,
        cv_splitter=GroupKFold(n_splits=4), groups=groups,
    )
    assert isinstance(scores, dict)
    assert len(scores['cv_scores']) == 4


def test_crossval_with_predefined_split(sample_data):
    train_set, y_train, _, _ = sample_data
    estimator = LogisticRegression()
    metric_function = lambda y_true, y_pred: (y_true == y_pred).mean()

    from sklearn.model_selection import PredefinedSplit
    test_fold = np.resize([0, 1], len(train_set))
    scores = crossval(
        estimator, train_set.values, y_train, metric_function,
        cv_splitter=PredefinedSplit(test_fold),
    )
    assert isinstance(scores, dict)
    assert len(scores['cv_scores']) == 2

def test_y_scrambling(sample_data, tmp_path):
    train_set, y_train, _, _ = sample_data
    estimator = LogisticRegression()
    metric_function = get_geometric_S
    
    with pytest.raises(ValueError, match="'safety_multiplier' must be greater than 0"):
        y_scrambling(
            estimator,
            train_set.values,
            y_train,
            metric_function,
            n_scrambles=5,
            safety_multiplier=0,
            plot=False,
        )

    # Test with valid number of iterations
    with patch('matplotlib.pyplot.show') as mock_show:
            y_scrambling(estimator, train_set.values, y_train, metric_function, n_scrambles=10)
            mock_show.assert_called_once()  # Ensure plt.show() is called
            result = y_scrambling(estimator, train_set.values, y_train, metric_function, n_scrambles=20, plot=False)
            output_file = tmp_path / 'y_scrambling_test_plot.png'
            plt.savefig(output_file)
            assert output_file.exists()
            assert 'safety_margin_ratio' in result

def test_y_scrambling_with_dataframe_inputs(sample_data):
    train_set, y_train, _, _ = sample_data
    estimator = LogisticRegression(max_iter=250)
    metric_function = get_geometric_S

    # Exercise DataFrame input branch in y_scrambling conversion logic.
    result = y_scrambling(
        estimator,
        train_set,
        y_train,
        metric_function,
        n_scrambles=2,
        plot=False,
    )
    assert isinstance(result, dict)
    assert 'scrambled_scores' in result


def test_y_scrambling_propagates_response_method_to_reference_and_scrambles(
    sample_data,
):
    train_set, y_train, _, _ = sample_data
    response_methods = []

    def fake_crossval(*args, **kwargs):
        response_methods.append(kwargs.get('response_method'))
        return {'cv_mean': 0.8 if len(response_methods) == 1 else 0.4}

    with patch(
        'mlchem.ml.modelling.model_evaluation.crossval',
        side_effect=fake_crossval,
    ):
        y_scrambling(
            LogisticRegression(),
            train_set.values,
            y_train,
            roc_auc_score,
            n_scrambles=1,
            response_method='predict_proba',
            plot=False,
        )

    assert response_methods == ['predict_proba', 'predict_proba']


def test_y_scrambling_disables_parallelism_on_each_scramble_clone(sample_data):
    train_set, y_train, _, _ = sample_data
    observed_estimators = []

    def fake_crossval(estimator, *args, **kwargs):
        observed_estimators.append(estimator)
        return {'cv_mean': 0.8 if estimator.n_jobs == 4 else 0.4}

    with patch(
        'mlchem.ml.modelling.model_evaluation.crossval',
        side_effect=fake_crossval,
    ):
        y_scrambling(
            RandomForestClassifier(n_estimators=5, n_jobs=4),
            train_set.values,
            y_train,
            roc_auc_score,
            n_scrambles=2,
            n_jobs=2,
            plot=False,
        )

    reference_estimator, *scramble_estimators = observed_estimators
    assert reference_estimator.n_jobs == 4
    assert len(scramble_estimators) == 2
    assert all(estimator is not reference_estimator for estimator in scramble_estimators)
    assert len({id(estimator) for estimator in scramble_estimators}) == 2
    assert all(estimator.n_jobs in (None, 1) for estimator in scramble_estimators)


def test_y_scrambling_logging_toggle(sample_data, caplog):
    train_set, y_train, _, _ = sample_data
    estimator = LogisticRegression(max_iter=250)
    metric_function = get_geometric_S

    with caplog.at_level(logging.INFO, logger='mlchem.ml.modelling.model_evaluation'):
        y_scrambling(
            estimator,
            train_set,
            y_train,
            metric_function,
            n_scrambles=2,
            plot=False,
            log_level='INFO',
        )
    full_text = '\n'.join(caplog.messages)
    assert 'Probability to obtain a better model by chance' in full_text
    assert 'Safety margin:' in full_text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='mlchem.ml.modelling.model_evaluation'):
        y_scrambling(
            estimator,
            train_set,
            y_train,
            metric_function,
            n_scrambles=2,
            plot=False,
            log_level='WARNING',
        )
    # At WARNING level, the same diagnostics are emitted at WARNING severity.
    assert len([m for m in caplog.messages if 'Probability' in m]) == 1


def test_leverage():
    # Test with a simple dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    result = ApplicabilityDomain.leverage(X)
    assert 'leverages' in result
    assert 'results' in result
    assert 'threshold' in result
    assert isinstance(result['leverages'], list)
    assert isinstance(result['results'], list)
    assert isinstance(result['threshold'], float)

    # Test with a larger dataset
    X = np.random.rand(100, 10)
    result = ApplicabilityDomain.leverage(X)
    assert len(result['leverages']) == 100
    assert len(result['results']) == 100

    # Test with edge case: single data point
    X = np.array([[1, 2, 3, 4, 5]])
    result = ApplicabilityDomain.leverage(X)
    assert len(result['leverages']) == 1
    assert len(result['results']) == 1


@pytest.fixture
def majority_vote_classification(sample_data):
    train_set, y_train, test_set, y_test = sample_data
    est_1 = LogisticRegression(random_state=1)
    est_2 = RandomForestClassifier(random_state=1)
    est_1.fit(train_set,y_train)
    est_2.fit(train_set,y_train)
    estimator_list = [est_1,est_2]
    column_list = [train_set.columns.tolist(), train_set.columns.tolist()]
    estimator_names = ['LR', 'RF']
    
    mv = MajorityVote(train_set=train_set,
                      test_set=test_set,
                      y_train=y_train,
                      y_test=y_test,
                      task_type='classification',
                      estimator_list=estimator_list,
                      column_list=column_list,
                      estimator_names=estimator_names)
    
    return mv

def test_majority_vote_init(majority_vote_classification):
    mv = majority_vote_classification
    assert mv.task_type == 'classification'
    assert len(mv.estimator_list) == 2
    assert mv.estimator_names == ['LR', 'RF']
    assert len(mv.column_list) == 2
    assert isinstance(mv.train_set, pd.DataFrame)
    assert isinstance(mv.test_set, pd.DataFrame)
    assert isinstance(mv.y_train, np.ndarray)
    assert isinstance(mv.y_test, np.ndarray)


@pytest.mark.parametrize('missing_from', ['train_set', 'test_set'])
def test_majority_vote_fit_rejects_missing_estimator_columns(
    sample_data, missing_from
):
    train_set, y_train, test_set, y_test = sample_data
    if missing_from == 'train_set':
        columns = ['missing_feature']
    else:
        columns = [train_set.columns[0]]
        test_set = test_set.drop(columns=columns)

    mv = MajorityVote(
        train_set=train_set,
        test_set=test_set,
        y_train=y_train,
        y_test=y_test,
        task_type='classification',
        estimator_list=[LogisticRegression()],
        column_list=[columns],
    )

    with pytest.raises(
        KeyError, match=f'Columns not found in {missing_from}'
    ):
        mv.fit()


def test_majority_vote_default_estimator_names_are_isolated(sample_data):
    train_set, y_train, test_set, y_test = sample_data
    estimator_list = [LogisticRegression(random_state=1)]
    column_list = [train_set.columns.tolist()]

    mv_a = MajorityVote(
        train_set=train_set,
        test_set=test_set,
        y_train=y_train,
        y_test=y_test,
        task_type='classification',
        estimator_list=estimator_list,
        column_list=column_list,
    )
    mv_b = MajorityVote(
        train_set=train_set,
        test_set=test_set,
        y_train=y_train,
        y_test=y_test,
        task_type='classification',
        estimator_list=estimator_list,
        column_list=column_list,
    )

    mv_a.estimator_names.append('mutated')

    assert mv_b.estimator_names == []


def test_majority_vote_copies_estimator_names_input(sample_data):
    train_set, y_train, test_set, y_test = sample_data
    estimator_names = ['lr']

    mv = MajorityVote(
        train_set=train_set,
        test_set=test_set,
        y_train=y_train,
        y_test=y_test,
        task_type='classification',
        estimator_list=[LogisticRegression(random_state=1)],
        column_list=[train_set.columns.tolist()],
        estimator_names=estimator_names,
    )

    estimator_names.append('external-mutation')

    assert mv.estimator_names == ['lr']

def test_majority_vote_fit(majority_vote_classification):
    mv = majority_vote_classification
    mv.fit()
    
    assert not mv.df_train_predictions_hard.empty
    assert not mv.df_test_predictions_hard.empty
    assert not mv.df_train_predictions_soft.empty
    assert not mv.df_test_predictions_soft.empty


@pytest.mark.parametrize(
    ('task_type', 'n_estimators'),
    [('classification', 55), ('regression', 105)],
)
def test_majority_vote_fit_builds_prediction_frames_without_fragmentation(
    task_type, n_estimators
):
    train_set = pd.DataFrame({'feature': np.arange(12)})
    test_set = pd.DataFrame({'feature': np.arange(12, 18)})
    if task_type == 'classification':
        y_train = np.array([0, 1] * 6)
        y_test = np.array([0, 1] * 3)
    else:
        y_train = np.arange(12)
        y_test = np.arange(12, 18)

    mv = MajorityVote(
        train_set=train_set,
        test_set=test_set,
        y_train=y_train,
        y_test=y_test,
        task_type=task_type,
        estimator_list=[_FastPredictionEstimator() for _ in range(n_estimators)],
        column_list=[['feature'] for _ in range(n_estimators)],
        n_jobs=1,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', pd.errors.PerformanceWarning)
        mv.fit()

    assert not any(
        issubclass(warning.category, pd.errors.PerformanceWarning)
        for warning in caught
    )
    if task_type == 'classification':
        assert len(mv.df_train_predictions) == len(train_set)
        assert len(mv.df_train_predictions.columns) == 2 * n_estimators + 3
    else:
        assert len(mv.df_train_predictions) == len(train_set)
        assert len(mv.df_train_predictions.columns) == n_estimators + 1


def test_majority_vote_fit_skips_failed_classification_estimator(sample_data):
    train_set, y_train, test_set, y_test = sample_data
    good_estimator = LogisticRegression(random_state=1)
    bad_estimator = FailingEstimator()

    mv = MajorityVote(
        train_set=train_set,
        test_set=test_set,
        y_train=y_train,
        y_test=y_test,
        task_type='classification',
        estimator_list=[good_estimator, bad_estimator],
        column_list=[train_set.columns.tolist(), train_set.columns.tolist()],
        estimator_names=['good_lr', 'bad_fit']
    )

    with pytest.warns(RuntimeWarning, match="Skipping estimator 'bad_fit'"):
        mv.fit()

    assert 'good_lr' in mv.df_train_predictions_hard.columns
    assert 'bad_fit' not in mv.df_train_predictions_hard.columns


def test_majority_vote_fit_raises_when_all_classification_estimators_fail(sample_data):
    train_set, y_train, test_set, y_test = sample_data

    mv = MajorityVote(
        train_set=train_set,
        test_set=test_set,
        y_train=y_train,
        y_test=y_test,
        task_type='classification',
        estimator_list=[FailingEstimator()],
        column_list=[train_set.columns.tolist()],
        estimator_names=['bad_fit']
    )

    with pytest.warns(RuntimeWarning):
        with pytest.raises(RuntimeError, match="No estimators were successfully fitted"):
            mv.fit()


def test_majority_vote_fit_skips_estimator_without_predict_proba(sample_data):
    train_set, y_train, test_set, y_test = sample_data
    good_estimator = LogisticRegression(random_state=1)
    no_proba = NoProbaEstimator()

    mv = MajorityVote(
        train_set=train_set,
        test_set=test_set,
        y_train=y_train,
        y_test=y_test,
        task_type='classification',
        estimator_list=[good_estimator, no_proba],
        column_list=[train_set.columns.tolist(), train_set.columns.tolist()],
        estimator_names=['good_lr', 'no_proba']
    )

    with pytest.warns(RuntimeWarning, match="Skipping estimator 'no_proba'"):
        mv.fit()

    assert 'good_lr' in mv.df_train_predictions_soft.columns
    assert 'no_proba' not in mv.df_train_predictions_soft.columns

def test_majority_vote_fit_returns_consensus_report_classification(majority_vote_classification):
    mv = majority_vote_classification
    result = mv.fit(return_prediction_dataframes=True)

    assert 'fit_report' in result
    assert 'train_predictions' in result
    assert 'test_predictions' in result
    assert result['fit_report']['successful_count'] >= 1


def test_majority_vote_evaluate_cv_classification_outputs_fold_metrics(
    majority_vote_classification,
):
    mv = majority_vote_classification
    metric_function = lambda y_true, y_pred: (y_true == y_pred).mean()

    results = mv.evaluate_cv(
        metric=metric_function,
        metric_name='accuracy',
        cv_iter=4,
        logic='greater',
    )

    assert not results.empty
    expected_cols = {
        'combination',
        'strategy',
        'accuracy_train_mean',
        'accuracy_train_se',
        'accuracy_validation_mean',
        'accuracy_validation_se',
        'accuracy_reliability_mean',
        'accuracy_reliability_se',
        'accuracy_train_folds',
        'accuracy_validation_folds',
        'accuracy_reliability_folds',
    }
    assert expected_cols.issubset(results.columns)
    assert results['accuracy_reliability_mean'].is_monotonic_decreasing
    assert all(len(fold_scores) == 4 for fold_scores in results['accuracy_train_folds'])
    assert all(len(fold_scores) == 4 for fold_scores in results['accuracy_validation_folds'])
    assert all(len(fold_scores) == 4 for fold_scores in results['accuracy_reliability_folds'])


def test_majority_vote_multiple_metrics_reuse_fits_and_match_single_calls():
    X = pd.DataFrame({'x': np.arange(20)})
    y = np.array([0, 1] * 10)
    mv = MajorityVote(
        train_set=X, y_train=y, task_type='classification',
        estimator_list=[DummyClassifier(strategy='prior')],
        column_list=[['x']], estimator_names=['dummy'],
    )
    predictions = {'accuracy': [], 'auc': []}

    def recorded_accuracy(y_true, y_pred):
        predictions['accuracy'].append(np.asarray(y_pred))
        return accuracy_score(y_true, y_pred)

    def recorded_auc(y_true, y_pred):
        predictions['auc'].append(np.asarray(y_pred))
        return roc_auc_score(y_true, y_pred)

    original_fit = DummyClassifier.fit
    with patch.object(DummyClassifier, 'fit', autospec=True, side_effect=original_fit) as fit:
        results = mv.evaluate_cv(
            metric=[recorded_accuracy, recorded_auc],
            metric_name=['accuracy', 'auc'],
            use_continuous_probabilities=[False, True],
            cv_iter=2, n_estimators_max=1,
        )
    assert fit.call_count == 2
    assert all(np.isin(pred, [0, 1]).all() for pred in predictions['accuracy'])
    assert any(np.any((pred > 0) & (pred < 1)) for pred in predictions['auc'])
    assert len(results) == 2
    assert set(results['strategy']) == {'hard', 'soft'}
    assert results['accuracy_reliability_mean'].is_monotonic_decreasing
    assert 'metric' not in results.columns
    for metric, name, continuous in [
        (accuracy_score, 'accuracy', False),
        (roc_auc_score, 'auc', True),
    ]:
        separate = mv.evaluate_cv(
            metric=metric, metric_name=name,
            use_continuous_probabilities=continuous, cv_iter=2, n_estimators_max=1,
        )
        columns = ['combination', 'strategy'] + [
            column for column in results.columns if column.startswith(f'{name}_')
        ]
        combined = results[columns].sort_values('combination').reset_index(drop=True)
        separate = separate[columns].sort_values('combination').reset_index(drop=True)
        pd.testing.assert_frame_equal(combined, separate)


def test_majority_vote_accepts_non_list_sequences():
    X = pd.DataFrame({'x': np.arange(20)})
    mv = MajorityVote(
        train_set=X, y_train=np.array([0, 1] * 10), task_type='classification',
        estimator_list=[DummyClassifier(strategy='prior')],
        column_list=[['x']], estimator_names=['dummy'],
    )
    results = mv.evaluate_cv(
        metric=UserList([accuracy_score, roc_auc_score]),
        metric_name=UserList(['accuracy', 'auc']),
        use_continuous_probabilities=UserList([False, True]),
        logic=UserList[Literal['lower', 'greater']](['greater', 'greater']),
        cv_iter=2, n_estimators_max=1,
    )
    assert len(results) == 2
    assert np.isfinite(results[['accuracy_validation_mean', 'auc_validation_mean']]).all().all()


def test_majority_vote_shortlist_cv_reuses_predictions():
    X = pd.DataFrame({'x': np.arange(20)})
    y = np.array([0, 1] * 10)
    names = ['model_1', 'model_2', 'model_3', 'discard']
    mv = MajorityVote(
        train_set=X, y_train=y, task_type='classification',
        estimator_list=[_FastPredictionEstimator() for _ in names],
        column_list=[['x']] * len(names), estimator_names=names,
    )
    kwargs = {
        'metric': [accuracy_score, roc_auc_score],
        'metric_name': ['accuracy', 'auc'],
        'use_continuous_probabilities': [False, True],
        'cv_iter': 2,
    }
    original_fit = _FastPredictionEstimator.fit
    with patch.object(_FastPredictionEstimator, 'fit', autospec=True, side_effect=original_fit) as fit:
        with patch('mlchem.ml.modelling.model_evaluation.combinations', side_effect=AssertionError):
            individual = mv.evaluate_cv(individual_only=True, **kwargs)
        assert fit.call_count == len(names) * 2
        assert len(individual) == len(names) * 2
        assert set(individual['strategy']) == {'hard', 'soft'}
        for _, row in individual.iterrows():
            for name, metric in [('accuracy', accuracy_score), ('auc', roc_auc_score)]:
                for fold, (train_idx, valid_idx) in enumerate(mv.cv_indices_):
                    train_pred = row[f'{name}_train_predictions_folds'][fold]
                    valid_pred = row[f'{name}_validation_predictions_folds'][fold]
                    assert row[f'{name}_train_folds'][fold] == pytest.approx(metric(y[train_idx], train_pred))
                    assert row[f'{name}_validation_folds'][fold] == pytest.approx(metric(y[valid_idx], valid_pred))

        shortlist = individual.copy()
        shortlist.drop(
            shortlist.index[
                (shortlist['estimator'] == 'discard')
                | (shortlist['strategy'] != 'hard')
                | (shortlist['accuracy_validation_mean'] < 0.5)
                | (shortlist['auc_validation_mean'] < 0.5)
            ],
            inplace=True,
        )
        shortlist.iloc[0]['accuracy_validation_predictions_folds'][0][:] = 99
        consensus = mv.evaluate_consensus_cv(shortlist, n_estimators_max=3)
        assert fit.call_count == len(names) * 2
        assert all('discard' not in label for label in consensus['combination'])
        assert any(all(name in label for name in names[:3]) for label in consensus['combination'])

    reference = MajorityVote(
        train_set=X, y_train=y, task_type='classification',
        estimator_list=[_FastPredictionEstimator() for _ in names[:3]],
        column_list=[['x']] * 3, estimator_names=names[:3],
    ).evaluate_cv(n_estimators_max=3, **kwargs)
    pd.testing.assert_frame_equal(consensus, reference)


def test_majority_vote_shortlist_cv_regression_matches_one_call():
    X = pd.DataFrame({'x': np.arange(6)})
    y = np.array([0., 1., 2., 3., 4., 5.])
    mv = MajorityVote(
        train_set=X, y_train=y, task_type='regression',
        estimator_list=[DummyRegressor(strategy='mean'), DummyRegressor(strategy='median')],
        column_list=[['x'], ['x']], estimator_names=['mean', 'median'],
    )
    kwargs = {
        'metric': mean_absolute_error, 'metric_name': 'mae',
        'logic': 'lower', 'cv_iter': 3,
    }
    individual = mv.evaluate_cv(individual_only=True, **kwargs)
    with patch.object(DummyRegressor, 'fit', side_effect=AssertionError), patch(
        'mlchem.ml.modelling.model_evaluation.combinations', wraps=combinations
    ) as generate:
        consensus = mv.evaluate_consensus_cv(individual, n_estimators_max=2)
    assert [call.args[1] for call in generate.call_args_list] == [1, 2]
    reference = mv.evaluate_cv(n_estimators_max=2, **kwargs)
    pd.testing.assert_frame_equal(consensus, reference)
    assert any('mean' in label and 'median' in label for label in consensus['combination'])


def test_majority_vote_shortlist_cv_rejects_invalid_or_stale_tables():
    X = pd.DataFrame({'x': np.arange(8)})
    mv = MajorityVote(
        train_set=X, y_train=np.array([0, 1] * 4), task_type='classification',
        estimator_list=[_FastPredictionEstimator()], column_list=[['x']],
        estimator_names=['model'],
    )
    with pytest.raises(ValueError, match='individual_only=True'):
        mv.evaluate_consensus_cv(pd.DataFrame())
    kwargs = {'metric': accuracy_score, 'metric_name': 'accuracy', 'cv_iter': 2}
    individual = mv.evaluate_cv(individual_only=True, **kwargs)
    with pytest.raises(ValueError, match='at least one estimator'):
        mv.evaluate_consensus_cv(individual.iloc[:0])
    with pytest.raises(ValueError, match="'estimator' column"):
        mv.evaluate_consensus_cv(individual.drop(columns='estimator'))
    unknown = individual.copy()
    unknown['estimator'] = 'unknown'
    with pytest.raises(ValueError, match='unknown estimator'):
        mv.evaluate_consensus_cv(unknown)
    mv.evaluate_cv(individual_only=True, **kwargs)
    with pytest.raises(ValueError, match='latest individual evaluation'):
        mv.evaluate_consensus_cv(individual)
    mv.evaluate_cv(**kwargs)
    with pytest.raises(ValueError, match='individual_only=True'):
        mv.evaluate_consensus_cv(individual)


@pytest.mark.parametrize('bad_size', [0, -1, 1.5, True, '2', None])
def test_majority_vote_rejects_invalid_combination_sizes_before_fitting(bad_size):
    X = pd.DataFrame({'x': np.arange(8)})
    mv = MajorityVote(
        train_set=X, y_train=np.array([0, 1] * 4), task_type='classification',
        estimator_list=[_FastPredictionEstimator()], column_list=[['x']],
        estimator_names=['model'],
    )
    kwargs = {'metric': accuracy_score, 'metric_name': 'accuracy', 'cv_iter': 2}
    with patch.object(_FastPredictionEstimator, 'fit', side_effect=AssertionError):
        with pytest.raises(ValueError, match='n_estimators_max'):
            mv.evaluate_cv(n_estimators_max=bad_size, **kwargs)

    individual = mv.evaluate_cv(individual_only=True, n_estimators_max=bad_size, **kwargs)
    with pytest.raises(ValueError, match='n_estimators_max'):
        mv.evaluate_consensus_cv(individual, n_estimators_max=bad_size)


def test_majority_vote_rejects_invalid_performance_score_before_fitting():
    X = pd.DataFrame({'x': np.arange(8)})
    mv = MajorityVote(
        train_set=X, y_train=np.array([0, 1] * 4), task_type='classification',
        estimator_list=[_FastPredictionEstimator()], column_list=[['x']],
    )
    with patch.object(_FastPredictionEstimator, 'fit', side_effect=AssertionError):
        with pytest.raises(ValueError, match='desired_performance_score'):
            mv.evaluate_cv(
                metric=accuracy_score, metric_name='accuracy',
                desired_performance_score='invalid',
            )


def test_majority_vote_classification_skips_even_combinations():
    X = pd.DataFrame({'x': np.arange(8)})
    names = ['a', 'b', 'c', 'd']
    mv = MajorityVote(
        train_set=X, y_train=np.array([0, 1] * 4), task_type='classification',
        estimator_list=[_FastPredictionEstimator() for _ in names],
        column_list=[['x']] * len(names), estimator_names=names,
    )
    individual = mv.evaluate_cv(
        metric=accuracy_score, metric_name='accuracy', cv_iter=2, individual_only=True,
    )
    with patch(
        'mlchem.ml.modelling.model_evaluation.combinations', wraps=combinations
    ) as generate:
        results = mv.evaluate_consensus_cv(individual, n_estimators_max=np.int64(5))
    assert [call.args[1] for call in generate.call_args_list] == [1, 3]
    assert len(results) == 2 * (len(names) + 4)
    expected = {
        f"{list(group)}_{strategy}"
        for size in (1, 3)
        for group in combinations(names, size)
        for strategy in ('hard', 'soft')
    }
    assert set(results['combination']) == expected


def test_majority_vote_failed_evaluation_preserves_screening_cache():
    X = pd.DataFrame({'x': np.arange(8)})
    mv = MajorityVote(
        train_set=X, y_train=np.array([0, 1] * 4), task_type='classification',
        estimator_list=[_FastPredictionEstimator()], column_list=[['x']],
        estimator_names=['model'],
    )
    individual = mv.evaluate_cv(
        metric=accuracy_score, metric_name='accuracy', cv_iter=2, individual_only=True,
    )
    original_indices = mv.cv_indices_

    def failing_metric(y_true, y_pred):
        raise RuntimeError('metric failed')

    with pytest.raises(ValueError, match='n_estimators_max'):
        mv.evaluate_cv(metric=accuracy_score, metric_name='accuracy', n_estimators_max=0)
    with pytest.raises(RuntimeError, match='metric failed'):
        mv.evaluate_cv(metric=failing_metric, metric_name='bad', cv_iter=2)

    assert mv.cv_indices_ is original_indices
    assert mv.final_results_cv is individual
    with patch.object(_FastPredictionEstimator, 'fit', side_effect=AssertionError):
        results = mv.evaluate_consensus_cv(individual, n_estimators_max=1)
    assert len(results) == 2


def test_majority_vote_multiple_metrics_support_per_metric_logic():
    X = pd.DataFrame({'x': np.zeros(4)})
    y = np.array([-10.0, -10.0, 10.0, 10.0])
    mv = MajorityVote(
        train_set=X, y_train=y, task_type='regression',
        estimator_list=[DummyRegressor(strategy='constant', constant=0.0),
                        DummyRegressor(strategy='constant', constant=8.0)],
        column_list=[['x'], ['x']], estimator_names=['stable', 'unstable'],
    )
    indices = [(np.array([0, 1]), np.array([2, 3])),
               (np.array([2, 3]), np.array([0, 1]))]
    score = lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred)
    results = mv.evaluate_cv(
        metric=(get_rmse, score), metric_name=('rmse', 'negative_mae'),
        logic=('lower', 'greater'), cv_indices=indices, n_estimators_max=1,
    )
    assert results['rmse_reliability_mean'].is_monotonic_decreasing
    assert results.iloc[0]['combination'] == "['stable']"
    by_negative_mae = results.sort_values('negative_mae_reliability_mean', ascending=False)
    assert by_negative_mae.iloc[0]['combination'] == "['unstable']"
    for metric, name, direction in [
        (get_rmse, 'rmse', 'lower'), (score, 'negative_mae', 'greater'),
    ]:
        separate = mv.evaluate_cv(
            metric=metric, metric_name=name, logic=direction,
            cv_indices=indices, n_estimators_max=1,
        )
        columns = ['combination', 'strategy'] + [
            column for column in results.columns if column.startswith(f'{name}_')
        ]
        combined = results[columns].sort_values('combination').reset_index(drop=True)
        separate = separate[columns].sort_values('combination').reset_index(drop=True)
        pd.testing.assert_frame_equal(combined, separate)


@pytest.mark.parametrize('single_fold', [False, True])
def test_majority_vote_fold_ranges(single_fold):
    X = pd.DataFrame({'x': np.zeros(4)})
    mv = MajorityVote(
        train_set=X, y_train=np.array([0.0, 2.0, 4.0, 8.0]),
        task_type='regression',
        estimator_list=[DummyRegressor(strategy='constant', constant=0.0)],
        column_list=[['x']], estimator_names=['dummy'],
    )
    folds = [
        (np.array([0, 1]), np.array([2, 3])),
        (np.array([0, 2]), np.array([1, 3])),
        (np.array([1, 2]), np.array([0, 3])),
    ]
    kwargs = {
        'metric': [mean_absolute_error, get_rmse],
        'metric_name': ['mae', 'rmse'],
        'logic': 'lower',
        'cv_indices': folds[:1] if single_fold else folds,
    }
    if single_fold:
        with pytest.warns(RuntimeWarning, match='never use'):
            results = mv.evaluate_cv(**kwargs)
    else:
        results = mv.evaluate_cv(**kwargs)

    for _, row in results.iterrows():
        for name in ('mae', 'rmse'):
            for prefix in ('train', 'validation', 'reliability'):
                scores = row[f'{name}_{prefix}_folds']
                assert row[f'{name}_{prefix}_min'] == pytest.approx(np.min(scores))
                assert row[f'{name}_{prefix}_max'] == pytest.approx(np.max(scores))
                if single_fold:
                    assert row[f'{name}_{prefix}_min'] == row[f'{name}_{prefix}_max']
                else:
                    assert row[f'{name}_{prefix}_min'] < row[f'{name}_{prefix}_max']


@pytest.mark.parametrize(
    'kwargs',
    [
        {'metric': [], 'metric_name': []},
        {'metric': [accuracy_score, roc_auc_score], 'metric_name': ['accuracy']},
        {'metric': [accuracy_score, roc_auc_score], 'metric_name': ['same', 'same']},
        {'metric': [accuracy_score, None], 'metric_name': ['accuracy', 'bad']},
        {'metric': [accuracy_score, roc_auc_score], 'metric_name': ['accuracy', '']},
        {'metric': [accuracy_score, roc_auc_score], 'metric_name': ['accuracy', 'auc'],
         'logic': ['greater']},
        {'metric': accuracy_score, 'metric_name': 'accuracy', 'logic': 'invalid'},
        {'metric': [accuracy_score, roc_auc_score], 'metric_name': ['accuracy', 'auc'],
         'use_continuous_probabilities': [False]},
        {'metric': accuracy_score, 'metric_name': 'accuracy',
         'use_continuous_probabilities': 'yes'},
    ],
)
def test_majority_vote_invalid_metrics_fail_before_fitting(kwargs):
    mv = MajorityVote(
        train_set=pd.DataFrame({'x': [0, 1, 2, 3]}), y_train=[0, 1, 0, 1],
        task_type='classification', estimator_list=[DummyClassifier()],
        column_list=[['x']],
    )
    with patch.object(DummyClassifier, 'fit') as fit:
        with pytest.raises(ValueError):
            mv.evaluate_cv(**kwargs)
    fit.assert_not_called()


def test_majority_vote_evaluate_cv_reports_progress(majority_vote_classification):
    mv = majority_vote_classification
    progress_descriptions = []

    def track_progress(iterable, **kwargs):
        progress_descriptions.append(kwargs.get('desc'))
        return iterable

    with patch('mlchem.ml.modelling.model_evaluation.tqdm', track_progress):
        mv.evaluate_cv(
            metric=lambda y_true, y_pred: (y_true == y_pred).mean(),
            metric_name='accuracy',
            cv_iter=2,
            n_estimators_max=1,
        )

    assert 'CV folds' in progress_descriptions
    assert any(
        description.startswith('Fitting estimators (fold ')
        for description in progress_descriptions
    )
    assert 'Evaluating combinations' in progress_descriptions


def test_majority_vote_evaluate_cv_can_use_continuous_probabilities(
    majority_vote_classification,
):
    mv = majority_vote_classification
    predictions = []

    def metric_function(y_true, y_pred):
        predictions.append(np.asarray(y_pred))
        return roc_auc_score(y_true, y_pred)

    results = mv.evaluate_cv(
        metric=metric_function,
        metric_name='roc_auc',
        use_continuous_probabilities=True,
        cv_iter=4,
    )

    assert not results.empty
    assert any(np.any((pred > 0) & (pred < 1)) for pred in predictions)
    assert np.isfinite(results['roc_auc_validation_mean']).all()


def test_majority_vote_default_soft_vote_matches_explicit_false(
    majority_vote_classification,
):
    mv = majority_vote_classification

    def collect_predictions(explicit_default):
        predictions = []

        def metric_function(y_true, y_pred):
            predictions.append(np.asarray(y_pred).copy())
            return (y_true == np.round(y_pred)).mean()

        kwargs = {'use_continuous_probabilities': False} if explicit_default else {}
        mv.evaluate_cv(
            metric=metric_function,
            metric_name='accuracy',
            cv_iter=2,
            n_estimators_max=1,
            **kwargs,
        )
        return predictions

    implicit_default = collect_predictions(explicit_default=False)
    explicit_false = collect_predictions(explicit_default=True)

    assert len(implicit_default) == len(explicit_false)
    for implicit, explicit in zip(implicit_default, explicit_false):
        np.testing.assert_array_equal(implicit, explicit)


def test_majority_vote_soft_auc_handles_string_class_labels(
    majority_vote_classification,
):
    mv = majority_vote_classification
    mv.y_train = np.where(mv.y_train == 1, 3, 0)

    results = mv.evaluate_cv(
        metric=roc_auc_score,
        metric_name='roc_auc',
        use_continuous_probabilities=True,
        cv_iter=2,
        n_estimators_max=1,
    )

    assert not results.empty
    assert np.isfinite(results['roc_auc_validation_mean']).all()


def test_majority_vote_evaluate_cv_uses_explicit_cv_indices(
    majority_vote_classification,
):
    mv = majority_vote_classification
    metric_function = lambda y_true, y_pred: (y_true == y_pred).mean()
    manifest = generate_cv_indices(
        mv.train_set.values,
        y=mv.y_train,
        cv_iter=3,
        task_type='classification',
        shuffle=True,
        random_state=7,
    )

    results = mv.evaluate_cv(
        metric=metric_function,
        metric_name='accuracy',
        cv_iter=10,
        cv_indices=manifest,
    )

    assert not results.empty
    assert all(len(fold_scores) == len(manifest) for fold_scores in results['accuracy_train_folds'])
    assert all(len(fold_scores) == len(manifest) for fold_scores in results['accuracy_validation_folds'])


@pytest.fixture
def majority_vote_regression(sample_data):
    train_set, y_train, test_set, y_test = sample_data
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    est_1 = LinearRegression()
    est_2 = LinearRegression()
    estimator_list = [est_1, est_2]
    column_list = [train_set.columns.tolist(), train_set.columns.tolist()]

    return MajorityVote(
        train_set=train_set,
        test_set=test_set,
        y_train=y_train,
        y_test=y_test,
        task_type='regression',
        estimator_list=estimator_list,
        column_list=column_list,
        estimator_names=[],
    )


def test_majority_vote_regression_fit_and_predict(majority_vote_regression):
    mv = majority_vote_regression
    result = mv.fit(return_prediction_dataframes=True)

    assert 'fit_report' in result
    assert 'train_predictions' in result
    assert 'test_predictions' in result
    assert not result['train_predictions'].empty
    assert not result['test_predictions'].empty


def test_majority_vote_lower_reliability_ranks_stable_equal_rmse_first():
    train_set = pd.DataFrame({'x': np.zeros(4)})
    mv = MajorityVote(
        train_set=train_set, y_train=np.array([-10.0, -10.0, 10.0, 10.0]),
        task_type='regression',
        estimator_list=[DummyRegressor(strategy='constant', constant=0.0),
                        DummyRegressor(strategy='constant', constant=8.0)],
        column_list=[['x'], ['x']], estimator_names=['stable', 'unstable'])
    results = mv.evaluate_cv(
        metric=get_rmse, metric_name='RMSE', logic='lower',
        n_estimators_max=1,
        cv_indices=[(np.array([0, 1]), np.array([2, 3])),
                    (np.array([2, 3]), np.array([0, 1]))])

    assert 'unstable' not in results.iloc[0]['combination']
    assert 'stable' in results.iloc[0]['combination']
    assert results['RMSE_train_mean'].tolist() == pytest.approx([10.0, 10.0])
    assert results['RMSE_validation_mean'].tolist() == pytest.approx([10.0, 10.0])
    assert results['RMSE_reliability_mean'].tolist() == pytest.approx([-10.0, -26.0])
    np.testing.assert_allclose(results.iloc[0]['RMSE_reliability_folds'], [-10.0, -10.0])
    np.testing.assert_allclose(results.iloc[1]['RMSE_reliability_folds'], [-26.0, -26.0])


def test_majority_vote_evaluate_cv_regression_outputs_fold_metrics(
    majority_vote_regression,
):
    mv = majority_vote_regression
    metric_function = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))

    results = mv.evaluate_cv(
        metric=metric_function,
        metric_name='mae',
        cv_iter=3,
        logic='lower',
    )

    assert not results.empty
    assert results['mae_reliability_mean'].is_monotonic_decreasing
    assert all(len(fold_scores) == 3 for fold_scores in results['mae_train_folds'])
    assert all(len(fold_scores) == 3 for fold_scores in results['mae_validation_folds'])
    assert all(len(fold_scores) == 3 for fold_scores in results['mae_reliability_folds'])


def test_majority_vote_fit_skips_failed_regression_estimator(sample_data):
    train_set, y_train, test_set, y_test = sample_data
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    mv = MajorityVote(
        train_set=train_set,
        test_set=test_set,
        y_train=y_train,
        y_test=y_test,
        task_type='regression',
        estimator_list=[LinearRegression(), FailingRegressor()],
        column_list=[train_set.columns.tolist(), train_set.columns.tolist()],
        estimator_names=['good_lr', 'bad_fit']
    )

    with pytest.warns(RuntimeWarning, match="Skipping estimator 'bad_fit'"):
        mv.fit()

    assert 'good_lr' in mv.df_train_predictions.columns
    assert 'bad_fit' not in mv.df_train_predictions.columns


def test_majority_vote_fit_raises_when_all_regression_estimators_fail(sample_data):
    train_set, y_train, test_set, y_test = sample_data
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    mv = MajorityVote(
        train_set=train_set,
        test_set=test_set,
        y_train=y_train,
        y_test=y_test,
        task_type='regression',
        estimator_list=[FailingRegressor()],
        column_list=[train_set.columns.tolist()],
        estimator_names=['bad_fit']
    )

    with pytest.warns(RuntimeWarning):
        with pytest.raises(RuntimeError, match="No estimators were successfully fitted"):
            mv.fit()

if __name__ == "__main__":
    pytest.main()