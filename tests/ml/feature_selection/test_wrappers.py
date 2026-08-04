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
import inspect
import logging
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.base import BaseEstimator, ClassifierMixin
from mlchem.ml.feature_selection.wrappers import (SequentialForwardSelection,
                                                  CombinatorialSelection)
from mlchem.metrics import get_geometric_S
import matplotlib.pyplot as plt


class _ParallelAwareEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, n_jobs=4):
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

@pytest.fixture
def fitted_sfs():
    sfs = SequentialForwardSelection(estimator=LogisticRegression(),
                                     estimator_string=None,
                                     metric=get_geometric_S,
                                     max_features=5,
                                     cv_iter=3,
                                     logic='greater')

    # create dataset
    X, y = make_classification(100, 10, n_informative=5,random_state=1)
    train_size = 0.8
    train_samples = int(train_size * len(X))

    X_train, y_train = X[:train_samples], y[:train_samples]
    X_test, y_test = X[train_samples:], y[train_samples:]

    train_set = pd.DataFrame(X_train, columns=np.arange(X_train.shape[1]))
    test_set = pd.DataFrame(X_test, columns=np.arange(X_test.shape[1]))

    # Fit the model
    sfs.fit(train_set, y_train, test_set, y_test)

    return sfs

def test_sequential_forward_selection_fit(fitted_sfs):
    sfs = fitted_sfs
    assert len(sfs.extending_features) > 0
    assert len(sfs.train_scores) > 0
    assert len(sfs.cv_scores) > 0
    assert len(sfs.cv_stds) > 0
    assert len(sfs.unseen_scores) > 0


def test_sequential_forward_selection_task_type_annotation_uses_classification():
    annotation = inspect.signature(
        SequentialForwardSelection.__init__
    ).parameters['task_type'].annotation

    assert 'classification' in str(annotation)
    assert 'classfication' not in str(annotation)

def test_sequential_forward_selection_find_best(fitted_sfs):
    best_features = fitted_sfs.find_best()
    assert 'best_score' in best_features
    assert 'features' in best_features
    assert len(best_features['features']) > 0

def test_sequential_forward_selection_plot(fitted_sfs, tmp_path, monkeypatch):
    plt.close('all')
    plt.switch_backend('Agg')  # Use the Agg backend for testing
    monkeypatch.chdir(tmp_path)
    with patch('matplotlib.pyplot.show') as mock_show:
        fitted_sfs.plot(best_feature=None,save=True)
        mock_show.assert_called_once()  # Ensure plt.show() is called
    assert True  # If no exceptions are raised, the test passes


def test_sequential_forward_selection_parallel_fit():
    sfs = SequentialForwardSelection(
        estimator=LogisticRegression(),
        estimator_string=None,
        metric=get_geometric_S,
        max_features=3,
        cv_iter=3,
        logic='greater',
    )

    X, y = make_classification(100, 8, n_informative=4, random_state=11)
    train_samples = int(0.8 * len(X))
    X_train, y_train = X[:train_samples], y[:train_samples]
    X_test, y_test = X[train_samples:], y[train_samples:]

    train_set = pd.DataFrame(X_train, columns=np.arange(X_train.shape[1]))
    test_set = pd.DataFrame(X_test, columns=np.arange(X_test.shape[1]))

    sfs.fit(train_set, y_train, test_set, y_test, n_jobs=2)

    assert len(sfs.extending_features) == 3
    assert len(sfs.cv_scores) == 3
    assert len(sfs.unseen_scores) == 3


def test_sequential_forward_selection_invalid_n_jobs():
    sfs = SequentialForwardSelection(
        estimator=LogisticRegression(),
        estimator_string=None,
        metric=get_geometric_S,
        max_features=2,
        cv_iter=2,
        logic='greater',
    )

    X, y = make_classification(60, 6, n_informative=3, random_state=7)
    train_samples = int(0.8 * len(X))
    X_train, y_train = X[:train_samples], y[:train_samples]
    X_test, y_test = X[train_samples:], y[train_samples:]
    train_set = pd.DataFrame(X_train, columns=np.arange(X_train.shape[1]))
    test_set = pd.DataFrame(X_test, columns=np.arange(X_test.shape[1]))

    with pytest.raises(ValueError, match="must be -1 or a positive integer"):
        sfs.fit(train_set, y_train, test_set, y_test, n_jobs=0)


def test_sequential_forward_selection_invalid_task_type_raises():
    with pytest.raises(ValueError, match="must be either 'classification' or 'regression'"):
        SequentialForwardSelection(
            estimator=LogisticRegression(),
            estimator_string=None,
            metric=get_geometric_S,
            task_type='invalid',
        )

@pytest.fixture
def fitted_cs_stage_1():
    estimator = LogisticRegression()
    metric = get_geometric_S
    cs = CombinatorialSelection(estimator=estimator, metric=metric, logic='greater')

    # create dataset
    X, y = make_classification(50, 6, n_informative=3,random_state=2)
    train_size = 0.8
    train_samples = int(train_size * len(X))

    X_train, y_train = X[:train_samples], y[:train_samples]
    X_test, y_test = X[train_samples:], y[train_samples:]

    train_set = pd.DataFrame(X_train, columns=np.arange(X_train.shape[1]))
    test_set = pd.DataFrame(X_test, columns=np.arange(X_test.shape[1]))

    # Fit stage 1
    cs.fit_stage_1(train_set=train_set, y_train=y_train,
                   test_set=test_set, y_test=y_test,
                   features=train_set.columns, training_threshold=0.7)

    return cs

@pytest.fixture
def fitted_cs_stage_2(fitted_cs_stage_1):
    # Fit stage 2
    fitted_cs_stage_1.fit_stage_2(top_n_subsets=10, cv_iter=5)
    return fitted_cs_stage_1

def test_combinatorial_selection_fit_stage_1(fitted_cs_stage_1):
    results_stage_1 = fitted_cs_stage_1.df_results_stage1
    assert isinstance(results_stage_1, pd.DataFrame)
    assert 'feature_subsets' in results_stage_1.columns
    assert 'training_score' in results_stage_1.columns
    assert 'cv_score' in results_stage_1.columns
    assert 'test_score' in results_stage_1.columns

def test_combinatorial_selection_fit_stage_2(fitted_cs_stage_2):
    results_stage_2 = fitted_cs_stage_2.df_results_stage2
    assert isinstance(results_stage_2, pd.DataFrame)
    assert 'feature_subsets' in results_stage_2.columns
    assert 'training_score' in results_stage_2.columns
    assert 'cv_score' in results_stage_2.columns
    assert 'test_score' in results_stage_2.columns

def test_combinatorial_selection_display_best_logs_summary(fitted_cs_stage_2, caplog):
    fitted_cs_stage_2.set_log_level('INFO')

    with caplog.at_level(logging.INFO, logger='mlchem.ml.feature_selection.wrappers'):
        fitted_cs_stage_2.display_best(row=1)

    full_text = "\n".join(caplog.messages)
    assert "Best Features" in full_text
    assert "Train Score" in full_text
    assert "CV Score" in full_text
    assert "Test Score" in full_text


def test_wrapper_logging_level_controls_output(fitted_cs_stage_2, caplog):
    fitted_cs_stage_2.set_log_level('WARNING')
    with caplog.at_level(logging.INFO, logger='mlchem.ml.feature_selection.wrappers'):
        fitted_cs_stage_2.display_best(row=1)
    # At WARNING level, INFO logs should not appear
    assert len([m for m in caplog.messages if 'Best Features' in m]) == 0

    caplog.clear()
    fitted_cs_stage_2.set_log_level('INFO')
    with caplog.at_level(logging.INFO, logger='mlchem.ml.feature_selection.wrappers'):
        fitted_cs_stage_2.display_best(row=1)
    assert any('Best Features' in msg for msg in caplog.messages)


def test_combinatorial_selection_stage_1_max_subsets_guard():
    estimator = LogisticRegression()
    metric = get_geometric_S
    cs = CombinatorialSelection(estimator=estimator, metric=metric, logic='greater')

    X, y = make_classification(60, 6, n_informative=3, random_state=9)
    train_size = 0.8
    train_samples = int(train_size * len(X))

    X_train, y_train = X[:train_samples], y[:train_samples]
    X_test, y_test = X[train_samples:], y[train_samples:]

    train_set = pd.DataFrame(X_train, columns=np.arange(X_train.shape[1]))
    test_set = pd.DataFrame(X_test, columns=np.arange(X_test.shape[1]))

    # C(6, 3) = 20, so max_subsets=10 must fail before subset generation.
    with pytest.raises(ValueError, match='exceeds max_subsets=10'):
        cs.fit_stage_1(
            train_set=train_set,
            y_train=y_train,
            test_set=test_set,
            y_test=y_test,
            features=train_set.columns,
            k=3,
            training_threshold=0.5,
            max_subsets=10,
        )


def test_combinatorial_selection_stage_1_copies_features_input():
    estimator = LogisticRegression()
    metric = get_geometric_S
    cs = CombinatorialSelection(estimator=estimator, metric=metric, logic='greater')

    X, y = make_classification(60, 5, n_informative=3, random_state=12)
    train_samples = int(0.8 * len(X))
    train_set = pd.DataFrame(X[:train_samples], columns=np.arange(X.shape[1]))
    test_set = pd.DataFrame(X[train_samples:], columns=np.arange(X.shape[1]))
    y_train = y[:train_samples]
    y_test = y[train_samples:]
    features = train_set.columns.tolist()

    cs.fit_stage_1(
        train_set=train_set,
        y_train=y_train,
        test_set=test_set,
        y_test=y_test,
        features=features,
        k=2,
        training_threshold=1.1,
    )

    features.append('external-mutation')

    assert cs.features == train_set.columns.tolist()


def test_combinatorial_selection_lower_logic_rejects_zero_cv_train_ratio():
    estimator = LogisticRegression()
    metric = get_geometric_S
    cs = CombinatorialSelection(estimator=estimator, metric=metric, logic='lower')

    X, y = make_classification(60, 5, n_informative=3, random_state=18)
    train_samples = int(0.8 * len(X))
    train_set = pd.DataFrame(X[:train_samples], columns=np.arange(X.shape[1]))
    test_set = pd.DataFrame(X[train_samples:], columns=np.arange(X.shape[1]))
    y_train = y[:train_samples]
    y_test = y[train_samples:]

    with pytest.raises(ValueError, match="greater than 0 when logic='lower'"):
        cs.fit_stage_1(
            train_set=train_set,
            y_train=y_train,
            test_set=test_set,
            y_test=y_test,
            features=train_set.columns.tolist(),
            cv_train_ratio=0.0,
        )


def test_combinatorial_selection_stage_1_max_subsets_none_and_parallel():
    estimator = LogisticRegression()
    metric = get_geometric_S
    cs = CombinatorialSelection(estimator=estimator, metric=metric, logic='greater')

    X, y = make_classification(70, 7, n_informative=3, random_state=5)
    train_size = 0.8
    train_samples = int(train_size * len(X))

    X_train, y_train = X[:train_samples], y[:train_samples]
    X_test, y_test = X[train_samples:], y[train_samples:]

    train_set = pd.DataFrame(X_train, columns=np.arange(X_train.shape[1]))
    test_set = pd.DataFrame(X_test, columns=np.arange(X_test.shape[1]))

    results = cs.fit_stage_1(
        train_set=train_set,
        y_train=y_train,
        test_set=test_set,
        y_test=y_test,
        features=train_set.columns,
        k=3,
        training_threshold=0.5,
        max_subsets=None,
        n_jobs=2,
    )

    assert isinstance(results, pd.DataFrame)


def test_combinatorial_selection_invalid_n_jobs_in_stage_1():
    estimator = LogisticRegression()
    metric = get_geometric_S
    cs = CombinatorialSelection(estimator=estimator, metric=metric, logic='greater')

    X, y = make_classification(50, 6, n_informative=3, random_state=3)
    train_samples = int(0.8 * len(X))
    X_train, y_train = X[:train_samples], y[:train_samples]
    X_test, y_test = X[train_samples:], y[train_samples:]
    train_set = pd.DataFrame(X_train, columns=np.arange(X_train.shape[1]))
    test_set = pd.DataFrame(X_test, columns=np.arange(X_test.shape[1]))

    with pytest.raises(ValueError, match="must be -1 or a positive integer"):
        cs.fit_stage_1(
            train_set=train_set,
            y_train=y_train,
            test_set=test_set,
            y_test=y_test,
            features=train_set.columns,
            n_jobs=0,
        )


def test_combinatorial_selection_invalid_task_type_raises():
    with pytest.raises(ValueError, match="must be either 'classification' or 'regression'"):
        CombinatorialSelection(
            estimator=LogisticRegression(),
            metric=get_geometric_S,
            task_type='invalid',
        )


def test_combinatorial_selection_stage_2_max_subsets_guard(fitted_cs_stage_1):
    # Force a deterministic recurrent pool: C(4, 2) = 6 > max_subsets=1.
    fitted_cs_stage_1.df_results_stage1 = pd.DataFrame(
        {
            'feature_subsets': [[0, 1], [1, 2], [2, 3]],
            'training_score': [0.9, 0.85, 0.8],
            'cv_score': [0.9, 0.85, 0.8],
            'test_score': [0.9, 0.85, 0.8],
        }
    )

    with pytest.raises(ValueError, match='exceeds max_subsets=1'):
        fitted_cs_stage_1.fit_stage_2(top_n_subsets=2, cv_iter=3, max_subsets=1)


def test_combinatorial_selection_stage_2_parallel_and_no_limit(fitted_cs_stage_1):
    results = fitted_cs_stage_1.fit_stage_2(
        top_n_subsets=2,
        cv_iter=3,
        max_subsets=None,
        n_jobs=2,
    )

    assert isinstance(results, pd.DataFrame)


def test_rank_features_by_relevance_redundancy_outputs_ranked_features():
    X, y = make_classification(120, 8, n_informative=4, random_state=23)
    train_set = pd.DataFrame(X, columns=np.arange(X.shape[1]))

    cs = CombinatorialSelection(
        estimator=LogisticRegression(),
        metric=get_geometric_S,
        logic='greater'
    )

    ranking = cs.rank_features_by_relevance_redundancy(
        dataframe=train_set,
        target=y,
        features=train_set.columns.tolist(),
        alpha=1.0,
        beta=0.3,
        top_features=5,
        relevance_metric='mutual_info',
        redundancy_metric='pearson',
    )

    assert isinstance(ranking, pd.DataFrame)
    assert list(ranking.columns) == [
        'rank', 'feature', 'relevance', 'redundancy',
        'score', 'alpha', 'beta'
    ]
    assert len(ranking) == 5
    assert ranking['rank'].tolist() == [1, 2, 3, 4, 5]
    assert ranking['score'].is_monotonic_decreasing


def test_combinatorial_stage_1_uses_ranked_features_subset():
    estimator = LogisticRegression()
    metric = get_geometric_S
    cs = CombinatorialSelection(estimator=estimator, metric=metric, logic='greater')

    X, y = make_classification(90, 9, n_informative=4, random_state=29)
    train_size = 0.8
    train_samples = int(train_size * len(X))

    X_train, y_train = X[:train_samples], y[:train_samples]
    X_test, y_test = X[train_samples:], y[train_samples:]

    train_set = pd.DataFrame(X_train, columns=np.arange(X_train.shape[1]))
    test_set = pd.DataFrame(X_test, columns=np.arange(X_test.shape[1]))

    cs.fit_stage_1(
        train_set=train_set,
        y_train=y_train,
        test_set=test_set,
        y_test=y_test,
        features=train_set.columns.tolist(),
        k=2,
        training_threshold=0.0,
        cv_train_ratio=0.0,
        ranking_target=y_train,
        alpha=1.0,
        beta=0.2,
        top_ranked_features=4,
        relevance_metric='mutual_info',
        redundancy_metric='pearson',
    )

    assert hasattr(cs, 'df_feature_ranking')
    assert len(cs.features) == 4
    assert set(cs.features).issubset(set(train_set.columns))


def test_combinatorial_stage_1_ranking_requires_target_when_top_requested():
    estimator = LogisticRegression()
    metric = get_geometric_S
    cs = CombinatorialSelection(estimator=estimator, metric=metric, logic='greater')

    X, y = make_classification(50, 6, n_informative=3, random_state=31)
    train_samples = int(0.8 * len(X))
    train_set = pd.DataFrame(X[:train_samples], columns=np.arange(X.shape[1]))
    test_set = pd.DataFrame(X[train_samples:], columns=np.arange(X.shape[1]))
    y_train = y[:train_samples]
    y_test = y[train_samples:]

    with pytest.raises(ValueError, match="ranking_target"):
        cs.fit_stage_1(
            train_set=train_set,
            y_train=y_train,
            test_set=test_set,
            y_test=y_test,
            features=train_set.columns.tolist(),
            top_ranked_features=3,
        )


def test_sfs_outer_parallel_forces_inner_estimator_n_jobs_to_one():
    seen_n_jobs = []

    def fake_crossval(estimator, X, y, metric, n_fold=5, task_type='classification', random_state=None, shuffle=False):
        seen_n_jobs.append(getattr(estimator, 'n_jobs', None))
        return np.array([0.6, 0.6, 0.6])

    sfs = SequentialForwardSelection(
        estimator=_ParallelAwareEstimator(n_jobs=4),
        estimator_string='parallel-aware',
        metric=lambda y_true, y_pred: (np.array(y_true) == np.array(y_pred)).mean(),
        max_features=2,
        cv_iter=3,
        logic='greater',
    )

    X, y = make_classification(80, 6, n_informative=3, random_state=13)
    train_samples = int(0.8 * len(X))
    train_set = pd.DataFrame(X[:train_samples], columns=np.arange(X.shape[1]))
    test_set = pd.DataFrame(X[train_samples:], columns=np.arange(X.shape[1]))
    y_train = y[:train_samples]
    y_test = y[train_samples:]

    with patch('mlchem.ml.feature_selection.wrappers.crossval', side_effect=fake_crossval):
        sfs.fit(train_set, y_train, test_set, y_test, n_jobs=2)

    assert len(seen_n_jobs) > 0
    assert all(value == 1 for value in seen_n_jobs)


def test_combinatorial_outer_parallel_forces_inner_estimator_n_jobs_to_one():
    seen_n_jobs = []

    def fake_crossval(estimator, X, y, metric, n_fold=5, task_type='classification', random_state=None, shuffle=False):
        seen_n_jobs.append(getattr(estimator, 'n_jobs', None))
        return np.array([0.6, 0.6, 0.6])

    cs = CombinatorialSelection(
        estimator=_ParallelAwareEstimator(n_jobs=8),
        metric=lambda y_true, y_pred: (np.array(y_true) == np.array(y_pred)).mean(),
        logic='greater'
    )

    X, y = make_classification(80, 6, n_informative=3, random_state=17)
    train_samples = int(0.8 * len(X))
    train_set = pd.DataFrame(X[:train_samples], columns=np.arange(X.shape[1]))
    test_set = pd.DataFrame(X[train_samples:], columns=np.arange(X.shape[1]))
    y_train = y[:train_samples]
    y_test = y[train_samples:]

    with patch('mlchem.ml.feature_selection.wrappers.crossval', side_effect=fake_crossval):
        cs.fit_stage_1(
            train_set=train_set,
            y_train=y_train,
            test_set=test_set,
            y_test=y_test,
            features=train_set.columns,
            k=2,
            training_threshold=0.0,
            cv_train_ratio=0.0,
            n_jobs=2,
        )

    assert len(seen_n_jobs) > 0
    assert all(value == 1 for value in seen_n_jobs)

if __name__ == '__main__':
    pytest.main()