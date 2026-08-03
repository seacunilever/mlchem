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
from unittest.mock import patch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification
from mlchem.ml.modelling.model_evaluation import (crossval,
                                                  y_scrambling,
                                                  ApplicabilityDomain,
                                                  MajorityVote)
from mlchem.metrics import get_geometric_S


class FailingEstimator:
    def fit(self, X, y):
        raise RuntimeError("Intentional fit failure")


class FailingRegressor:
    def fit(self, X, y):
        raise RuntimeError("Intentional fit failure")

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
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 5

def test_crossval_regression(sample_data):
    train_set, y_train, _, _ = sample_data
    estimator = LinearRegression()
    y_train_reg = y_train.astype(float)
    metric_function = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))

    scores = crossval(estimator, train_set.values, y_train_reg, metric_function, n_fold=5, task_type='regression')
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 5


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

    assert isinstance(scores, np.ndarray)
    assert len(scores) == 5


@pytest.mark.parametrize('task_type, estimator, y_values', [
    ('classification', LogisticRegression(), 'classification'),
    ('regression', LinearRegression(), 'regression'),
])
def test_crossval_shuffle_true_propagates_random_state(sample_data, task_type, estimator, y_values):
    train_set, y_train, _, _ = sample_data
    metric_function = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    y_input = y_train if y_values == 'classification' else y_train.astype(float)

    with patch('sklearn.model_selection.cross_val_score', return_value=np.array([1.0])) as mock_cross_val_score:
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

    cv_splitter = mock_cross_val_score.call_args.kwargs['cv']
    assert cv_splitter.shuffle is True
    assert cv_splitter.random_state == 77

def test_y_scrambling(sample_data, tmp_path, monkeypatch):
    train_set, y_train, test_set, y_test = sample_data
    estimator = LogisticRegression()
    metric_function = get_geometric_S
    monkeypatch.chdir(tmp_path)
    
    with pytest.raises(ValueError, match='empty'):
        # Test with invalid number of iterations
        y_scrambling(estimator, train_set.values, y_train, test_set.values, y_test, metric_function, n_iter=-1)

    # Test with valid number of iterations
    with patch('matplotlib.pyplot.show') as mock_show:
            y_scrambling(estimator, train_set.values, y_train, test_set.values, y_test, metric_function, n_iter=10)
            mock_show.assert_called_once()  # Ensure plt.show() is called
            plot = y_scrambling(estimator, train_set.values, y_train, test_set.values, y_test, metric_function, n_iter=100,plot=False)
            plt.savefig('y_scrambling_test_plot.png')

def test_y_scrambling_with_dataframe_inputs(sample_data):
    train_set, y_train, test_set, y_test = sample_data
    estimator = LogisticRegression(max_iter=250)
    metric_function = get_geometric_S

    # Exercise DataFrame input branch in y_scrambling conversion logic.
    y_scrambling(
        estimator,
        train_set,
        y_train,
        test_set,
        y_test,
        metric_function,
        n_iter=2,
        plot=False,
    )


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

def test_majority_vote_predict(majority_vote_classification):
    mv = majority_vote_classification
    mv.fit()
    
    metric_function = lambda y_true, y_pred: (y_true == y_pred).mean()
    
    with patch('mlchem.helper.generate_combination_cascade') as mock_generate_combination_cascade:
        mock_generate_combination_cascade.return_value = [['LR'], ['RF']]
        
        mv.predict(metric=metric_function, metric_name='accuracy', n_estimators_max=2)
        
        assert not mv.final_results.empty
        assert 'accuracy_train' in mv.final_results.columns
        assert 'accuracy_test' in mv.final_results.columns


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
    mv.fit()

    assert not mv.df_train_predictions.empty
    assert not mv.df_test_predictions.empty

    metric_function = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    mv.predict(metric=metric_function, metric_name='mae', n_estimators_max=2)

    assert not mv.final_results.empty
    assert 'mae_train' in mv.final_results.columns
    assert 'mae_test' in mv.final_results.columns


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