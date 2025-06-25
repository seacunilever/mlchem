import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from mlchem.ml.modelling.model_evaluation import (crossval,
                                                  y_scrambling,
                                                  ApplicabilityDomain,
                                                  MajorityVote)
from mlchem.metrics import get_geometric_S

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
    estimator = LogisticRegression()
    metric_function = lambda y_true, y_pred: (y_true == y_pred).mean()
    
    scores = crossval(estimator, train_set.values, y_train, metric_function, n_fold=5, task_type='regression')
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 5

def test_y_scrambling(sample_data):
    train_set, y_train, test_set, y_test = sample_data
    estimator = LogisticRegression()
    metric_function = get_geometric_S
    
    with pytest.raises(ValueError):
        # Test with invalid number of iterations
        y_scrambling(estimator, train_set.values, y_train, test_set.values, y_test, metric_function, n_iter=-1)

    # Test with valid number of iterations
    with patch('matplotlib.pyplot.show') as mock_show:
            y_scrambling(estimator, train_set.values, y_train, test_set.values, y_test, metric_function, n_iter=10)
            mock_show.assert_called_once()  # Ensure plt.show() is called
            plot = y_scrambling(estimator, train_set.values, y_train, test_set.values, y_test, metric_function, n_iter=100,plot=False)
            plt.savefig('y_scrambling_test_plot.png')


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

def test_majority_vote_fit(majority_vote_classification):
    mv = majority_vote_classification
    mv.fit()
    
    assert not mv.df_train_predictions_hard.empty
    assert not mv.df_test_predictions_hard.empty
    assert not mv.df_train_predictions_soft.empty
    assert not mv.df_test_predictions_soft.empty

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

if __name__ == "__main__":
    pytest.main()