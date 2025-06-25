import pytest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from mlchem.ml.feature_selection.wrappers import (SequentialForwardSelection,
                                                  CombinatorialSelection)
from mlchem.metrics import get_geometric_S
import matplotlib.pyplot as plt

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

def test_sequential_forward_selection_find_best(fitted_sfs):
    best_features = fitted_sfs.find_best()
    assert 'best_score' in best_features
    assert 'features' in best_features
    assert len(best_features['features']) > 0

def test_sequential_forward_selection_plot(fitted_sfs):
    plt.close('all')
    plt.switch_backend('Agg')  # Use the Agg backend for testing
    with patch('matplotlib.pyplot.show') as mock_show:
        fitted_sfs.plot(best_feature=None,save=True)
        mock_show.assert_called_once()  # Ensure plt.show() is called
    assert True  # If no exceptions are raised, the test passes

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

def test_combinatorial_selection_display_best(fitted_cs_stage_2, capsys):
    # Display best
    fitted_cs_stage_2.display_best(row=1)
    captured = capsys.readouterr()
    assert "Best Features" in captured.out
    assert "Train Score" in captured.out
    assert "CV Score" in captured.out
    assert "Test Score" in captured.out

if __name__ == '__main__':
    pytest.main()