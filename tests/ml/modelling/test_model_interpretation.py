import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from mlchem.ml.modelling.model_interpretation import ShapExplainer
from mlchem.ml.modelling.model_interpretation import DescriptorExplainer
from mlchem.metrics import get_rmse
from sklearn.datasets import make_classification

@pytest.fixture
def sample_data():
    n_features = 10
    X, y = make_classification(100, n_features, n_informative=4,random_state=1)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    return X, y

@pytest.fixture
def shap_explainer_tree(sample_data):
    X, y = sample_data
    estimator = RandomForestClassifier()
    estimator.fit(X, y)
    explainer = ShapExplainer(estimator=estimator, data=X, y=y, is_tree=True)
    explainer.explain()
    return explainer

@pytest.fixture
def shap_explainer_notree(sample_data):
    X, y = sample_data
    estimator = LogisticRegression()
    estimator.fit(X, y)
    explainer = ShapExplainer(estimator=estimator, data=X, y=y, is_tree=False)
    explainer.explain()
    return explainer


def test_shap_explainer_explain_tree(shap_explainer_tree):
    assert hasattr(shap_explainer_tree, 'explainer')
    assert hasattr(shap_explainer_tree, 'base_values')
    assert hasattr(shap_explainer_tree, 'shap_values')

def test_shap_explainer_explain_notree(shap_explainer_notree):
    assert hasattr(shap_explainer_notree, 'explainer')
    assert hasattr(shap_explainer_notree, 'base_values')
    assert hasattr(shap_explainer_notree, 'shap_values')

def test_shap_explainer_tree_load(shap_explainer_tree):
    base_values = np.random.rand(100)
    shap_values = np.random.rand(100, 10)
    shap_explainer_tree.load(base_values, shap_values)
    assert np.array_equal(shap_explainer_tree.base_values, base_values)
    assert np.array_equal(shap_explainer_tree.shap_values, shap_values)

def test_shap_explainer_notree_load(shap_explainer_notree):
    base_values = np.random.rand(100)
    shap_values = np.random.rand(100, 10)
    shap_explainer_notree.load(base_values, shap_values)
    assert np.array_equal(shap_explainer_notree.base_values, base_values)
    assert np.array_equal(shap_explainer_notree.shap_values, shap_values)

@patch('joblib.dump')
def test_shap_tree_explainer_save(mock_dump, shap_explainer_tree):
    shap_explainer_tree.save(path='test_path', filename='test_filename')
    assert mock_dump.call_count == 2

@patch('joblib.dump')
def test_shap_notree_explainer_save(mock_dump, shap_explainer_notree):
    shap_explainer_notree.save(path='test_path', filename='test_filename')
    assert mock_dump.call_count == 2

@patch('shap.initjs')
@patch('shap.force_plot')
def test_shap_explainer_treeforce_plot(mock_force_plot, mock_initjs, shap_explainer_tree):
    shap_explainer_tree.force_plot()
    mock_initjs.assert_called_once()
    mock_force_plot.assert_called()

@patch('shap.initjs')
@patch('shap.force_plot')
def test_shap_explainer_notree_force_plot(mock_force_plot, mock_initjs, shap_explainer_notree):
    shap_explainer_notree.force_plot()
    mock_initjs.assert_called_once()
    mock_force_plot.assert_called()

@patch('shap.initjs')
@patch('shap.force_plot')
def test_shap_explainer_tree_force_plot_single(mock_force_plot, mock_initjs, shap_explainer_tree):
    shap_explainer_tree.force_plot_single(i=0)
    mock_initjs.assert_called_once()
    mock_force_plot.assert_called()

@patch('shap.initjs')
@patch('shap.force_plot')
def test_shap_explainer_notree_force_plot_single(mock_force_plot, mock_initjs, shap_explainer_notree):
    shap_explainer_notree.force_plot_single(i=0)
    mock_initjs.assert_called_once()
    mock_force_plot.assert_called()

@patch('shap.initjs')
@patch('shap.dependence_plot')
def test_shap_explainer_tree_dependence_plot(mock_dependence_plot, mock_initjs, shap_explainer_tree):
    shap_explainer_tree.dependence_plot(column=0)
    mock_initjs.assert_called_once()
    mock_dependence_plot.assert_called()

@patch('shap.initjs')
@patch('shap.dependence_plot')
def test_shap_explainer_notree_dependence_plot(mock_dependence_plot, mock_initjs, shap_explainer_notree):
    shap_explainer_notree.dependence_plot(column=0)
    mock_initjs.assert_called_once()
    mock_dependence_plot.assert_called()

@patch('shap.initjs')
@patch('shap.summary_plot')
def test_shap_explainer_tree_summary_plot(mock_summary_plot, mock_initjs, shap_explainer_tree):
    shap_explainer_tree.summary_plot(plot_type='dot')
    mock_initjs.assert_called_once()
    mock_summary_plot.assert_called()

@patch('shap.initjs')
@patch('shap.summary_plot')
def test_shap_explainer_notree_summary_plot(mock_summary_plot, mock_initjs, shap_explainer_notree):
    shap_explainer_notree.summary_plot(plot_type='dot')
    mock_initjs.assert_called_once()
    mock_summary_plot.assert_called()

@patch('shap.initjs')
@patch('shap.waterfall_plot')
def test_shap_explainer_tree_waterfall_plot(mock_waterfall_plot, mock_initjs, shap_explainer_tree):
    with pytest.raises(ValueError, match='Waterfall plot not available for Tree models.'):
        shap_explainer_tree.waterfall_plot(i=0)

@patch('shap.initjs')
@patch('shap.waterfall_plot')
def test_shap_explainer_notree_waterfall_plot(mock_waterfall_plot, mock_initjs, shap_explainer_notree):
    shap_explainer_notree.waterfall_plot(i=0)
    mock_initjs.assert_called_once()
    mock_waterfall_plot.assert_called()

@patch('shap.initjs')
@patch('shap.summary_plot')
def test_shap_explainer_tree_bar_plot(mock_summary_plot, mock_initjs, shap_explainer_tree):
    shap_explainer_tree.bar_plot()
    mock_initjs.assert_called_once()
    mock_summary_plot.assert_called()

@patch('shap.initjs')
@patch('shap.plots.bar')
def test_shap_explainer_notree_bar_plot(mock_bar_plot, mock_initjs, shap_explainer_notree):
    shap_explainer_notree.bar_plot()
    mock_initjs.assert_called_once()
    mock_bar_plot.assert_called()

@patch('shap.initjs')
@patch('shap.plots.heatmap')
def test_shap_explainer_tree_heatmap(mock_heatmap, mock_initjs, shap_explainer_tree):
    with pytest.raises(ValueError, match='Heatmap plot not available for Tree models.'):
        shap_explainer_tree.heatmap()

@patch('shap.initjs')
@patch('shap.plots.heatmap')
def test_shap_explainer_notree_heatmap(mock_heatmap, mock_initjs, shap_explainer_notree):
    shap_explainer_notree.heatmap()
    mock_initjs.assert_called_once()
    mock_heatmap.assert_called()


@patch('shap.initjs')
@patch('matplotlib.pyplot.show')
def test_shap_explainer_tree_feature_importance(mock_show, mock_initjs, shap_explainer_tree):
    if str(shap_explainer_tree.estimator).startswith('XGB'):
        feature_importance_df = shap_explainer_tree.feature_importance()
        mock_initjs.assert_called_once()
        mock_show.assert_called()
        assert isinstance(feature_importance_df, pd.DataFrame)
        assert 'W' in feature_importance_df.columns
        assert 'correlation' in feature_importance_df.columns
        assert 'overall_impact' in feature_importance_df.columns
        assert len(feature_importance_df) == shap_explainer_tree.data.shape[1]
    else:
        with pytest.raises(
            ValueError):
            shap_explainer_tree.feature_importance()
            mock_initjs.assert_called_once()
            mock_show.assert_called()

@patch('shap.initjs')
@patch('matplotlib.pyplot.show')
def test_shap_explainer_notree_feature_importance(mock_show, mock_initjs, shap_explainer_notree):
    feature_importance_df = shap_explainer_notree.feature_importance()
    mock_initjs.assert_called_once()
    mock_show.assert_called()
    assert isinstance(feature_importance_df, pd.DataFrame)
    assert 'W' in feature_importance_df.columns
    assert 'correlation' in feature_importance_df.columns
    assert 'overall_impact' in feature_importance_df.columns
    assert len(feature_importance_df) == shap_explainer_notree.data.shape[1]


@patch('shap.initjs')
@patch('shap.decision_plot')
def test_shap_explainer_tree_decision_plot(mock_decision_plot, mock_initjs, shap_explainer_tree):
    shap_explainer_tree.decision_plot(interval_lower=0.2, interval_upper=0.8)
    mock_initjs.assert_called_once()
    mock_decision_plot.assert_called()

@patch('shap.initjs')
@patch('shap.decision_plot')
def test_shap_explainer_notree_decision_plot(mock_decision_plot, mock_initjs, shap_explainer_notree):
    with pytest.raises(ValueError,match='Not supported yet for this non-tree model.'):
        shap_explainer_notree.decision_plot(interval_lower=0.2, interval_upper=0.8)


@pytest.fixture
def sample_data_descriptor_explainer():
    np.random.seed(1)
    X_train = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y_train = pd.DataFrame(np.random.rand(100, 1), columns=['target'])
    X_test = pd.DataFrame(np.random.rand(50, 10), columns=[f'feature_{i}' for i in range(10)])
    y_test = pd.DataFrame(np.random.rand(50, 1), columns=['target'])
    return X_train, y_train, X_test, y_test

@pytest.fixture
def descriptor_explainer(sample_data_descriptor_explainer):
    X_train, y_train, X_test, y_test = sample_data_descriptor_explainer
    estimator = LinearRegression()
    metric = get_rmse
    return DescriptorExplainer(df_train=X_train, df_test=X_test, target_train=y_train,
                               target_test=y_test, estimator=estimator, metric=metric,
                               logic='lower')

@pytest.fixture
def stage_1_fitted_explainer(descriptor_explainer):
    descriptor_explainer.fit_stage_1()
    return descriptor_explainer


def test_descriptor_explainer_fit_stage_1(stage_1_fitted_explainer):
    assert hasattr(stage_1_fitted_explainer, 'df_results_stage_1')
    assert not stage_1_fitted_explainer.df_results_stage_1.empty, "Stage 1 results are empty"

def test_descriptor_explainer_fit_stage_2(stage_1_fitted_explainer):
    assert not stage_1_fitted_explainer.df_results_stage_1.empty, "Stage 1 results are empty"
    stage_1_fitted_explainer.fit_stage_2()
    assert hasattr(stage_1_fitted_explainer, 'df_results_stage_2')
    assert not stage_1_fitted_explainer.df_results_stage_2.empty, "Stage 2 results are empty"

@patch('matplotlib.pyplot.show')
def test_descriptor_explainer_display(mock_show, stage_1_fitted_explainer):
    assert not stage_1_fitted_explainer.df_results_stage_1.empty, "Stage 1 results are empty"
    stage_1_fitted_explainer.fit_stage_2()
    assert not stage_1_fitted_explainer.df_results_stage_2.empty, "Stage 2 results are empty"
    # Ensure the best_features attribute is set correctly
    subset_index = 0
    stage_1_fitted_explainer.best_features = list(stage_1_fitted_explainer.df_results_stage_2.feature_subsets)[subset_index]
    # Mock the estimator's predict method to return a numpy array
    mock_predict = MagicMock(return_value=np.random.rand(len(stage_1_fitted_explainer.target_train)))
    stage_1_fitted_explainer.estimator.predict = mock_predict
    
    # Call the display method and check for the plot call
    stage_1_fitted_explainer.display(subset_index=subset_index)
    
    mock_show.assert_called_once()