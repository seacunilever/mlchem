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

import matplotlib.pyplot as plt
import numpy as np
from typing import Literal, Iterable
import pandas as pd
from IPython.display import HTML


class ShapExplainer:
    """
A class to generate SHAP (SHapley Additive exPlanations) values for 
model interpretability.

This class provides methods to explain the output of machine learning 
models using SHAP values. It supports both tree-based models and general 
estimators, and offers a variety of visualisation tools to understand 
the impact of each feature on model predictions.

Attributes
----------
model_type : str
    The type of model ('tree' or other).

data : pandas.DataFrame
    The dataset used for generating SHAP values.

y : array-like
    The target values corresponding to the dataset.

estimator : object
    The machine learning model to be explained.

Methods
-------
explain()
    Generates SHAP values for the model.

load(base_values, shap_values)
    Loads precomputed SHAP values.

save(path, filename)
    Saves the SHAP values to the specified path.

force_plot()
    Generates a force plot to visualise SHAP values.

force_plot_single(i)
    Generates a force plot for a single instance.

dependence_plot(column)
    Generates a dependence plot for a specified feature.

dependence_plot_all()
    Generates dependence plots for all features.

summary_plot(plot_type='dot')
    Generates a summary plot of SHAP values.

waterfall_plot(i)
    Generates a waterfall plot for a single instance.

bar_plot()
    Generates a bar plot of SHAP values.

feature_importance()
    Calculates and returns feature importance based on SHAP values.

decision_plot(interval_lower, interval_upper)
    Generates a decision plot for a specified probability interval.
"""


    def __init__(self, estimator, data: pd.DataFrame,
                 y: Iterable, is_tree: bool = False) -> None:
        """
Initialise the ShapExplainer with a model, dataset, and target values.

Parameters
----------
estimator : object
    The machine learning model to be explained.

data : pandas.DataFrame
    The dataset used for generating SHAP values.

y : Iterable
    The target values corresponding to the dataset.

is_tree : bool, optional (default=False)
    Whether the model is a tree-based model (e.g., XGBoost, LightGBM).
"""


        self.estimator = estimator
        self.data = data
        self.y = y
        self.is_tree = is_tree

    def explain(self) -> None:
        """
        Generate SHAP values for the model.

        This method creates a SHAP explainer based on the model type and
        computes the SHAP values for the provided dataset.

        Returns
        -------
        None
        """

        import shap

        if self.is_tree:
            self.explainer = shap.TreeExplainer(self.estimator)
            self.base_values = self.explainer.expected_value
            self.shap_values = self.explainer.shap_values(self.data)
        else:
            self.predictive_function = lambda x: self.estimator.\
                predict_proba(x)[:, 1]
            self.explainer = shap.Explainer(self.predictive_function,
                                            self.data)
            self.shap_explanation_object = self.explainer(self.data)
            self.base_values = self.shap_explanation_object.base_values
            self.shap_values = self.shap_explanation_object.values

    def load(self,
             base_values: np.ndarray,
             shap_values: np.ndarray) -> None:
        """
Load precomputed SHAP values.

Parameters
----------
base_values : numpy.ndarray
    The precomputed base values.

shap_values : numpy.ndarray
    The precomputed SHAP values.

Returns
-------
None
"""

        self.base_values = base_values
        self.shap_values = shap_values

    def save(self, path: str, filename: str) -> None:
        """
Save SHAP values to the specified path.

Parameters
----------
path : str
    The directory path where the SHAP values will be saved.

filename : str
    The base filename for the saved SHAP values.

Returns
-------
None
"""

        import os
        import joblib

        try:
            os.mkdir(path)
        except FileExistsError:
            pass

        joblib.dump(self.base_values, f'{path}/base_values_{filename}')
        joblib.dump(self.shap_values, f'{path}/shap_values_{filename}')

    def force_plot(self) -> HTML:
        """
Generate a force plot to visualise SHAP values.

Returns
-------
IPython.core.display.HTML
    An HTML object containing the SHAP force plot.
"""

        import shap
        shap.initjs()

        if self.is_tree:
            if str(self.estimator).startswith('XGB'):     # if XGBoost:
                plot = shap.\
                    force_plot(base_value=self.base_values,
                               shap_values=self.shap_values,
                               features=self.data, link='logit',
                               )
            else:
                try:
                    plot = shap.\
                        force_plot(base_value=self.base_values[1],
                                   shap_values=self.shap_values[1],
                                   features=self.data)
                except IndexError:
                    plot = shap.\
                        force_plot(base_value=self.base_values,
                                   shap_values=self.shap_values,
                                   features=self.data)
        else:
            plot = shap.\
                force_plot(base_value=self.base_values,
                           shap_values=self.shap_values,
                           features=self.data)

        return HTML(f"<div style='background-colour:white;'>{shap.getjs() + plot.html()}</div>")

    def force_plot_single(self, i: int) -> HTML:
        """
Generate a force plot for a single instance.

Parameters
----------
i : int
    The index of the instance to visualise.

Returns
-------
IPython.core.display.HTML
    An HTML object containing the SHAP force plot for the specified instance.
"""

        import shap
        shap.initjs()

        print(self.data.iloc[i, :])
        if self.is_tree:
            if str(self.estimator).startswith('XGB'):
                self.plot = shap.force_plot(
                    base_value=self.base_values,
                    shap_values=self.shap_values[i],
                    features=self.data.iloc[i, :],
                    link='logit'
                )
            else:
                try:
                    self.plot = shap.force_plot(
                        base_value=self.base_values[1],
                        shap_values=self.shap_values[1][i, :],
                        features=self.data.iloc[i, :]
                    )
                except IndexError:
                    self.plot = shap.force_plot(
                        base_value=self.base_values,
                        shap_values=self.shap_values[i],
                        features=self.data.iloc[i, :]
                    )
        else:
            self.plot = shap.force_plot(
                base_value=self.base_values[i],
                shap_values=self.shap_values[i],
                features=self.data.iloc[i]
            )

        return HTML(
            f"<div style='background-colour:White;'>"
            f"{shap.getjs() + self.plot.html()}"
            f"</div>"
        )

    def dependence_plot(self, column: int) -> None:
        """
Generate a dependence plot for a specified feature.

Parameters
----------
column : int
    The index of the feature to visualise.

Returns
-------
None
"""

        import shap
        shap.initjs()

        if self.is_tree:
            try:
                self.plot = shap.dependence_plot(
                    column,
                    self.shap_values[1],
                    self.data
                )
            except IndexError:
                self.plot = shap.dependence_plot(
                    column,
                    self.shap_values,
                    self.data
                )
        else:
            self.plot = shap.dependence_plot(
                column,
                self.shap_values,
                self.data
            )
        return self.plot

    def dependence_plot_all(self) -> None:
        """
Generate dependence plots for all features.

Returns
-------
None
"""

        import shap
        shap.initjs()

        if self.is_tree:
            try:
                for column in range(self.data.shape[1]):
                    shap.dependence_plot(
                        column,
                        self.shap_values[1],
                        self.data
                    )
            except IndexError:
                for column in range(self.data.shape[1]):
                    shap.dependence_plot(
                        column,
                        self.shap_values,
                        self.data
                    )
        else:
            for column in range(self.data.shape[1]):
                shap.dependence_plot(
                    column,
                    self.shap_values,
                    self.data
                )

    def summary_plot(self, plot_type: Literal['dot',
                                              'bar',
                                              'violin',
                                              'layered_violin',
                                              'compact_dot'] = 'dot') -> None:
        """
Generate a summary plot of SHAP values.

Parameters
----------
plot_type : {'dot', 'bar', 'violin', 'layered_violin', 'compact_dot'}, 
optional (default='dot')
    The type of plot to generate.

Returns
-------
None
"""

        import shap
        shap.initjs()

        self.plot_type = plot_type
        if self.is_tree:
            try:
                self.plot = shap.summary_plot(
                    self.shap_values[1],
                    self.data,
                    plot_type=self.plot_type
                )
            except Exception:
                self.plot = shap.summary_plot(
                    self.shap_values,
                    self.data,
                    plot_type=self.plot_type
                )
        else:
            self.plot = shap.summary_plot(
                self.shap_values,
                self.data,
                plot_type=self.plot_type
            )
        return self.plot

    def waterfall_plot(self, i: int) -> None:
        """
Generate a waterfall plot for a single instance.

Note: Not available for tree-based models.

Parameters
----------
i : int
    The index of the instance to visualise.

Returns
-------
None
"""

        import shap
        shap.initjs()

        if self.is_tree:
            raise ValueError('Waterfall plot not available for Tree models.')
        else:
            try:
                self.plot = shap.waterfall_plot(
                    self.shap_values,
                    instance_order=self.shap_values.sum(1)
                )
            except IndexError:
                self.plot = shap.waterfall_plot(
                    shap_values=self.shap_explanation_object[i]
                )
            return self.plot

    def bar_plot(self) -> None:
        """
Generate a bar plot of SHAP values.

Returns
-------
None
"""

        import shap
        shap.initjs()

        if self.is_tree:
            try:
                self.plot = shap.summary_plot(
                    self.shap_values[1],
                    self.data,
                    plot_type='bar'
                )
            except IndexError:
                self.plot = shap.summary_plot(
                    self.shap_values,
                    self.data,
                    plot_type='bar'
                )
        else:
            try:
                self.plot = shap.plots.bar(self.shap_values)
            except IndexError:
                self.plot = shap.plots.bar(self.shap_explanation_object)
        return self.plot

    def heatmap(self) -> None:
        """
Generate a heatmap of SHAP values.

This method creates a heatmap to visualise the SHAP values for the model's
predictions. It initialises the SHAP JavaScript visualisation and displays
the plot.

Note
----
Heatmaps are not available for tree-based models.

Returns
-------
None
"""

        import shap
        shap.initjs()

        if self.is_tree:
            raise ValueError('Heatmap plot not available for Tree models.')
        else:
            try:
                self.plot = shap.plots.heatmap(
                    self.shap_values,
                    instance_order=self.shap_values.sum(1)
                )
            except IndexError:
                self.plot = shap.plots.heatmap(
                    self.shap_explanation_object,
                    instance_order=self.shap_explanation_object.sum(1)
                )
        return self.plot

    def feature_importance(self) -> pd.DataFrame:
        """
Calculate and visualise feature importance based on SHAP values.

This method computes feature importance by summing the absolute SHAP values
for each feature. It also calculates the Spearman correlation between each
feature and its SHAP values, and derives an overall impact score. The results
are visualised using a color-coded bar plot and returned as a DataFrame.

Returns
-------
pandas.DataFrame
    A DataFrame containing the feature importance weights, correlations,
    and overall impact scores for each feature.
"""


        import shap
        shap.initjs()
        import pandas as pd
        from scipy.stats import spearmanr
        import seaborn as sns
        try:
            importances = [
                abs(self.shap_values)[:,i].sum()/\
                    (abs(self.shap_values).sum().sum())
                    for i in range(len(self.data.columns))
                    ] # abs importance over the total  for each feature
            correlations = [
                spearmanr(self.shap_values[:,i],
                            self.data[self.data.columns[i]].values
                            )[0] for i in range(len(self.data.columns))]
            
            correlations = list(np.nan_to_num(correlations))
            overall_impacts = [w*(c**2) if c>=0 else -w*(c**2) for w,c in zip(importances,correlations)]
        except Exception as e:
            raise ValueError(
                f'Not supported yet for the model {self.estimator}: {e}'
                )
        self.feat_importances = pd.DataFrame(index=self.data.columns)
        self.feat_importances['W'] = importances
        self.feat_importances['correlation'] = correlations
        self.feat_importances['overall_impact'] = overall_impacts

        self.feat_importances.sort_values('W',ascending=False,inplace=True)

        modified_weights = []
        for feature, weight, impact in zip(
            self.feat_importances.index,
            self.feat_importances.W,
            self.feat_importances.overall_impact
        ):
            if impact >=0:
                modified_weights.append(weight)
            else:
                modified_weights.append(-1 * weight)
        modified_weights = np.array(modified_weights)

        modified_weights_reds = []
        modified_weights_light_reds = []
        modified_weights_blues = []
        modified_weights_light_blues = []

        for mw,imp in zip(modified_weights,self.feat_importances.overall_impact):
            if imp >= 0.1:
                modified_weights_reds.append(mw)
                modified_weights_light_reds.append(0)
                modified_weights_blues.append(0)
                modified_weights_light_blues.append(0)
            elif 0 <= imp < 0.1:
                modified_weights_reds.append(0)
                modified_weights_light_reds.append(mw)
                modified_weights_blues.append(0)
                modified_weights_light_blues.append(0)
            elif imp <= -0.1:
                modified_weights_reds.append(0)
                modified_weights_light_reds.append(0)
                modified_weights_blues.append(mw)
                modified_weights_light_blues.append(0)
            elif -0.1 < imp < 0:
                modified_weights_reds.append(0)
                modified_weights_light_reds.append(0)
                modified_weights_blues.append(0)
                modified_weights_light_blues.append(mw)

        plt.figure(figsize=(9, 5))
        self.ax_red = sns.barplot(
            x=modified_weights_reds,
            y=self.feat_importances.index,
            color='red',
            alpha=0.8,
            width=0.5
        )
        self.ax_light_red = sns.barplot(
            x=modified_weights_light_reds,
            y=self.feat_importances.index,
            color='red',
            alpha=0.5,
            width=0.5
        )
        self.ax_blue = sns.barplot(
            x=modified_weights_blues,
            y=self.feat_importances.index,
            color='blue',
            alpha=0.8,
            width=0.5
        )
        self.ax_light_blue = sns.barplot(
            x=modified_weights_light_blues,
            y=self.feat_importances.index,
            color='blue',
            alpha=0.5,
            width=0.5
        )
        plt.grid(axis='x', alpha=0.5)
        plt.ylabel('Descriptors',size=12)
        plt.xlabel('Importance',size=12)
        plt.yticks(size=14)
        plt.xticks(size=14)
        plt.show()

        
        return self.feat_importances

    def decision_plot(self, interval_lower: float, interval_upper: float) -> None:
        """
Generate a decision plot for a specified probability interval.

This method creates a SHAP decision plot to visualise how features contribute
to model predictions for samples within a given probability interval. It also
prints classification statistics for the selected interval.

Note
----
Currently only supported for tree-based models.

Parameters
----------
interval_lower : float
    The lower bound of the probability interval.

interval_upper : float
    The upper bound of the probability interval.

Returns
-------
None
"""

        print("Scikit-learn version 1.6 modified the API around its 'tags', "
              "and that's the cause of some known errors.\nXGBoost has made the"
              "necessary changes in PR11021, but at present that hasn't "
              "made it into a released version.\nYou can either keep your"
              "sklearn version <1.6, or build XGBoost directly from github"
              "(or upgrade XGBoost, after a new version is released)."
              "\nIn sklearn 1.6.1, the error was downgraded to a warning "
              "(to be returned to an error in 1.7). So you may also "
              "install sklearn >=1.6.1,<1.7 and just expect DeprecationWarnings."
              )

        import shap
        shap.initjs()
        from mlchem.helper import create_mask

        self.misclassified = self.estimator.predict(self.data) != self.y
        self.y_pred_proba = self.estimator.predict_proba(self.data)[:, 1]
        self.mask = create_mask(
            np.array(self.y_pred_proba),
            interval_lower,
            interval_upper
        )
        self.samples_in_interval = np.array(self.y)[self.mask].shape[0]
        self.misclassified_samples = self.misclassified[self.mask].sum()
        self.accuracy = 1 - (self.misclassified_samples /
                             self.samples_in_interval)

        print(
            f'Samples in the interval {interval_lower:.2f} - {interval_upper:.2f}: '
            f'{self.samples_in_interval}\nMisclassified samples: '
            f'{self.misclassified_samples}\nCorrectly classified samples: '
            f'{self.accuracy:.2f}'
        )

        if self.is_tree:
            if str(self.estimator).startswith('XGB'):
                return shap.decision_plot(
                    self.base_values,
                    self.shap_values[self.mask],
                    features=self.data,
                    feature_order='hclust',
                    highlight=self.misclassified[self.mask],
                    ignore_warnings=True,
                    link='logit'
                )
            else:
                try:
                    return shap.decision_plot(
                        self.base_values[1],
                        self.shap_values[1][self.mask],
                        features=self.data,
                        feature_order='hclust',
                        highlight=self.misclassified[self.mask],
                        ignore_warnings=True
                    )
                except IndexError:
                    return shap.decision_plot(
                        self.base_values,
                        self.shap_values[self.mask],
                        features=self.data,
                        feature_order='hclust',
                        highlight=self.misclassified[self.mask],
                        ignore_warnings=True
                    )
        else:
            raise ValueError('Not supported yet for this non-tree model.')


class DescriptorExplainer:
    """
A class to perform feature selection and model evaluation using 
combinatorial selection methods.

This class provides methods to fit and evaluate machine learning models 
using combinatorial feature selection. It supports both classification 
and regression tasks, offering tools to identify the best feature 
subsets and visualise model performance.

Attributes
----------
df_train : pandas.DataFrame
    The training dataset.

df_test : pandas.DataFrame
    The testing dataset.

target_train : pandas.DataFrame
    The target values for the training dataset.

target_test : pandas.DataFrame
    The target values for the testing dataset.

target_name : str
    The name of the target variable.

estimator : object
    The machine learning model to be used for feature selection and evaluation.

metric : callable
    The metric function to evaluate the model performance.

logic : {'lower', 'greater'}
    The logic to determine whether to minimise or maximise the metric.

task_type : {'classification', 'regression'}
    The type of task to perform.
"""


    def __init__(
            self,
            df_train: pd.DataFrame,
            df_test: pd.DataFrame,
            target_train: pd.DataFrame,
            target_test: pd.DataFrame,
            estimator,
            metric,
            logic: Literal['lower', 'greater'] = 'greater',
            task_type: Literal['classification', 'regression'] = 'regression',
            ) -> None:
        """
Initialise the DescriptorExplainer with training/testing data, model, 
and evaluation settings.

Parameters
----------
df_train : pandas.DataFrame
    The training dataset.

df_test : pandas.DataFrame
    The testing dataset.

target_train : pandas.DataFrame
    The target values for the training dataset.

target_test : pandas.DataFrame
    The target values for the testing dataset.

estimator : object
    The machine learning model to be used.

metric : callable
    The evaluation metric function.

logic : {'lower', 'greater'}, optional (default='greater')
    Whether to minimise or maximise the metric.

task_type : {'classification', 'regression'}, optional (default='regression')
    The type of task to perform.
"""

        from mlchem.ml.feature_selection import wrappers

        self.task_type = task_type
        self.df_train = df_train
        self.df_test = df_test
        self.target_train = target_train
        self.target_test = target_test
        self.target_name = self.target_train.columns[0]

        self.estimator = estimator
        self.metric = metric
        self.logic = logic
        self.CombSelector = wrappers.CombinatorialSelection(
            self.estimator,
            self.metric,
            self.logic,
            self.task_type
        )

    def fit_stage_1(
            self,
            k: int = 2,
            training_threshold: float = 1.5,
            cv_train_ratio: float = 0.8,
            cv_iter: int = 5) -> None:
        """
Perform the first stage of combinatorial feature selection.

This method evaluates combinations of features using cross-validation
and filters subsets based on a training score threshold.

Parameters
----------
k : int, optional (default=2)
    The number of features to combine.

training_threshold : float, optional (default=1.5)
    The threshold for the training score to consider a subset.

cv_train_ratio : float, optional (default=0.8)
    The minimum accepted cv score/train score ratio.

cv_iter : int, optional (default=5)
    The number of cross-validation iterations.

Returns
-------
None
"""

        self.df_results_stage_1 = self.CombSelector.fit_stage_1(
            train_set=self.df_train,
            y_train=self.target_train.values,
            test_set=self.df_test,
            y_test=self.target_test.values,
            features=list(self.df_train.columns),
            k=k,
            training_threshold=training_threshold,
            cv_train_ratio=cv_train_ratio,
            cv_iter=cv_iter
        )

    def fit_stage_2(
        self,
        top_n_subsets: int = 10,
        cv_iter: int = 5
    ) -> None:
        """
Perform the second stage of combinatorial feature selection.

This method evaluates the top feature subsets from stage 1 using
cross-validation to identify the best-performing combinations.

Parameters
----------
top_n_subsets : int, optional (default=10)
    The number of top feature subsets to consider.

cv_iter : int, optional (default=5)
    The number of cross-validation iterations.

Returns
-------
None
"""

        self.df_results_stage_2 = self.CombSelector.fit_stage_2(
            top_n_subsets=top_n_subsets,
            cv_iter=cv_iter
        )

    def display(self, subset_index: int) -> None:
        """
Display the performance of the model using a selected feature subset.

This method visualises model performance using the selected feature subset.
For regression tasks, it plots true vs. predicted values and prints model
coefficients and adjusted R². For classification, it shows a confusion matrix
and prints performance metrics.

Parameters
----------
subset_index : int
    The index of the feature subset to use for evaluation.

Returns
-------
None
"""

        from mlchem.metrics import get_r2
        from mlchem.helper import count_features, sort_list_by_other_list
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        self.best_features = list(self.df_results_stage_2.
                                  feature_subsets)[subset_index]
        df_fit = self.df_train[self.best_features]

        x = df_fit.values
        y = self.target_train.values
        self.estimator.fit(x, y)
        coefs = self.estimator.coef_[0]
        intercept = self.estimator.intercept_[0]

        best_features_sorted,coefs_sorted = sort_list_by_other_list(
                                            self.best_features,coefs
                                            )
        

        if self.task_type == 'regression':
            plt.scatter(y, self.estimator.predict(x))
            plt.plot([min(y), max(y)], [min(y), max(y)], color='black')
            plt.xlabel(f'true {self.target_train.columns[0]}')
            plt.ylabel(f'pred {self.target_train.columns[0]}')

            R2 = get_r2(np.hstack(y), np.hstack(self.estimator.predict(x)))
            print(f'Train R2: {R2:.2f}')
            n = len(x)
            p = count_features(self.best_features)
            print(f'Adj Train R2: {1 - (1 - R2) * ((n - 1) / (n - p - 1)):.2f}')
            print(list(zip(best_features_sorted, np.round(coefs_sorted, 2))),
                f'intercept: {intercept:.2f}')
            print(f'std_y: {y.std():.2f}')

        else:

            y_test_pred = self.estimator.predict(self.df_test[self.best_features])
            confmat = confusion_matrix(self.target_test.values,y_test_pred)
            print(f'Train score: {list(self.df_results_stage_2.training_score)[subset_index]:.2f}')
            print(f'CV score: {list(self.df_results_stage_2.cv_score)[subset_index]:.2f}')
            print(f'Test score: {list(self.df_results_stage_2.test_score)[subset_index]:.2f}')
            print(list(zip(best_features_sorted, np.round(coefs_sorted, 2))),
                f'intercept: {intercept:.2f}')

            sns.matrix.heatmap(confmat,vmax=confmat.max(),
                               vmin=0,annot=True,fmt='d',cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        plt.show()
