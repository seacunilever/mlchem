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

import matplotlib.pyplot as plt
from mlchem.helper import loadingbar, generate_combination_cascade
from mlchem.ml.modelling.model_evaluation import crossval


class SequentialForwardSelection:
    """
  Sequential Forward Feature Selection wrapper.

  This class performs Sequential Forward Feature Selection by iteratively
  adding features that yield the highest gain in cross-validation score.

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
      Number of cross-validation iterations. Default is 5.
  logic : {'lower', 'greater'}, optional
      Whether to minimize or maximize the cross-validation score. Default is 'greater'.
  task_type : {'classfication', 'regression'}, optional
      Type of task. Default is 'classification'.

  Examples
  --------
  >>> import pandas as pd
  >>> import numpy as np
  >>> from sklearn.linear_model import LogisticRegression
  >>> from sklearn.datasets import make_classification
  >>> from mlchem.metrics import get_geometric_S

  >>> sfs = SequentialForwardSelection(estimator=LogisticRegression(),
  ...                                  metric=get_geometric_S,
  ...                                  max_features=5,
  ...                                  cv_iter=3,
  ...                                  logic='greater')

  >>> X, y = make_classification(300, 10, n_informative=5)
  >>> train_size = 0.8
  >>> train_samples = int(train_size * len(X))
 
  >>> X_train, y_train = X[:train_samples], y[:train_samples]
  >>> X_test, y_test = X[train_samples:], y[train_samples:]

  >>> train_set = pd.DataFrame(X_train, columns=np.arange(X_train.shape[1]))
  >>> test_set = pd.DataFrame(X_test, columns=np.arange(X_test.shape[1]))

  >>> sfs.fit(train_set, y_train, test_set, y_test)
  >>> sfs.plot(best_feature='auto')
  """

    def __init__(self,
                 estimator,
                 estimator_string: Optional[str],
                 metric: Callable,
                 max_features: int = 25,
                 cv_iter: int = 5,
                 logic: Literal['lower', 'greater'] = 'greater',
                 task_type: Literal[
                     'classfication', 'regression'] = 'classification'
                 ) -> None:
        """
  Initialise the SequentialForwardSelection object.

  Parameters
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
      Number of cross-validation iterations. Default is 5.
  logic : {'lower', 'greater'}, optional
      Whether to minimise or maximise the cross-validation score. Default is 'greater'.
  task_type : {'classfication', 'regression'}, optional
      Type of task. Default is 'classification'.
  """

        self.estimator = estimator
        if not estimator_string:
            estimator_string = str(estimator)
        self.estimator_string = estimator_string
        self.metric = metric
        self.max_features = max_features
        self.cv_iter = cv_iter
        self.logic = logic
        self.task_type = task_type

        # Where to store the temporarily best feature set at each iteration
        self.extending_features = []

        # Where to store all training scores obtained from the model
        # using the accepted features
        self.train_scores = []

        # Where to store all cross-validation scores obtained from the
        # model using the accepted features
        self.cv_scores = []

        # Where to store the standard deviations of the cv scores
        self.cv_stds = []

        # Where to store test scores
        self.unseen_scores = []

    def fit(
        self,
        train_set: pd.DataFrame,
        y_train: Iterable,
        test_set: pd.DataFrame,
        y_test: Iterable
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

        Returns
        -------
        None
        """

        self.train_set = train_set
        self.y_train = y_train
        self.test_set = test_set
        self.y_test = y_test
        self.feature_labels = self.train_set.columns

        for cycle in range(self.max_features):

            # Temporary lists where to store cross-validation scores
            # and standard deviations.
            cv_scores_storage = []
            cv_stds_storage = []

            # List of features to be assessed
            self.list_available_features = [feat for feat in
                                            self.feature_labels if
                                            feat not in self.extending_features
                                            ]

            # Hypothetically assess model if an extra feature is added.
            # Do it for all unexplored features.
            for feat in self.list_available_features:
                features_to_test = self.extending_features + [feat]
                train_set_temp = self.train_set[features_to_test]
                self.estimator.fit(train_set_temp, self.y_train)
                cvscores = crossval(self.estimator, train_set_temp.values,
                                    y_train, self.metric,
                                    self.cv_iter, self.task_type)
                cv_scores_storage.append(np.mean(cvscores))
                cv_stds_storage.append(np.std(cvscores))

            # Include in the model the feature with best CV gains.
            if self.logic == 'greater':
                index = np.argmax(cv_scores_storage)
            else:
                index = np.argmin(cv_scores_storage)

            self.cv_scores.append(cv_scores_storage[index])
            self.cv_stds.append(cv_stds_storage[index])
            feature_to_add = self.list_available_features[index]
            self.extending_features.append(feature_to_add)
            loadingbar(cycle + 1, self.max_features, 50)

            # Get score on unseen test data
            train_set_temp = self.train_set[self.extending_features]
            test_set_temp = self.test_set[self.extending_features]
            self.estimator.fit(train_set_temp, y_train)
            y_train_pred = self.estimator.predict(train_set_temp)
            y_test_pred = self.estimator.predict(test_set_temp)
            self.train_scores.append(self.metric(self.y_train, y_train_pred))
            self.unseen_scores.append(self.metric(self.y_test, y_test_pred))

    def find_best(self, which: Optional[int] = None) -> dict:
        """
        Find the best feature subset based on evaluation criteria.

        Parameters
        ----------
        which : int, optional
            If specified, returns the feature subset at the given index.
            If None, the best subset is determined automatically using a 
            scoring algorithm.

        Returns
        -------
        dict
            A dictionary containing:
            - 'best_score': float
            - 'variability_contribution': float
            - 'geometric_contribution': float
            - 'train_test_difference': float
            - 'best_index': int
            - 'features': list

        Notes
        -----
        The automatic algorithm works as follows:

        1. Calculate the standard deviation of the training, 
        cross-validation, and unseen test scores for each feature subset.
        2. Initialise the best score and best index to zero.
        3. Define coefficients for variability contribution, percentile, 
        and contributions from training, cross-validation, and unseen test scores.
        4. Iterate through each feature subset up to the maximum number 
        of features:
            - Add a variability contribution if the standard deviation is 
            below a certain percentile.
            - Add a geometric contribution based on the product of the 
            training, cross-validation, and unseen test scores.
            - Add the absolute difference between the training and unseen 
            test scores.
            - Update the best score and best index if the current total 
            score is higher than the best score.
        """

        if which is None:

            # Apply algorithm to select the best feature subset.
            index = 0
            deviations = [np.std([tr, v, t]) for tr, v, t in
                          zip(self.train_scores,
                              self.cv_scores,
                              self.unseen_scores)
                          ]

            best_score = 0
            best_index = 0

            variability_contribution = 1.5
            percentile = 20
            train_coef = 1.5
            cv_coef = 1.5
            unseen_coef = 2

            while index < self.max_features:
                total_score = 0

                # Variability contribution
                if deviations[index] < np.percentile(deviations, percentile):
                    total_score += variability_contribution

                # Global contribution
                geometric_contribution = 10*(
                    train_coef *     # train score importance
                    self.train_scores[index] *     # train score
                    cv_coef *     # cv score importance
                    self.cv_scores[index] *     # cv score
                    unseen_coef *     # test score importance
                    self.unseen_scores[index]     # test score
                    )**(1/3)

                total_score += geometric_contribution

                train_test_difference = abs(self.train_scores[index] -
                                            self.unseen_scores[index]
                                            )
                total_score += train_test_difference
                index += 1     # evaluate next feature subset
                if total_score > best_score:
                    best_score = total_score
                    best_index = index
            best_index = best_index
            dictionary = {'best_score': best_score,
                          'variability_contribution': variability_contribution,
                          'geometric_contribution': geometric_contribution,
                          'train_test_difference': train_test_difference,
                          'best_index': best_index,
                          'features': self.extending_features[:best_index]
                          }
        else:     # if which == int
            best_index = which
            dictionary = {'best_index': best_index,
                          'features': self.extending_features[:best_index]
                          }
        return dictionary

    def plot(
        self,
        best_feature: Optional[int] = None,
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

        Parameters
        ----------
        best_feature : int, optional
            Index of the best feature subset to highlight. If None, it 
            is determined automatically.
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
        is the same as described in `find_best`.
        """

        assert best_feature is None or isinstance(best_feature, int), \
            "'best_feature_ must be either an integer or None'."

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
                         np.array(self.cv_scores) - np.array(self.cv_stds),
                         np.array(self.cv_scores) + np.array(self.cv_stds),
                         alpha=0.2, color=colours[1])
        # Plot test scores
        plt.plot(range(1, len(self.train_scores)+1),
                 self.unseen_scores,
                 label='test score',
                 color=colours[2])

        plt.legend(fontsize=legendsize, loc='best')

        ind = self.find_best(which=best_feature)['best_index']
        colours = [
            'steelblue',
            'orange',
            'green',
            'black',
                   ]

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

        print('Number of features:', ind)
        print(f'Winner feature subset: {self.extending_features[:ind]}')
        print(f'Train Score: {self.train_scores[ind - 1]:.3f}')
        print(f'CV Score: {self.cv_scores[ind - 1]:.3f} ± {self.cv_stds[ind - 1]:.3f}')
        print(f'Test Score: {self.unseen_scores[ind - 1]:.3f}')


class CombinatorialSelection:
    """
    Combinatorial feature selection using a given estimator and metric.

    This class performs a two-stage combinatorial feature selection process
    to identify optimal feature subsets based on model performance.

    Attributes
    ----------
    estimator : object
        The machine learning estimator used to fit the data.
    metric : callable
        A metric function to evaluate estimator performance. Must accept 
        (y_true, y_pred).
    logic : {'greater', 'lower'}
        Determines whether a higher or lower score is considered better.
    task_type : {'classification', 'regression'}
        Specifies the type of task.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> from mlchem.metrics import get_geometric_S

    >>> cs = CombinatorialSelection(estimator=LogisticRegression(),
    ...                              metric=get_geometric_S,
    ...                              logic='greater')

    >>> X, y = make_classification(500, 10, n_informative=4)
    >>> X_train, y_train = X[:350], y[:350]
    >>> X_test, y_test = X[350:], y[350:]
  
    >>> train_set = pd.DataFrame(X_train, columns=np.arange(X_train.shape[1]))
    >>> test_set = pd.DataFrame(X_test, columns=np.arange(X_test.shape[1]))

    >>> results_stage_1 = cs.fit_stage_1(train_set, y_train, test_set, y_test,
    ...                                  train_set.columns, training_threshold=0.7)
    >>> results_stage_2 = cs.fit_stage_2(top_n_subsets=10, cv_iter=5)
    """

    def __init__(self,
                 estimator,
                 metric,
                 logic: Literal['lower', 'greater'] = 'greater',
                 task_type: Literal[
                     'classification', 'regression'
                     ] = 'classification'
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
        """

        self.estimator = estimator
        self.metric = metric
        self.logic = logic
        self.task_type = task_type

    def fit_stage_1(
        self,
        train_set: pd.DataFrame,
        y_train: Iterable,
        test_set: pd.DataFrame,
        y_test: Iterable,
        features: list[str] = [],
        k: int = 2,
        training_threshold: float = 0.25,
        cv_train_ratio: float = 0.7,
        cv_iter: int = 5
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

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the results of the first stage of 
            feature selection.

        Notes
        -----
        - Generates all possible feature subsets of size `k`.
        - Evaluates each subset using training, cross-validation, and 
        test scores.
        - Filters and ranks subsets based on geometric mean of scores.
        """


        def is_better(a: float | int, b: float | int) -> bool:
            return a > b if self.logic == 'greater' else a < b

        self.train_set = train_set
        self.y_train = y_train
        self.test_set = test_set
        self.y_test = y_test

        self.features = features
        self.k = k
        self.training_threshold = training_threshold
        self.cv_train_ratio = cv_train_ratio
        self.cv_iter = cv_iter

        assert 0 <= self.cv_train_ratio <= 1, \
            "'cv_train_ratio' must be between 0 and 1."

        # Set cv threshold based on the desired cv/train ratio
        self.cv_threshold = self.training_threshold * self.cv_train_ratio \
            if self.logic == 'greater' else \
            self.training_threshold / self.cv_train_ratio
        
        self.ascending_decision = False if self.logic == 'greater' else \
        True

        self.feature_subsets = generate_combination_cascade(self.features,
                                                            self.k)

        self.dict_results = {
            'feature_subsets': [],
            'training_score': [],
            'cv_score': [],
            'test_score': []
            }

        for i, subset in enumerate(self.feature_subsets):
            loadingbar(i+1,
                       len(self.feature_subsets),
                       50)
            x = self.train_set[subset]
            self.estimator.fit(x.values, self.y_train)
            y_train_pred = self.estimator.predict(x.values)
            train_score = self.metric(self.y_train, y_train_pred)
            if not is_better(train_score, self.training_threshold):
                pass
            else:
                cv_score = crossval(
                    self.estimator,
                    x,
                    y_train,
                    self.metric,
                    self.cv_iter,
                    self.task_type
                    ).mean()

                if not is_better(cv_score, self.cv_threshold):
                    pass
                else:
                    self.dict_results['feature_subsets'].append(subset)
                    y_test_pred = self.estimator.predict(self.test_set[subset])
                    test_score = self.metric(self.y_test, y_test_pred)
                    self.dict_results['training_score'].append(train_score)
                    self.dict_results['cv_score'].append(cv_score)
                    self.dict_results['test_score'].append(test_score)
        self.df_results_stage1 = pd.DataFrame(
            self.dict_results,
            columns=self.dict_results.keys()
            )
        self.df_results_stage1['geometric_mean'] = (
            self.df_results_stage1.training_score*
            self.df_results_stage1.cv_score*
            self.df_results_stage1.test_score
            )**(1/3)
        self.df_results_stage1.sort_values(
            by='geometric_mean',
            ascending=self.ascending_decision,
            inplace=True)
        return self.df_results_stage1

    def fit_stage_2(self,
                    top_n_subsets: int = 10,
                    cv_iter: int = 5) -> pd.DataFrame:
        """
        Perform the second stage of combinatorial feature selection.

        Parameters
        ----------
        top_n_subsets : int, optional
            Number of top feature subsets from stage 1 to consider. 
            Default is 10.
        cv_iter : int, optional
            Number of cross-validation iterations. Default is 5.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the results of the second stage of 
            feature selection.

        Notes
        -----
        - Identifies most recurrent features from top subsets.
        - Generates new combinations and evaluates them.
        - Filters and ranks based on geometric mean of scores.
        """

        def is_better(a: float | int, b: float | int) -> bool:
            return a > b if self.logic == 'greater' else a < b

        self.cv_iter = cv_iter
        self.best_recurrent = np.unique(
            np.hstack(
                self.df_results_stage1.head(top_n_subsets).
                feature_subsets.values)
                )

        self.feature_subsets = generate_combination_cascade(
            self.best_recurrent, top_n_subsets
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

        for i, subset in enumerate(self.feature_subsets):
            loadingbar(i+1, len(self.feature_subsets), 50)
            x = self.train_set[subset]
            self.estimator.fit(x.values, self.y_train)
            y_train_pred = self.estimator.predict(x.values)
            train_score = self.metric(self.y_train, y_train_pred)
            if not is_better(train_score, self.training_threshold_2):
                pass
            else:
                cv_score = crossval(self.estimator,
                                    x,
                                    self.y_train,
                                    self.metric,
                                    self.cv_iter,
                                    self.task_type
                                    ).mean()
                if not is_better(cv_score, self.cv_threshold_2):
                    pass
                else:
                    self.dict_results_2['feature_subsets'].append(subset)
                    y_test_pred = self.estimator.predict(self.test_set[subset])
                    test_score = self.metric(self.y_test, y_test_pred)
                    self.dict_results_2['training_score'].append(train_score)
                    self.dict_results_2['cv_score'].append(cv_score)
                    self.dict_results_2['test_score'].append(test_score)
        self.df_results_stage2 = pd.DataFrame(
            self.dict_results_2, columns=self.dict_results_2.keys()
            )
        self.df_results_stage2['geometric_mean'] = (
            self.df_results_stage2.training_score*
            self.df_results_stage2.cv_score*
            self.df_results_stage2.test_score
            )**(1/3)
        self.df_results_stage2.sort_values(
            by='geometric_mean',
            ascending=self.ascending_decision,
            inplace=True)
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
            self.task_type
        )

        # Display results
        print(f'# of Features: {len(self.best_cols)}')
        print(f'Best Features: {self.best_cols}')
        print(f'Train Score: {self.metric(self.y_train,
                                          self.y_train_pred):.3f}')
        print(f'CV Score: {self.cv_performance.
                           mean():.3f} ± {self.cv_performance.std():.3f}')
        print(f'Test Score: {self.metric(self.y_test, self.y_test_pred):.3f}')
