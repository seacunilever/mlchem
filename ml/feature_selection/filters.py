from typing import Literal
import pandas as pd
import numpy as np
from mlchem.chem.calculator.tools import shannon_entropy
from mlchem.helper import suppress_warnings
suppress_warnings()


def collinearity_filter(
    df: pd.DataFrame,
    threshold: float,
    target_variable: str = None,
    method: Literal['pearson', 'kendall', 'spearman'] = 'pearson',
    numeric_only: bool = False
    ) -> pd.DataFrame:
    """
Filter features based on collinearity threshold.

Returns a subset of DataFrame columns whose squared correlation (R²)
values are below the specified threshold. If a target variable is
provided, the function retains the feature with the higher correlation
to the target when multiple features are collinear.

Parameters
----------
df : pandas.DataFrame
    The input dataset.
threshold : float
    The maximum allowed squared correlation between features.
target_variable : str, optional
    The name of the target variable. If provided, it is used to resolve
    collinearity conflicts.
method : {'pearson', 'kendall', 'spearman'}, optional
    The correlation method to use. Default is 'pearson'.
numeric_only : bool, optional
    Whether to include only numeric columns. Default is False.

Returns
-------
pandas.DataFrame
    A DataFrame containing the filtered columns.
"""


    # Calculate the squared correlation matrix
    corr_matrix = df.corr(method=method,numeric_only=numeric_only)**2
    columns_to_keep = []

    for column in corr_matrix.columns:
        if column == target_variable:
            continue

        # Find columns with correlation above the threshold
        high_corr_columns = corr_matrix.index[
            corr_matrix[column] > threshold
            ].tolist()

        if target_variable:
            # Retain the feature with the highest correlation to the
            # target variable
            if any(col in columns_to_keep for col in high_corr_columns):
                correlations_with_target = corr_matrix.loc[
                    high_corr_columns, target_variable
                    ]
                feature_to_keep = correlations_with_target.idxmax()
                columns_to_keep = [col for col
                                   in columns_to_keep
                                   if col not in high_corr_columns
                                   or col == feature_to_keep
                                   ]
                if feature_to_keep not in columns_to_keep:
                    columns_to_keep.append(feature_to_keep)
            else:
                columns_to_keep.append(column)
        else:
            # Retain columns with correlation below the threshold
            if all(
                col not in columns_to_keep
                for col
                in high_corr_columns
                 ):
                columns_to_keep.append(column)

    # Ensure the target variable is included in the final DataFrame;
    # Remove 100% correlation edge case
    if target_variable:
        columns_to_keep = [col for col in columns_to_keep 
                           if corr_matrix.at[col, target_variable] < 1.0]
        
        if target_variable not in columns_to_keep:
            columns_to_keep.append(target_variable)
        

    return df[columns_to_keep]


def diversity_filter(
    df: pd.DataFrame,
    threshold: float,
    target_variable: str = None
) -> pd.DataFrame:
    """
Filter features based on diversity ratio using Shannon entropy.

Calculates the diversity ratio of each feature by comparing its
Shannon entropy to that of an ideal uniform distribution. Retains
features with diversity ratios above the specified threshold.

Parameters
----------
df : pandas.DataFrame
    The input dataset.
threshold : float
    The minimum diversity ratio required to retain a feature.
target_variable : str, optional
    The name of the target variable to retain regardless of its
    diversity score.

Returns
-------
pandas.DataFrame
    A DataFrame containing the filtered columns with diversity
    higher than the threshold.
"""


    diversities = {}

    for column in df.columns:
        if column != target_variable:
            actual_entropy = shannon_entropy(df[column])
            ideal_entropy = shannon_entropy(np.unique(df[column]))
            diversity = actual_entropy / \
                ideal_entropy if ideal_entropy != 0 else 0
            diversities[column] = diversity

    # Filter features based on diversity threshold
    selected_features = [column for column,
                         diversity in diversities.items() if 
                         diversity >= threshold]

    # Ensure the target variable is included in the final DataFrame
    if target_variable and target_variable not in selected_features:
        selected_features.append(target_variable)

    return df[selected_features]
