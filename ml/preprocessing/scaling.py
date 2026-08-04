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
from sklearn.preprocessing import (StandardScaler,
                                   MinMaxScaler,
                                   RobustScaler)


def _is_binary_column(series: pd.Series) -> bool:
    unique_values = pd.unique(series.dropna())
    if len(unique_values) == 0:
        return False
    return set(unique_values).issubset({0, 1, 0.0, 1.0, False, True})


def _split_scale_columns(
    df: pd.DataFrame,
    last_columns_to_preserve: int,
    skip_binary_columns: bool
) -> tuple[pd.DataFrame, list[str]]:
    if last_columns_to_preserve == 0:
        sliced_df = df
    elif last_columns_to_preserve > 0:
        sliced_df = df.iloc[:, :-last_columns_to_preserve]
    else:
        raise ValueError("'last_columns_to_preserve' must be >= 0")

    if skip_binary_columns:
        columns_to_scale = [
            column for column in sliced_df.columns
            if not _is_binary_column(sliced_df[column])
        ]
    else:
        columns_to_scale = sliced_df.columns.to_list()

    return sliced_df, columns_to_scale


def _fit_transform_columns(
    scaler: StandardScaler | MinMaxScaler | RobustScaler,
    sliced_df: pd.DataFrame,
    columns_to_scale: list[str],
    index: pd.Index
) -> pd.DataFrame:
    if not columns_to_scale:
        return pd.DataFrame(index=index)

    scale_df = sliced_df[columns_to_scale]
    try:
        return pd.DataFrame(
            scaler.fit_transform(scale_df),
            columns=columns_to_scale,
            index=index
        )
    except IndexError:
        try:
            return pd.DataFrame(
                scaler.fit_transform(scale_df.values),
                columns=columns_to_scale,
                index=index
            )
        except ValueError as e:
            raise ValueError(f"Error in scaling data: {e}")


def _transform_columns(
    scaler: StandardScaler | MinMaxScaler | RobustScaler,
    sliced_df: pd.DataFrame,
    columns_to_scale: list[str],
    index: pd.Index
) -> pd.DataFrame:
    if not columns_to_scale:
        return pd.DataFrame(index=index)

    scale_df = sliced_df[columns_to_scale]
    try:
        return pd.DataFrame(
            scaler.transform(scale_df),
            columns=columns_to_scale,
            index=index
        )
    except IndexError:
        try:
            return pd.DataFrame(
                scaler.transform(scale_df.values),
                columns=columns_to_scale,
                index=index
            )
        except ValueError as e:
            raise ValueError(f"Error in scaling data: {e}")


def scale_df_standard(
    df: pd.DataFrame,
    last_columns_to_preserve: int = 0,
    skip_binary_columns: bool = False
) -> tuple[pd.DataFrame, StandardScaler]:
    """
Scale a DataFrame using standard scaling, preserving specified columns.

Parameters
----------
df : pandas.DataFrame
    The input DataFrame.

last_columns_to_preserve : int, default=0
    Number of columns at the end of the DataFrame to exclude from scaling.

skip_binary_columns : bool, default=False
    If True, binary 0/1 columns are excluded from scaling.

Returns
-------
tuple of pandas.DataFrame and StandardScaler
    The scaled DataFrame and the fitted StandardScaler.
"""
    scaler = StandardScaler()
    sliced_df, columns_to_scale = _split_scale_columns(
        df=df,
        last_columns_to_preserve=last_columns_to_preserve,
        skip_binary_columns=skip_binary_columns,
    )
    dataframe_scaled = _fit_transform_columns(
        scaler=scaler,
        sliced_df=sliced_df,
        columns_to_scale=columns_to_scale,
        index=df.index,
    )

    # Build output: scaled columns + skipped binary columns (reintroduced)
    scaled_dfs = [dataframe_scaled] if not dataframe_scaled.empty else []
    
    # Reintroduce binary feature columns that were skipped from scaling
    if skip_binary_columns:
        binary_feature_cols = [
            col for col in sliced_df.columns
            if col not in columns_to_scale and _is_binary_column(sliced_df[col])
        ]
        if binary_feature_cols:
            scaled_dfs.append(sliced_df[binary_feature_cols])
    
    # Add preserved columns (end of dataframe), but exclude binary target variables
    if last_columns_to_preserve > 0:
        preserved_cols = df.iloc[:, -last_columns_to_preserve:]
        if skip_binary_columns:
            # Only include non-binary preserved columns (exclude binary target variables)
            preserved_cols = preserved_cols[[
                col for col in preserved_cols.columns 
                if not _is_binary_column(preserved_cols[col])
            ]]
        if not preserved_cols.empty:
            scaled_dfs.append(preserved_cols)
    
    if scaled_dfs:
        scaled_output = pd.concat(scaled_dfs, axis=1)
    else:
        scaled_output = pd.DataFrame(index=df.index)

    scaler._mlchem_columns_scaled = columns_to_scale
    scaler._mlchem_last_columns_to_preserve = last_columns_to_preserve
    scaler._mlchem_skip_binary_columns = skip_binary_columns

    return scaled_output, scaler


def scale_df_minmax(
    df: pd.DataFrame,
    last_columns_to_preserve: int = 0,
    skip_binary_columns: bool = False
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scale a DataFrame using min-max scaling, preserving specified columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.

    last_columns_to_preserve : int, default=0
        Number of columns at the end of the DataFrame to exclude from scaling.

    skip_binary_columns : bool, default=False
        If True, binary 0/1 columns are excluded from scaling.

    Returns
    -------
    tuple of pandas.DataFrame and MinMaxScaler
        The scaled DataFrame and the fitted MinMaxScaler.
    """
    scaler = MinMaxScaler()
    sliced_df, columns_to_scale = _split_scale_columns(
        df=df,
        last_columns_to_preserve=last_columns_to_preserve,
        skip_binary_columns=skip_binary_columns,
    )
    dataframe_scaled = _fit_transform_columns(
        scaler=scaler,
        sliced_df=sliced_df,
        columns_to_scale=columns_to_scale,
        index=df.index,
    )

    # Build output: scaled columns + skipped binary columns (reintroduced)
    scaled_dfs = [dataframe_scaled] if not dataframe_scaled.empty else []
    
    # Reintroduce binary feature columns that were skipped from scaling
    if skip_binary_columns:
        binary_feature_cols = [
            col for col in sliced_df.columns
            if col not in columns_to_scale and _is_binary_column(sliced_df[col])
        ]
        if binary_feature_cols:
            scaled_dfs.append(sliced_df[binary_feature_cols])
    
    # Add preserved columns (end of dataframe), but exclude binary target variables
    if last_columns_to_preserve > 0:
        preserved_cols = df.iloc[:, -last_columns_to_preserve:]
        if skip_binary_columns:
            # Only include non-binary preserved columns (exclude binary target variables)
            preserved_cols = preserved_cols[[
                col for col in preserved_cols.columns 
                if not _is_binary_column(preserved_cols[col])
            ]]
        if not preserved_cols.empty:
            scaled_dfs.append(preserved_cols)
    
    if scaled_dfs:
        scaled_output = pd.concat(scaled_dfs, axis=1)
    else:
        scaled_output = pd.DataFrame(index=df.index)

    scaler._mlchem_columns_scaled = columns_to_scale
    scaler._mlchem_last_columns_to_preserve = last_columns_to_preserve
    scaler._mlchem_skip_binary_columns = skip_binary_columns

    return scaled_output, scaler


def scale_df_robust(
    df: pd.DataFrame,
    last_columns_to_preserve: int = 0,
    skip_binary_columns: bool = False
) -> tuple[pd.DataFrame, RobustScaler]:
    """
Scale a DataFrame using robust scaling, preserving specified columns.

Parameters
----------
df : pandas.DataFrame
    The input DataFrame.

last_columns_to_preserve : int, default=0
    Number of columns at the end of the DataFrame to exclude from scaling.

skip_binary_columns : bool, default=False
    If True, binary 0/1 columns are excluded from scaling.

Returns
-------
tuple of pandas.DataFrame and RobustScaler
    The scaled DataFrame and the fitted RobustScaler.
"""
    scaler = RobustScaler()
    sliced_df, columns_to_scale = _split_scale_columns(
        df=df,
        last_columns_to_preserve=last_columns_to_preserve,
        skip_binary_columns=skip_binary_columns,
    )
    dataframe_scaled = _fit_transform_columns(
        scaler=scaler,
        sliced_df=sliced_df,
        columns_to_scale=columns_to_scale,
        index=df.index,
    )

    # Build output: scaled columns + skipped binary columns (reintroduced)
    scaled_dfs = [dataframe_scaled] if not dataframe_scaled.empty else []
    
    # Reintroduce binary feature columns that were skipped from scaling
    if skip_binary_columns:
        binary_feature_cols = [
            col for col in sliced_df.columns
            if col not in columns_to_scale and _is_binary_column(sliced_df[col])
        ]
        if binary_feature_cols:
            scaled_dfs.append(sliced_df[binary_feature_cols])
    
    # Add preserved columns (end of dataframe), but exclude binary target variables
    if last_columns_to_preserve > 0:
        preserved_cols = df.iloc[:, -last_columns_to_preserve:]
        if skip_binary_columns:
            # Only include non-binary preserved columns (exclude binary target variables)
            preserved_cols = preserved_cols[[
                col for col in preserved_cols.columns 
                if not _is_binary_column(preserved_cols[col])
            ]]
        if not preserved_cols.empty:
            scaled_dfs.append(preserved_cols)
    
    if scaled_dfs:
        scaled_output = pd.concat(scaled_dfs, axis=1)
    else:
        scaled_output = pd.DataFrame(index=df.index)

    scaler._mlchem_columns_scaled = columns_to_scale
    scaler._mlchem_last_columns_to_preserve = last_columns_to_preserve
    scaler._mlchem_skip_binary_columns = skip_binary_columns

    return scaled_output, scaler


def transform_df(
    df: pd.DataFrame,
    scaler: StandardScaler | MinMaxScaler | RobustScaler,
    last_columns_to_preserve: int,
    skip_binary_columns: bool = False
) -> tuple[pd.DataFrame, StandardScaler | MinMaxScaler | RobustScaler]:
    """
Transform a DataFrame using a provided scaler, preserving specified columns.

Parameters
----------
df : pandas.DataFrame
    The input DataFrame.

scaler : StandardScaler or MinMaxScaler or RobustScaler
    The fitted scaler to use for transformation.

last_columns_to_preserve : int
    Number of columns at the end of the DataFrame to exclude from 
    transformation.

skip_binary_columns : bool, default=False
    If True, binary 0/1 columns are excluded from transformation.

Returns
-------
tuple of pandas.DataFrame and scaler
    The transformed DataFrame and the scaler used.
"""
    sliced_df, derived_columns_to_scale = _split_scale_columns(
        df=df,
        last_columns_to_preserve=last_columns_to_preserve,
        skip_binary_columns=skip_binary_columns,
    )

    fitted_columns = getattr(scaler, '_mlchem_columns_scaled', None)
    if isinstance(fitted_columns, list):
        columns_to_scale = [
            column for column in fitted_columns
            if column in sliced_df.columns
        ]
    else:
        columns_to_scale = derived_columns_to_scale

    dataframe_transformed = _transform_columns(
        scaler=scaler,
        sliced_df=sliced_df,
        columns_to_scale=columns_to_scale,
        index=df.index,
    )

    # Build output: transformed columns + skipped binary columns (reintroduced)
    transformed_dfs = [dataframe_transformed] if not dataframe_transformed.empty else []
    
    # Reintroduce binary feature columns that were skipped from transformation
    if skip_binary_columns:
        binary_feature_cols = [
            col for col in sliced_df.columns
            if col not in columns_to_scale and _is_binary_column(sliced_df[col])
        ]
        if binary_feature_cols:
            transformed_dfs.append(sliced_df[binary_feature_cols])
    
    # Add preserved columns (end of dataframe), but exclude binary target variables
    if last_columns_to_preserve > 0:
        preserved_cols = df.iloc[:, -last_columns_to_preserve:]
        if skip_binary_columns:
            # Only include non-binary preserved columns (exclude binary target variables)
            preserved_cols = preserved_cols[[
                col for col in preserved_cols.columns 
                if not _is_binary_column(preserved_cols[col])
            ]]
        if not preserved_cols.empty:
            transformed_dfs.append(preserved_cols)
    
    if transformed_dfs:
        transformed_output = pd.concat(transformed_dfs, axis=1)
    else:
        transformed_output = pd.DataFrame(index=df.index)

    return transformed_output, scaler
