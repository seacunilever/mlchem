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


def scale_df_standard(
    df: pd.DataFrame,
    last_columns_to_preserve: int = 0
) -> tuple[pd.DataFrame, StandardScaler]:
    """
Scale a DataFrame using standard scaling, preserving specified columns.

Parameters
----------
df : pandas.DataFrame
    The input DataFrame.

last_columns_to_preserve : int, default=0
    Number of columns at the end of the DataFrame to exclude from scaling.

Returns
-------
tuple of pandas.DataFrame and StandardScaler
    The scaled DataFrame and the fitted StandardScaler.
"""


    scaler = StandardScaler()
    if last_columns_to_preserve == 0:
        sliced_df = df
    elif last_columns_to_preserve > 0:
        sliced_df = df.iloc[:, :-last_columns_to_preserve]
    else:
        raise ValueError("'last_columns_to_preserve' must be >= 0")

    sliced_columns = sliced_df.columns
    try:
        dataframe_scaled = pd.DataFrame(
            scaler.fit_transform(sliced_df),
            columns=sliced_columns,
            index=df.index
        )
    except IndexError:
        try:
            dataframe_scaled = pd.DataFrame(
                scaler.fit_transform(df.values),
                columns=df.columns,
                index=df.index
            )
        except ValueError as e:
            raise ValueError(f"Error in scaling data: {e}")
    return dataframe_scaled, scaler


def scale_df_minmax(
    df: pd.DataFrame,
    last_columns_to_preserve: int = 0
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scale a DataFrame using min-max scaling, preserving specified columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.

    last_columns_to_preserve : int, default=0
        Number of columns at the end of the DataFrame to exclude from scaling.

    Returns
    -------
    tuple of pandas.DataFrame and MinMaxScaler
        The scaled DataFrame and the fitted MinMaxScaler.
    """


    scaler = MinMaxScaler()
    if last_columns_to_preserve == 0:
        sliced_df = df
    elif last_columns_to_preserve > 0:
        sliced_df = df.iloc[:, :-last_columns_to_preserve]
    else:
        raise ValueError("'last_columns_to_preserve' must be >= 0")

    sliced_columns = sliced_df.columns
    try:
        dataframe_scaled = pd.DataFrame(
            scaler.fit_transform(sliced_df),
            columns=sliced_columns,
            index=df.index
        )
    except IndexError:
        try:
            dataframe_scaled = pd.DataFrame(
                scaler.fit_transform(df.values),
                columns=df.columns,
                index=df.index
            )
        except ValueError as e:
            raise ValueError(f"Error in scaling data: {e}")
    return dataframe_scaled, scaler


def scale_df_robust(
    df: pd.DataFrame,
    last_columns_to_preserve: int = 0
) -> tuple[pd.DataFrame, RobustScaler]:
    """
Scale a DataFrame using robust scaling, preserving specified columns.

Parameters
----------
df : pandas.DataFrame
    The input DataFrame.

last_columns_to_preserve : int, default=0
    Number of columns at the end of the DataFrame to exclude from scaling.

Returns
-------
tuple of pandas.DataFrame and RobustScaler
    The scaled DataFrame and the fitted RobustScaler.
"""


    scaler = RobustScaler()
    if last_columns_to_preserve == 0:
        sliced_df = df
    elif last_columns_to_preserve > 0:
        sliced_df = df.iloc[:, :-last_columns_to_preserve]
    else:
        raise ValueError("'last_columns_to_preserve' must be >= 0")

    sliced_columns = sliced_df.columns
    try:
        dataframe_scaled = pd.DataFrame(
            scaler.fit_transform(sliced_df),
            columns=sliced_columns,
            index=df.index
        )
    except IndexError:
        try:
            dataframe_scaled = pd.DataFrame(
                scaler.fit_transform(df.values),
                columns=df.columns,
                index=df.index
            )
        except ValueError as e:
            raise ValueError(f"Error in scaling data: {e}")
    return dataframe_scaled, scaler


def transform_df(
    df: pd.DataFrame,
    scaler: StandardScaler | MinMaxScaler | RobustScaler,
    last_columns_to_preserve: int
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

Returns
-------
tuple of pandas.DataFrame and scaler
    The transformed DataFrame and the scaler used.
"""

    if last_columns_to_preserve == 0:
        sliced_df = df
    elif last_columns_to_preserve > 0:
        sliced_df = df.iloc[:, :-last_columns_to_preserve]
    else:
        raise ValueError("'last_columns_to_preserve' must be >= 0")

    sliced_columns = sliced_df.columns
    try:
        dataframe_transformed = pd.DataFrame(
            scaler.transform(sliced_df),
            columns=sliced_columns,
            index=df.index
        )
    except IndexError:
        try:
            dataframe_transformed = pd.DataFrame(
                scaler.transform(df.values),
                columns=df.columns,
                index=df.index
            )
        except ValueError as e:
            raise ValueError(f"Error in scaling data: {e}")
    return dataframe_transformed, scaler
