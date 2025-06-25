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
