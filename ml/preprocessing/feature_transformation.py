import pandas as pd


def polynomial_expansion(
        dataframe: pd.DataFrame,
        degree: int) -> pd.DataFrame:
    """
Expand features of a DataFrame to polynomial features of a given degree.

Parameters
----------
dataframe : pandas.DataFrame
    Input DataFrame containing the original features.

degree : int
    Degree of the polynomial expansion.

Returns
-------
pandas.DataFrame
    DataFrame containing the expanded polynomial features.
"""


    if dataframe.empty:
        return pd.DataFrame()
    
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree, include_bias=False)
    poly.fit(dataframe)
    dataframe_poly = pd.DataFrame(
        data=poly.transform(dataframe),
        columns=poly.get_feature_names_out(dataframe.columns),
        index=dataframe.index
    )
    return dataframe_poly
