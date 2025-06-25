from typing import Literal
import pandas as pd


class Compressor:
    """
A class for compressing dataframes using various dimensionality reduction techniques.

This class provides a unified interface to apply multiple dimensionality
reduction algorithms such as PCA, t-SNE, UMAP, Spectral Embedding, MDS,
LLE, and Isomap. It supports optional exclusion of initial columns from
compression and allows passing custom parameters to each algorithm.

Attributes
----------
dataframe : pandas.DataFrame
    The input dataframe to be compressed.

initial_columns_to_ignore : int, optional (default=0)
    Number of initial columns to exclude from compression.

algorithm : object
    The dimensionality reduction algorithm instance used for compression.

params_ : dict or None
    Parameters of the compression algorithm after initialization.

X_compressed : numpy.ndarray
    The compressed feature matrix.

dataframe_compressed : pandas.DataFrame
    The dataframe after applying dimensionality reduction.

Methods
-------
compress_PCA(...)
    Compress the dataframe using Principal Component Analysis.

compress_TSNE(...)
    Compress the dataframe using t-Distributed Stochastic Neighbor Embedding.

compress_SE(...)
    Compress the dataframe using Spectral Embedding.

compress_UMAP(...)
    Compress the dataframe using Uniform Manifold Approximation and Projection.

compress_MDS(...)
    Compress the dataframe using Multidimensional Scaling.

compress_LLE(...)
    Compress the dataframe using Locally Linear Embedding.

compress_ISOMAP(...)
    Compress the dataframe using Isomap.

Example
--------
>>> from mlchem.chem.calculator.descriptors import get_rdkitDesc
>>> from mlchem.ml.preprocessing import scaling
>>> df = get_rdkitDesc(['CCCC', 'CCN', 'c1ccccc1', 'CF', 'CCO', 'CCCNC(OCCC)CCO'])
>>> df = scaling.scale_df_standard(df, last_columns_to_preserve=0)[0]
>>> c = Compressor(df)
>>> c.compress_PCA(n_components_or_variance=0.6)
>>> df_pca = c.dataframe_compressed
>>> c.compress_TSNE(dataframe=df_pca, random_state=1)
>>> compressed_df = c.dataframe_compressed
"""

    def __init__(self, dataframe: pd.DataFrame,
                 initial_columns_to_ignore: int = 0):
        """
Initialize the Compressor with a dataframe and optional column exclusion.

Parameters
----------
dataframe : pandas.DataFrame
    The input dataframe to be compressed.

initial_columns_to_ignore : int, optional (default=0)
    Number of initial columns to exclude from compression.
"""

        self.dataframe = dataframe
        self.initial_columns_to_ignore = initial_columns_to_ignore
        self.algorithm = None
        self.params_ = None

    def compress_PCA(self,
                     n_components_or_variance: int | float = 0.8,
                     svd_solver: Literal['auto',
                                         'full',
                                         'arpack',
                                         'randomized'] = 'full',
                     dataframe: pd.DataFrame | None = None,
                     random_state: int = 1,
                     dict_params: dict | None = None) -> None:
        """
Compress the dataframe using Principal Component Analysis (PCA).

This method reduces the dimensionality of the dataframe using PCA,
either by specifying the number of components or the amount of variance
to retain.

Parameters
----------
n_components_or_variance : int or float, optional (default=0.8)
    Number of components to keep or the amount of variance to retain.

svd_solver : {'auto', 'full', 'arpack', 'randomized'}, optional (default='full')
    SVD solver to use for the decomposition.

dataframe : pandas.DataFrame or None, optional
    DataFrame to compress. If None, uses the instance's dataframe.

random_state : int, optional (default=1)
    Random seed for reproducibility.

dict_params : dict or None, optional
    Dictionary of parameters to pass directly to the PCA constructor.

Returns
-------
None
"""

        from sklearn.decomposition import PCA

        self.algorithm = PCA(**dict_params) if dict_params else PCA(
            n_components=n_components_or_variance,
            svd_solver=svd_solver,
            random_state=random_state
        )
        self.params_ = self.algorithm.get_params()
        df = dataframe if dataframe is not None else self.dataframe
        self.X_compressed = self.algorithm.fit_transform(
            df.values[:, self.initial_columns_to_ignore:]
            )
        self.dataframe_compressed = pd.concat(
            [pd.DataFrame(df.iloc[:, :self.initial_columns_to_ignore],
                          columns=df.columns[:self.initial_columns_to_ignore]),
             pd.DataFrame(self.X_compressed,
                          index=df.index,
                          columns=[f'DIM_{i+1}' for i in range(self.
                                                               X_compressed.
                                                               shape[1])])],
            axis=1
        )

    def compress_TSNE(self,
                      n_components: int = 2,
                      neighbours_number_or_fraction: float | int = 0.9,
                      dataframe: pd.DataFrame | None = None,
                      random_state: int = 1,
                      dict_params: dict | None = None) -> None:
        """
Compress the dataframe using t-Distributed Stochastic Neighbor Embedding (t-SNE).

This method applies t-SNE to reduce the dimensionality of the dataframe
based on neighborhood probabilities.

Parameters
----------
n_components : int, optional (default=2)
    Number of dimensions to reduce to.

neighbours_number_or_fraction : float or int, optional (default=0.9)
    Number or fraction of neighbors to consider for perplexity.

dataframe : pandas.DataFrame or None, optional
    DataFrame to compress. If None, uses the instance's dataframe.

random_state : int, optional (default=1)
    Random seed for reproducibility.

dict_params : dict or None, optional
    Dictionary of parameters to pass directly to the t-SNE constructor.

Returns
-------
None
"""
        from sklearn.manifold import TSNE

        if dict_params is None:
            if isinstance(neighbours_number_or_fraction, float):
                assert 0 < neighbours_number_or_fraction <= 1, (
                    "'neighbours_number_or_fraction' must be between 0 and 1 "
                    "if float."
                )
                self.n_neighbours = round(neighbours_number_or_fraction *
                                          len(self.dataframe))
            else:
                assert neighbours_number_or_fraction < len(self.dataframe), (
                    "'neighbours_number_or_fraction' must be less than the "
                    "number of samples if int."
                )
                self.n_neighbours = neighbours_number_or_fraction
            self.algorithm = TSNE(
                n_components=n_components,
                perplexity=self.n_neighbours,
                init='random',
                n_jobs=-1,
                random_state=random_state
            )
        else:
            self.algorithm = TSNE(**dict_params)
        self.params_ = self.algorithm.get_params()
        df = dataframe if dataframe is not None else self.dataframe
        self.X_compressed = self.algorithm.fit_transform(
            df.values[:, self.initial_columns_to_ignore:]
        )
        self.dataframe_compressed = pd.concat(
            [pd.DataFrame(df.iloc[:, :self.initial_columns_to_ignore],
                          columns=df.columns[:self.initial_columns_to_ignore]),
             pd.DataFrame(self.X_compressed, index=df.index,
                          columns=[f'DIM_{i+1}' for i in range(
                              self.X_compressed.shape[1])])],
            axis=1
        )

    def compress_SE(
        self,
        n_components: int,
        neighbours_number_or_fraction: float | int,
        dataframe: pd.DataFrame | None = None,
        random_state: int = 1,
        dict_params: dict | None = None
    ) -> None:
        """
Compress the dataframe using Spectral Embedding.

This method applies Spectral Embedding to reduce the dimensionality
based on graph Laplacian eigenmaps.

Parameters
----------
n_components : int
    Number of dimensions to reduce to.

neighbours_number_or_fraction : float or int
    Number or fraction of neighbors to consider for graph construction.

dataframe : pandas.DataFrame or None, optional
    DataFrame to compress. If None, uses the instance's dataframe.

random_state : int, optional (default=1)
    Random seed for reproducibility.

dict_params : dict or None, optional
    Dictionary of parameters to pass directly to the SpectralEmbedding constructor.

Returns
-------
None
"""

        from sklearn.manifold import SpectralEmbedding

        if dict_params is None:
            if isinstance(neighbours_number_or_fraction, float):
                assert 0 < neighbours_number_or_fraction <= 1, (
                    "'neighbours_number_or_fraction' must be between 0 and 1 "
                    "if float."
                )
                self.n_neighbours = round(neighbours_number_or_fraction *
                                          len(self.dataframe))
            else:
                self.n_neighbours = neighbours_number_or_fraction
            self.algorithm = SpectralEmbedding(
                n_components=n_components,
                n_neighbors=self.n_neighbours,
                n_jobs=-1,
                random_state=random_state
            )
        else:
            self.algorithm = SpectralEmbedding(**dict_params)
        self.params_ = self.algorithm.get_params()
        df = dataframe if dataframe is not None else self.dataframe
        self.X_compressed = self.algorithm.fit_transform(
            df.values[:, self.initial_columns_to_ignore:]
        )
        self.dataframe_compressed = pd.concat(
            [pd.DataFrame(df.iloc[:, :self.initial_columns_to_ignore],
                          columns=df.columns[:self.initial_columns_to_ignore]),
             pd.DataFrame(self.X_compressed, index=df.index,
                          columns=[f'DIM_{i+1}' for i in range(
                              self.X_compressed.shape[1])])],
            axis=1
        )

    def compress_UMAP(
        self,
        n_components: int,
        neighbours_number_or_fraction: float | int,
        dataframe: pd.DataFrame | None = None,
        random_state: int = 1,
        dict_params: dict | None = None
    ) -> None:
        """
Compress the dataframe using Uniform Manifold Approximation and Projection (UMAP).

This method applies UMAP to reduce the dimensionality of the dataframe
while preserving local and global structure.

Parameters
----------
n_components : int
    Number of dimensions to reduce to.

neighbours_number_or_fraction : float or int
    Number or fraction of neighbors to consider for local connectivity.

dataframe : pandas.DataFrame or None, optional
    DataFrame to compress. If None, uses the instance's dataframe.

random_state : int, optional (default=1)
    Random seed for reproducibility.

dict_params : dict or None, optional
    Dictionary of parameters to pass directly to the UMAP constructor.

Returns
-------
None
"""

        import umap

        if dict_params is None:
            if isinstance(neighbours_number_or_fraction, float):
                assert 0 < neighbours_number_or_fraction <= 1, (
                    "'neighbours_number_or_fraction' must be between 0 and 1 "
                    "if float."
                )
                self.n_neighbours = round(neighbours_number_or_fraction *
                                          len(self.dataframe))
            else:
                self.n_neighbours = neighbours_number_or_fraction
            self.algorithm = umap.UMAP(
                n_components=n_components,
                n_neighbors=self.n_neighbours,
                n_jobs=-1,
                random_state=random_state
            )
        else:
            self.algorithm = umap.UMAP(**dict_params)
        self.params_ = self.algorithm.get_params()
        df = dataframe if dataframe is not None else self.dataframe
        self.X_compressed = self.algorithm.fit_transform(
            df.values[:, self.initial_columns_to_ignore:]
        )
        self.dataframe_compressed = pd.concat(
            [pd.DataFrame(df.iloc[:, :self.initial_columns_to_ignore],
                          columns=df.columns[:self.initial_columns_to_ignore]),
             pd.DataFrame(self.X_compressed, index=df.index,
                          columns=[f'DIM_{i+1}' for i in range(
                            self.X_compressed.shape[1])])],
            axis=1
        )

    def compress_MDS(
        self,
        n_components: int,
        dataframe: pd.DataFrame | None = None,
        random_state: int = 1,
        dict_params: dict | None = None
    ) -> None:
        """
Compress the dataframe using Multidimensional Scaling (MDS).

This method applies MDS to reduce the dimensionality of the dataframe
based on pairwise distances.

Parameters
----------
n_components : int
    Number of dimensions to reduce to.

dataframe : pandas.DataFrame or None, optional
    DataFrame to compress. If None, uses the instance's dataframe.

random_state : int, optional (default=1)
    Random seed for reproducibility.

dict_params : dict or None, optional
    Dictionary of parameters to pass directly to the MDS constructor.

Returns
-------
None
"""

        from sklearn.manifold import MDS

        if dict_params is None:
            self.algorithm = MDS(
                n_components=n_components,
                n_jobs=-1,
                random_state=random_state
            )
        else:
            self.algorithm = MDS(**dict_params)
        self.params_ = self.algorithm.get_params()
        df = dataframe if dataframe is not None else self.dataframe
        self.X_compressed = self.algorithm.fit_transform(
            df.values[:, self.initial_columns_to_ignore:]
        )
        self.dataframe_compressed = pd.concat(
            [pd.DataFrame(df.iloc[:, :self.initial_columns_to_ignore],
                          columns=df.columns[:self.initial_columns_to_ignore]),
             pd.DataFrame(self.X_compressed, index=df.index,
                          columns=[f'DIM_{i+1}' for i in range(
                            self.X_compressed.shape[1])])],
            axis=1
        )

    def compress_LLE(
        self,
        n_components: int,
        neighbours_number_or_fraction: float | int,
        dataframe: pd.DataFrame | None = None,
        random_state: int = 1,
        dict_params: dict | None = None
    ) -> None:
        """
Compress the dataframe using Locally Linear Embedding (LLE).

This method applies LLE to reduce the dimensionality of the dataframe
by preserving local neighborhood relationships.

Parameters
----------
n_components : int
    Number of dimensions to reduce to.

neighbours_number_or_fraction : float or int
    Number or fraction of neighbors to consider for local reconstruction.

dataframe : pandas.DataFrame or None, optional
    DataFrame to compress. If None, uses the instance's dataframe.

random_state : int, optional (default=1)
    Random seed for reproducibility.

dict_params : dict or None, optional
    Dictionary of parameters to pass directly to the LLE constructor.

Returns
-------
None
"""

        from sklearn.manifold import LocallyLinearEmbedding

        if dict_params is None:
            if isinstance(neighbours_number_or_fraction, float):
                assert 0 < neighbours_number_or_fraction <= 1, (
                    "'neighbours_number_or_fraction' must be between 0 "
                    "and 1 if float."
                )
                self.n_neighbours = round(neighbours_number_or_fraction *
                                          len(self.dataframe))
            else:
                self.n_neighbours = neighbours_number_or_fraction
            self.algorithm = LocallyLinearEmbedding(
                n_components=n_components,
                n_neighbors=self.n_neighbours,
                n_jobs=-1,
                random_state=random_state
            )
        else:
            self.algorithm = LocallyLinearEmbedding(**dict_params)
        self.params_ = self.algorithm.get_params()
        df = dataframe if dataframe is not None else self.dataframe
        self.X_compressed = self.algorithm.fit_transform(
            df.values[:, self.initial_columns_to_ignore:]
        )
        self.dataframe_compressed = pd.concat(
            [pd.DataFrame(df.iloc[:, :self.initial_columns_to_ignore],
                          columns=df.columns[:self.initial_columns_to_ignore]),
             pd.DataFrame(self.X_compressed, index=df.index,
                          columns=[f'DIM_{i+1}' for i in range(
                            self.X_compressed.shape[1])])],
            axis=1
        )

    def compress_ISOMAP(
        self,
        n_components: int,
        neighbours_number_or_fraction: float | int,
        dataframe: pd.DataFrame | None = None,
        dict_params: dict | None = None
    ) -> None:
        """
Compress the dataframe using Isomap.

This method applies Isomap to reduce the dimensionality of the dataframe
by preserving geodesic distances between all points.

Parameters
----------
n_components : int
    Number of dimensions to reduce to.

neighbours_number_or_fraction : float or int
    Number or fraction of neighbors to consider for neighborhood 
    graph construction.

dataframe : pandas.DataFrame or None, optional
    DataFrame to compress. If None, uses the instance's dataframe.

dict_params : dict or None, optional
    Dictionary of parameters to pass directly to the Isomap constructor.

Returns
-------
None
"""

        from sklearn.manifold import Isomap

        if dict_params is None:
            if isinstance(neighbours_number_or_fraction, float):
                assert 0 < neighbours_number_or_fraction <= 1, (
                    "'neighbours_number_or_fraction' must be between 0 "
                    " and 1 if float."
                )
                self.n_neighbours = round(neighbours_number_or_fraction *
                                          len(self.dataframe))
            else:
                self.n_neighbours = neighbours_number_or_fraction
            self.algorithm = Isomap(
                n_components=n_components,
                n_neighbors=self.n_neighbours,
                n_jobs=-1
            )
        else:
            self.algorithm = Isomap(**dict_params)
        self.params_ = self.algorithm.get_params()
        df = dataframe if dataframe is not None else self.dataframe
        self.X_compressed = self.algorithm.fit_transform(
            df.values[:, self.initial_columns_to_ignore:]
        )
        self.dataframe_compressed = pd.concat(
            [pd.DataFrame(df.iloc[:, :self.initial_columns_to_ignore],
                          columns=df.columns[:self.initial_columns_to_ignore]),
             pd.DataFrame(self.X_compressed, index=df.index,
                          columns=[f'DIM_{i+1}' for i in range(
                            self.X_compressed.shape[1])])],
            axis=1
        )
