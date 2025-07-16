# mlchem

**mlchem** is a Python cheminformatics library designed for the scientific community. It provides a comprehensive set of tools for data handling, molecule manipulation, drawing, machine learning, and plotting.
The library has been tested for python 3.11, 3.12 and 3.13.

## Documentation

Available at https://ubiquitous-adventure-ggl1ng1.pages.github.io/

## Features

- **Data Handling**: Efficiently manage and process chemical data, including loading, cleaning, and transforming datasets.
- **Molecule Manipulation**: Tools for manipulating molecular structures, such as adding or removing atoms, modifying bonds, and generating molecular conformations.
- **Pattern Recognition**: An extensive list of functions to search for specific structural patterns.
- **Molecule Drawing**: Visualise molecules with customisable drawing options, creating high-quality images for presentations and publications.
- **Machine Learning**: Implement machine learning models for cheminformatics, including training, evaluating, and deploying models to predict chemical properties and activities.
- **Feature Analysis and Interpretation**: Interpret model features and provide insightful plots.

## Architecture

### mlchem
#### chem/
##### calculator/
###### descriptors.py
###### tools.py
##### visualise/
###### drawing.py
###### simmaps.py
###### space.py
##### manipulation.py
#### ml/
##### feature_selection/
###### filters.py
###### wrappers.py
##### modelling/
###### model_evaluation.py
###### model_interpretation.py
##### preprocessing/
###### dimensional_reduction.py
###### feature_transformation.py
###### scaling.py 
###### undersampling.py

## Modules

### chem.visualise/

- **space.py**: Computes and visualises datasets in a lower-dimensional space.
- **simmaps.py**: Generates "rdkit-like" similarity maps based on atomic importance weights.
- **drawing.py**: Handles the drawing of molecular structures with many customisable options.

### chem.calculator/

- **tools.py**: Provides numerous tools for chemical calculations.
- **descriptors.py**: Calculates various descriptors for molecules, including RDKit and Mordred descriptors, atomic descriptors, chemotypes, fingerprints, and some quantum chemistry properties.

### chem.manipulation.py

The `mlchem.chem.manipulation` module offers a variety of tools for creating, converting, manipulating molecular structures, generate new molecules and recognise molecular patterns.

### ml.feature_selection/

- **filters.py**: Provides functionalities for filtering features.
- **wrappers.py**: Offers simplified interfaces for feature selection.

### ml.modelling/

- **model_interpretation.py**: Provides tools for interpreting machine learning models.
- **model_evaluation.py**: Contains tools for evaluating machine learning models.

### ml.preprocessing/

- **dimensional_reduction.py**: Provides functionalities for compressing dataframes using various dimensionality reduction techniques.
- **feature_transformation.py**: Expands features to polynomial features.
- **scaling.py**: Provides functionalities for scaling dataframes using different scaling techniques.
- **undersampling.py**: Contains techniques for handling imbalanced datasets.

## Installation

To install **mlchem**, open your command prompt and use the following command:

```bash
pip install https://github.com/seacunilever/mlchem.git
```

Alternatively, go to your favourite Python version directory/Lib/site-packages and open your command prompt. Once done, use the following command:

```bash
git clone https://github.com/seacunilever/mlchem.git
```

Subsequently, use the following command to install requirements:

```bash
pip install -r mlchem/requirements.txt
```

## Usage

Here's a basic example of how to use **mlchem** (this calculates rdkit descriptors for two molecules):

```python
from mlchem.chem.manipulation import create_molecule
from mlchem.chem.calculator import descriptors
mol1 = create_molecule('c1ccccc1CCCO')
mol2 = create_molecule('CCCCCN')
desc_df = descriptors.get_rdkitDesc([mol1, mol2],include_3D=True)
```

More examples in the **examples** folder.

## Contributing

We welcome contributions to **mlchem**. Users are free to propose new functionalities, flag new bugs, fix old bugs and issue pull requests.

## Third-Party Dependencies

This project uses the https://github.com/aspuru-guzik-group/selfies Python package for molecular string representations.  
SELFIES is licensed under the [Apache License 2.0](https://www.apache.org/). In accordance with its license, the relevant license is included in this repository.

This project uses and adapts code from the [RDKit](https://www.rdkit.org) cheminformatics toolkit, which is licensed under the [BSD 3-Clause License](https://interoperable-europe.ec.europa.eu/licence/bsd-3-clause-clear-license).

## License

This project is licensed under the BSD-3 License.

Note: This project includes components licensed under the Apache License 2.0 (e.g., the SELFIES package), as well as source code taken and adapted from RDKit library.


## Acknowledgements

Special thanks to the Safety, Environmental & Regulatory Science (SERS) Department at Unilever.
