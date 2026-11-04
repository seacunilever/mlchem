Architecture
============

mlchem separates chemistry-specific operations from the reusable machine
learning workflow. A typical molecular-property workflow starts with molecular
structures, calculates descriptors, prepares the resulting feature table, then
selects features and evaluates models. These stages are independent and can be
used separately.

Package responsibilities
------------------------

``chem.calculator``
    Calculates molecular descriptors and fingerprints, including RDKit- and
    Mordred-based features.

``chem.manipulation``
    Creates, converts, and modifies molecular structures and provides
    molecular pattern-recognition utilities.

``chem.visualise``
    Draws molecular structures, displays atom-level importance maps, and
    visualises datasets in chemical space.

``ml.preprocessing``
    Transforms feature tables through scaling, feature expansion, dimensionality
    reduction, and sampling utilities.

``ml.feature_selection``
    Provides feature filters and wrapper-based selection methods, including
    sequential forward and combinatorial selection.

``ml.modelling``
    Evaluates estimators, supports ensemble workflows, assesses applicability
    domains, and interprets model predictions.

Shared utilities
----------------

The root-level ``helper.py`` module provides reusable data, logging, and
visualisation utilities. ``metrics.py`` contains scores used by feature
selection and model evaluation. These shared modules support the package
workflows without requiring users to follow a single prescribed pipeline.
