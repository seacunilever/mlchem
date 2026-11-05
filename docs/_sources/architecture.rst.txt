Architecture
============

mlchem separates chemistry-specific operations from the reusable machine
learning workflow. A typical molecular-property workflow starts with molecular
structures, calculates descriptors, prepares the resulting feature table, then
selects features and evaluates models. These stages are independent and can be
used separately.

For a runnable end-to-end example, start with the :doc:`quickstart <welcome>`.
The :doc:`API reference <modules>` links to the full documentation for each
package.

Package responsibilities
------------------------

:doc:`chem.calculator API <mlchem.chem.calculator>`
    Calculates molecular descriptors and fingerprints, including RDKit- and
    Mordred-based features. Common entry points include
    :func:`mlchem.chem.calculator.descriptors.get_rdkitDesc` and
    :func:`mlchem.chem.calculator.descriptors.get_fingerprint_df`.

:doc:`chem.manipulation API <mlchem.chem>`
    Creates, converts, and modifies molecular structures and provides
    molecular pattern-recognition utilities. Start with
    :func:`mlchem.chem.manipulation.create_molecule`.

:doc:`chem.visualise API <mlchem.chem.visualise>`
    Draws molecular structures, displays atom-level importance maps, and
    visualises datasets in chemical space.

:doc:`ml.preprocessing API <mlchem.ml.preprocessing>`
    Transforms feature tables through scaling, feature expansion, dimensionality
    reduction, and sampling utilities. See
    :func:`mlchem.ml.preprocessing.scaling.scale_df_standard`,
    :func:`mlchem.ml.preprocessing.feature_transformation.polynomial_expansion`,
    and :func:`mlchem.ml.preprocessing.undersampling.undersample`.

:doc:`ml.feature_selection API <mlchem.ml.feature_selection>`
    Provides feature filters and wrapper-based selection methods, including
    :class:`mlchem.ml.feature_selection.wrappers.SequentialForwardSelection`
    and :class:`mlchem.ml.feature_selection.wrappers.CombinatorialSelection`.

:doc:`ml.modelling API <mlchem.ml.modelling>`
    Evaluates estimators, supports ensemble workflows, assesses applicability
    domains, and interprets model predictions. See
    :func:`mlchem.ml.modelling.model_evaluation.crossval`,
    :func:`mlchem.ml.modelling.model_evaluation.y_scrambling`, and
    :meth:`mlchem.ml.modelling.model_evaluation.MajorityVote.evaluate_cv`.

Shared utilities
----------------

The root-level ``helper.py`` module provides reusable data, logging, and
visualisation utilities. :mod:`mlchem.metrics` contains scores used by feature
selection and model evaluation. These shared modules support the package
workflows without requiring users to follow a single prescribed pipeline.
