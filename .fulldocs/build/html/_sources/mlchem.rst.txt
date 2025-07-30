===========================
**chem and ml subpackages**
===========================

.. toctree::
   :maxdepth: 4

   mlchem.chem
   mlchem.ml


mlchem.helper module
--------------------

.. automodule:: mlchem.helper
   :members:
   :show-inheritance:
   :undoc-members:

mlchem.importables module
-------------------------

Useful collections to perform various tasks:
   1. metal_list: List with all metals in SMILES notation.
   2. chemical_dictionary: dictionary in the form {'fragment': int} to create a bag of fragments useful in :ref:`chem.manipulation`.
   3. colour_dictionary: dictionary of RGB tuples for a number of predefined colours.
   4. chemotype_dictionary: collection of many pattern recognition functions that will be used by :py:func:`chem.calculator.descriptors.get_chemotypes`.
   5. bokeh_dictionary: collection of predefined bokeh plotting parameters.
   6. bokeh_tooltips: predefined HMTL bokeh tooltips to interactively visualise chemical space.
   7. interpretable_descriptors_rdkit: list of rdkit descriptors with a simple meaning.
   8. interpretable_descriptors_mordred: list of mordred descriptors with a simple meaning.
   9. similarity_metric_dictionary: dictionary in the form {metric_name: func} collecting various similarity metric functions (functions can be called from :ref:`mlchem.metrics` as well).


.. automodule:: mlchem.importables
   :members:
   :show-inheritance:
   :undoc-members:


.. _mlchem.metrics:

mlchem.metrics module
---------------------


.. automodule:: mlchem.metrics
   :members:
   :show-inheritance:
   :undoc-members:


