===================
Datasets in Pharmpy
===================

Datasets in Pharmpy are represented using the :py:class:`pharmpy.data.PharmDataFrame` class. It is a subclass of the pandas DataFrame and have some additions specific to Pharmacometrics. 

.. math::

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Retrieving the dataset from a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::
   :hide-output:
   :hide-code:

   from pathlib import Path
   path = Path('tests/testdata/nonmem/')

.. jupyter-execute::

   from pharmpy import Model

   model = Model(path / "pheno_real.mod")
   df = model.dataset
   df

This is the dataset after applying any model specific filtering and handling of special values.

The raw dataset can also be accessed

.. jupyter-execute::

   raw = model.read_raw_dataset()
   raw

Note that all values here are strings

.. jupyter-execute::

   raw.dtypes

A new or updated dataset can be set to a model

.. jupyter-execute::

   import numpy as np

   df['DV'] = np.log(df['DV'], where=(df['DV'] != 0))
   model.dataset = df 
