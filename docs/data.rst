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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Update the dataset of a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new or updated dataset can be set to a model

.. jupyter-execute::

   import numpy as np

   df['DV'] = np.log(df['DV'], where=(df['DV'] != 0))
   model.dataset = df 

~~~~~~~~
Subjects
~~~~~~~~

An array of all subject IDs can be retrieved using the pharmpy dataframe API extension.

.. jupyter-execute::

   model = Model(path / "pheno_real.mod")
   df = model.dataset
   ids = df.pharmpy.ids
   ids

The number of subjects in the dataset is the length of this array.

.. jupyter-execute::

    len(ids)


~~~~~~~~~~~~
Observations
~~~~~~~~~~~~

The observations of the dataset indexed on subject ID and the independent variable can be extracted.

.. jupyter-execute::

   obs = df.pharmpy.observations
   obs

The total number of observations is the length of this series.

.. jupyter-execute::

    len(obs)

~~~~~~
Dosing
~~~~~~

Extract dosing information
==========================

The doses of the dataset indexed on subject ID and the independent variable can be extracted.

.. jupyter-execute::

   doses = df.pharmpy.doses
   doses

All unique doses can be listed

.. jupyter-execute::

    doses.unique()

as well as the largest and the smallest dose

.. jupyter-execute::

    doses.min()

.. jupyter-execute::

    doses.max()

Dose grouping
=============

It is possible to add a DOSEID column with a numbering of each dose period starting from 1.

.. jupyter-execute::

    df.pharmpy.add_doseid()
    df

Time after dose
===============

Add a column for time after dose (TAD)

.. jupyter-execute::

    df.pharmpy.add_time_after_dose()
    df

Concentration parameters
========================

Extract pharmacokinetic concentration parameters from the dataset

.. jupyter-execute::

    df.pharmpy.concentration_parameters()
