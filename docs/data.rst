===================
Datasets in Pharmpy
===================

.. warning::

    This section is being reworked.

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

   from pharmpy.modeling import read_model

   model = read_model(path / "pheno_real.mod")
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

An array of all subject IDs can be retrieved.

.. jupyter-execute::

    from pharmpy.modeling import get_ids
    model = read_model(path / "pheno_real.mod")
    get_ids(model)

The number of subjects in the dataset could optionally be retrieved directly.

.. jupyter-execute::

    from pharmpy.modeling import get_number_of_individuals
    get_number_of_individuals(model)


~~~~~~~~~~~~
Observations
~~~~~~~~~~~~

The observations of the dataset indexed on subject ID and the independent variable can be extracted.

.. jupyter-execute::

    from pharmpy.modeling import get_observations
    get_observations(model)

The total number of observations can optionally be retrieved directly.

.. jupyter-execute::

    from pharmpy.modeling import get_number_of_observations
    get_number_of_observations(model)

~~~~~~
Dosing
~~~~~~

Extract dosing information
==========================

The doses of the dataset indexed on subject ID and the independent variable can be extracted.

.. jupyter-execute::

    from pharmpy.modeling import get_doses
    doses = get_doses(model)
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

It is possible to create a DOSEID that groups each dose period starting from 1.

.. jupyter-execute::

    from pharmpy.modeling import get_doseid
    ser = get_doseid(model)
    ser

Time after dose
===============

Add a column for time after dose (TAD)

.. jupyter-execute::

    from pharmpy.modeling import add_time_after_dose
    add_time_after_dose(model)
    model.dataset['TAD']

Concentration parameters
========================

Extract pharmacokinetic concentration parameters from the dataset

.. jupyter-execute::

    from pharmpy.modeling import get_concentration_parameters_from_data
    get_concentration_parameters_from_data
