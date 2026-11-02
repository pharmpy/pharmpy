.. SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _dataset:

=======
Dataset
=======

Datasets in Pharmpy are represented using the :py:class:`pd.DataFrame` class (which becomes a regular
:code:`data.frame` in pharmr) and a separate
:py:class:`pharmpy.model.DataInfo` class that provides additional information about the dataset. This could contain
for example a description of how the columns are used in the model or the units used for the data.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Retrieving the dataset from a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset connected to a model can be retrieved via the :code:`dataset` attribute.

.. pharmpy-execute::
   :hide-output:
   :hide-code:

   from pathlib import Path
   path = Path('tests/testdata/nonmem/')

.. pharmpy-execute::

   from pharmpy.modeling import read_model

   model = read_model(path / "pheno_real.mod")
   df = model.dataset
   df

This is the dataset after applying any model specific filtering and handling of special values.

The raw dataset can also be accessed

.. pharmpy-execute::

   raw = model.read_raw_dataset()
   raw

Note that all values here are strings

.. pharmpy-execute::

   raw.dtypes

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Update the dataset of a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the Pharmpy dataset is a dataframe, it can be manipulated as such. A new or updated dataset can be set to
a model like this:

.. pharmpy-execute::
   :hide-output:

   import numpy as np
   from pharmpy.modeling import set_dataset

   df['DV'] = np.log(df['DV'], where=(df['DV'] != 0).values)
   model = set_dataset(model, df)

The :py:mod:`pharmpy.modeling` module has several functions to examine and modify the dataset, see the user guide for
:ref:`dataset modeling<modeling_dataset>`.

.. _datainfo:

~~~~~~~~
DataInfo
~~~~~~~~

Every model has a :py:class:`DataInfo <pharmpy.model.DataInfo>` object that describes the dataset.

.. note::
    A datainfo file can be created for .csv-files `here <https://pharmpy.github.io/amdui/datainfo>`_.

.. pharmpy-execute::

    di = model.datainfo
    di

The path to the dataset file if one exists.

.. pharmpy-execute::
   :hide-output:

    di.path

The path to the dataset is ``None`` if the original read in dataset is no longer possible to use, e.g. if a
column has been added or changed.

Separator character for the dataset file.

.. pharmpy-execute::

    di.separator

The provenance describes which operations have been made to a dataset. In other words: which steps are needed to go from
the read in dataset to the current dataset.

.. pharmpy-execute::

    di.provenance

ColumnInfo
==========

Each column of the dataset is described by a :py:class:`ColumnInfo <pharmpy.model.ColumnInfo>` object which contains information on the datatype, if the column is droped or not and which variable or variables are contained in this column.

.. pharmpy-execute::

    model.datainfo['AMT']

drop
----

A boolean that is set to `True` if the column is not going to be used by the model or `False` otherwise.

datatype
--------

The datatype of the column data. This describes the low level encoding of the data. See :attr:`pharmpy.ColumnInfo.datatype` for a list of all supported datatypes.

variable_id
-----------

If the column holds data for multiple variables (e.g. `DV` for datasets with multiple types of observations) :code:`variable_id` will be set to the name of the column acting as the selector (e.g. `DVID`).

variables
---------

Description of the variable if only one or a mapping from selector values to variable if more than one.


DataVariable
============

Each column describes one or more :py:class:`data variables <pharmpy.model.DataVariable>`.

.. pharmpy-execute::

    model.datainfo['AMT'].variables

type
----

Column ``type`` is the role a data column has in the model. Some basic examples of types are ``id`` for the subject identification column, ``idv`` for the independent
variable (mostly time), ``dv`` for the dependent variable and ``dose`` for the dose amount column. Columns that not have been given any particular type
will get the type value ``unknown``. See :attr:`pharmpy.ColumnInfo.type` for a list of all supported types.

scale
-----

The ``scale`` of a column is the statistical scale of measurement of its data using "Stevens' typology" (see https://en.wikipedia.org/wiki/Level_of_measurement). The scale can be one of ``nominal`` for non-ordered categorical data, ``ordinal`` for ordered categorical data, ``interval`` for numeric data were ratios cannot be taken and ``ratio`` for general numeric data. Note that ``nominal`` and ``ordinal`` data is always discrete, but ``interval`` and ``ratio`` data can be both discrete and continuous.

count
-----

If this is ``True`` the data is count.

properties
----------

Various metadata for the data variable.

categories
----------

A ``list`` of all values that the data column could have. Not all values have to be present in the dataset. Instead ``categories`` creates a possibility to annotate all possible values. It is also possible to name the categories by using a ``dict`` from the name to its numerical encoding.

unit
----

The physical unit of the column data. Units can be input as a string, e.g. "kg" or "mg/L."

descriptor
----------

The descriptor can provide a high level understanding of the data in a machine readable way. See :attr:`pharmpy.ColumnInfo.descriptor` for a list of all supported descriptors.

molar_mass
----------

The molar mass in of the substance in g/mol.

datainfo file
=============

If a dataset file has an accompanying file with the same name and the extension ``.datainfo`` this will be read in when handling the dataset in Pharmpy. This file is a representation (i.e. a serialization) of a ``DataInfo`` object and its content can be created manually, with an external tool or by Pharmpy. Here is an example of the content:

.. pharmpy-execute::

    di.to_json()


.. _nonmem_dataset:

Working with DataInfo
=====================

There are functions to create and manipulate ``DataInfo`` objects either inside models or stand alone.

.. pharmpy-execute::

    from pharmpy.modeling import annotate_unit, set_property

    model = annotate_unit(model, "WGT", "kg")
    di = set_property(model.datainfo, "APGR", "categories", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

For a list of available functions see the :ref:`reference documentation <dataset_api>`.

~~~~~~~~~~~~~~
NONMEM dataset
~~~~~~~~~~~~~~

Since Pharmpy 2.2.0, Pharmpy supports using the original dataset if possible when estimating etc.

Parsing
=======

Pharmpy does the following filter when parsing a NONMEM dataset:

* Filter individuals without observations
* Filter all expression specified by IGNORE/ACCEPT statements in $DATA

These filters will be added in ``DataInfo.provenance``.

.. pharmpy-execute::
   :hide-output:
   :hide-code:

    from pharmpy.modeling import read_model, filter_dataset
    model = read_model(path / 'pheno_pd.mod')
    model = filter_dataset(model, 'DVID==2')

As an example, if we have the following model which filters out records with ``DVID == 2``:

.. pharmpy-execute::
   :hide-code:

   from pharmpy.modeling import print_model_code
   print_model_code(model)

We can examine the provenance of that dataset:

.. pharmpy-execute::

   model.datainfo.provenance

Here we can see the reading operation and then a filtering operation.

Updating
========

If a change is made to the dataset which allows us to use the original dataset (e.g. filtering rows or removing columns),
DROPs and IGNOREs will be added to the NONMEM code as needed. As an example, if we filter out some more records from the
previous model:

.. pharmpy-execute::

   from pharmpy.modeling import filter_dataset
   model = filter_dataset(model, 'APGR >= 8')

The provenance will add an ignore operation:

.. pharmpy-execute::

   model.datainfo.provenance

And $DATA and $INPUT will be updated accordingly:

.. pharmpy-execute::

   print_model_code(model)

In some cases, the original dataset can no longer be used. For example if a new column is added:

.. pharmpy-execute::

   from pharmpy.modeling import add_time_after_dose
   model = add_time_after_dose(model)

The provenance will still have a new add column operation:

.. pharmpy-execute::

   model.datainfo.provenance

But the NONMEM code will now have a dummy path as the path and the IGNORE statements will be removed (since the
Pharmpy dataset has removed those rows):

.. pharmpy-execute::

   print_model_code(model)

If writing the model (e.g. when writing the files or estimating the model), ``DUMMYPATH`` will be replaced with a real
path.
