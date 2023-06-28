.. _dataset:

===================
Datasets in Pharmpy
===================

Datasets in Pharmpy are represented using the :py:class:`pd.DataFrame` class and a separate
:py:class:`pharmpy.model.DataInfo` class that provides additional information about the dataset. This could contain
for example a description of how the columns are used in the model or the units used for the data.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Retrieving the dataset from a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset connected to a model can be retrieved from the `dataset` attribute.

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

Since the Pharmpy dataset is a pandas dataframe, it can be manipulated as such. A new or updated dataset can be set to
a model like this:

.. pharmpy-execute::
   :hide-output:

   import numpy as np

   df['DV'] = np.log(df['DV'], where=(df['DV'] != 0))
   model = model.replace(dataset=df)

The :py:mod:`pharmpy.modeling` module has several functions to examine and modify the dataset, see the user guide for
:ref:`dataset modeling<modeling_dataset>`.

.. _datainfo:

~~~~~~~~
DataInfo
~~~~~~~~

Every model has a `DataInfo` object that describes the dataset.

.. note::
    A datainfo file can be created for .csv-files `here <https://pharmpy.github.io/amdui/datainfo>`_.

.. pharmpy-execute::

    di = model.datainfo
    di

The path to the dataset file if one exists.

.. pharmpy-execute::
   :hide-output:

    di.path

Separator character for the dataset file.

.. pharmpy-execute::

    di.separator

ColumnInfo
==========

Each column of the dataset can here be given some additional information.

.. pharmpy-execute::

    model.datainfo['AMT']

type
----

Column ``type`` is the role a data column has in the model. Some basic examples of types are ``id`` for the subject identification column, ``idv`` for the independent
variable (mostly time), ``dv`` for the dependent variable and ``dose`` for the dose amount column. Columns that not have been given any particular type
will get the type value ``unknown``. See :attr:`pharmpy.ColumnInfo.type` for a list of all supported types.

scale
-----

The ``scale`` of a column is the statistical scale of measurement of its data using "Stevens' typology" (see https://en.wikipedia.org/wiki/Level_of_measurement). The scale can be one of ``nominal`` for non-ordered categorical data, ``ordinal`` for ordered categorical data, ``interval`` for numeric data were ratios cannot be taken and ``ratio`` for general numeric data. Note that ``nominal`` and ``ordinal`` data is always discrete, but ``interval`` and ``ratio`` data can be both discrete and continuous.

continuous
----------

If this is ``True`` the data is continuous and if it is ``False`` it is discrete. Note that ratio data can be seen as discrete for example
if it has been rounded to whole numbers and cannot take on any real number.

categories
----------

A ``list`` of all values that the data column could have. Not all values have to be present in the dataset. Instead ``categories`` creates a possibility to annotate all possible values. It is also possible to name the categories by using a ``dict`` from the name to its numerical encoding.

unit
----

The physical unit of the column data. Units can be input as a string, e.g. "kg" or "mg/L."

drop
----

A boolean that is set to `True` if the column is not going to be used by the model or `False` otherwise.

datatype
--------

The datatype of the column data. This describes the low level encoding of the data. See :attr:`pharmpy.ColumnInfo.datatype` for a list of all supported datatypes.

descriptor
----------

The descriptor can provide a high level understanding of the data in a machine readable way. See :attr:`pharmpy.ColumnInfo.descriptor` for a list of all supported descriptors.

datainfo file
=============

If a dataset file has an accompanying file with the same name and the extension ``.datainfo`` this will be read in when handling the dataset in Pharmpy. This file is a representation (a serialization) of a ``DataInfo`` object and its content can be created manually, with an external tool or by Pharmpy. Here is an example of the content:

.. pharmpy-execute::

    di.to_json()

It is a json file with the following top level structure:

.. csv-table::
   :header: "Name", "Type"

      ``columns``, array of columns
      ``path``, string
      ``separator``, string

And the columns structure:

.. csv-table::
    :header: Name, Type

        ``type``, string
        ``scale``, string
        ``continuous``, boolean
        ``categories``, array of numbers or string-number map
        ``unit``, string
        ``drop``, boolean
        ``datatype``, string
        ``descriptor``, string
