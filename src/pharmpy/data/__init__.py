r"""
Data
====

The PharmPy data package is a standalone package.

The main class is the PharmDataFrame which is a pandas DataFrame with some extras
relevant to pharmacometrics.

Manipulation of stand alone datasets

.. list-table:: Options for the data module
   :widths: 25 25 50 150
   :header-rows: 1

   * - Option name
     - Default value
     - Type
     - Description
   * - ``na_values``
     - ``[-99]``
     - List
     - Data values to be converted to NA when reading in data
   * - ``na_rep``
     - ``'-99'``
     - str
     - Data value to convert NA to when writing data

"""

import pharmpy.config as config
from pharmpy.data.data_frame import ColumnType, DatasetError, DatasetWarning, PharmDataFrame
from pharmpy.data.read import read_csv, read_nonmem_dataset


class DataConfiguration(config.Configuration):
    module = 'pharmpy.data'
    na_values = config.ConfigItem(
        [-99], 'List of data values to be converted to NA when reading data'
    )
    na_rep = config.ConfigItem('-99', 'What to replace NA with in written datasets')


conf = DataConfiguration()

__all__ = [
    'DatasetError',
    'DatasetWarning',
    'ColumnType',
    'PharmDataFrame',
    'read_nonmem_dataset',
    'read_csv',
]
