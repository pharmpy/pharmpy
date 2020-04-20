"""
Data package
============

Collection of modules for relevant manipulation of stand alone datasets
"""

import pharmpy.config as config
from pharmpy.data.data_frame import ColumnType, DatasetError, DatasetWarning, PharmDataFrame
from pharmpy.data.read import read_csv, read_nonmem_dataset


class DataConfiguration(config.Configuration):
    na_values = config.ConfigItem([-99],
                                  'List of data values to be converted to NA when reading data')
    na_rep = config.ConfigItem('-99',
                               'What to replace NA with in written datasets')


conf = DataConfiguration()

__all__ = ['DatasetError', 'DatasetWarning', 'ColumnType', 'PharmDataFrame', 'read_nonmem_dataset',
           'read_csv']
