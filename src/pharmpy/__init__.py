r"""
=======
Pharmpy
=======

Pharmpy is a python package for pharmacometric modeling.

Configuration
=============

.. list-table:: Pharmpy core configuration options
   :widths: 25 25 50 150
   :header-rows: 1

   * - Option name
     - Default value
     - Type
     - Description
   * - ``missing_data_token``
     - ``'-99'``
     - str
     - Data token to be converted to or from NA when reading or writing data


Definitions
===========
"""

__version__ = '1.3.0'

import pharmpy.config as config


class PharmpyConfiguration(config.Configuration):
    module = 'pharmpy'
    missing_data_token = config.ConfigItem(
        '-99', 'Data token to be converted to or from NA when reading or writing data'
    )


conf = PharmpyConfiguration()
