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

__version__ = '1.2.0'

import logging

import pharmpy.config as config

logging.getLogger(__name__).addHandler(logging.NullHandler())


class PharmpyConfiguration(config.Configuration):
    module = 'pharmpy'
    missing_data_token = config.ConfigItem(
        '-99', 'Data token to be converted to or from NA when reading or writing data'
    )


conf = PharmpyConfiguration()
