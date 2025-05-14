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
   * - ``dispatcher``
     - ``local_dask``
     - str
     - Name of the default dispatcher to use when running tools
   * - ``broadcaster``
     - ``terminal``
     - str
     - Name of the default broadcaster for messages from tools


Definitions
===========
"""

__version__ = '1.7.1'

import pharmpy.config as config


class PharmpyConfiguration(config.Configuration):
    module = 'pharmpy'
    missing_data_token = config.ConfigItem(
        '-99', 'Data token to be converted to or from NA when reading or writing data'
    )
    dispatcher = config.ConfigItem(
        'local_dask',
        'Name of default dispatcher to use when running tools',
    )
    broadcaster = config.ConfigItem(
        'terminal',
        'Name of default broadcaster',
    )


conf = PharmpyConfiguration()

DEFAULT_SEED = 1234
