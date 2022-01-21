import pathlib

import pharmpy.config as config

from .model import convert_model

r"""
.. list-table:: Options for the nlmixr plugin
   :widths: 25 25 50 150
   :header-rows: 1

   * - Option name
     - Default value
     - Type
     - Description
   * - ``rpath``
     - ````
     - str
     - Path to R installation directory
"""


class NLMIXRConfiguration(config.Configuration):
    module = 'pharmpy.plugins.nlmixr'
    rpath = config.ConfigItem(
        pathlib.Path(''),
        'Path to R installation directory',
        cls=pathlib.Path,
    )


conf = NLMIXRConfiguration()

__all__ = ['convert_model']
