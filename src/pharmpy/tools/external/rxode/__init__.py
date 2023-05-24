import pathlib

import pharmpy.config as config
from pharmpy.model.external.rxode import Model, convert_model

from .run import execute_model, parse_modelfit_results, verification

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


class RxODEConfiguration(config.Configuration):
    module = 'pharmpy.plugins.rxode'
    rpath = config.ConfigItem(
        pathlib.Path(''),
        'Path to R installation directory',
        cls=pathlib.Path,
    )


conf = RxODEConfiguration()

__all__ = ('Model', 'convert_model', 'parse_modelfit_results', 'verification', 'execute_model')
