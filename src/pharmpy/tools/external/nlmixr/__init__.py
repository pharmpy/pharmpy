# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pathlib

from pharmpy import config
from pharmpy.model.external.nlmixr import Model, convert_model

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


class NLMIXRConfiguration(config.Configuration):
    module = 'pharmpy.plugins.nlmixr'
    rpath = config.ConfigItem(
        pathlib.Path(''),
        'Path to R installation directory',
        cls=pathlib.Path,
    )


conf = NLMIXRConfiguration()

__all__ = ('Model', 'convert_model', 'execute_model', 'parse_modelfit_results', 'verification')
