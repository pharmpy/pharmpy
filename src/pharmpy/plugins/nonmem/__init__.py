import pathlib

import pharmpy.config as config

from .model import Model, detect_model

r"""
.. list-table:: Options for the nonmem plugin
   :widths: 25 25 50 150
   :header-rows: 1

   * - Option name
     - Default value
     - Type
     - Description
   * - ``parameter_names``
     - ``'basic'``
     - Str
     - Naming scheme of NONMEM parameters. One of 'basic' and 'comment'
   * - ``default_nonmem_path``
     - Path()
     - pathlib.path
     - Full path to the default NONMEM installation directory

"""


class NONMEMConfiguration(config.Configuration):
    module = 'pharmpy.plugins.nonmem'
    parameter_names = config.ConfigItem(
        'basic', 'Naming scheme of NONMEM parameters. One of "basic" and "comment"'
    )
    default_nonmem_path = config.ConfigItem(
        pathlib.Path(''), 'Full path to the default NONMEM installation directory', cls=pathlib.Path
    )


conf = NONMEMConfiguration()


__all__ = ['detect_model', 'Model']
