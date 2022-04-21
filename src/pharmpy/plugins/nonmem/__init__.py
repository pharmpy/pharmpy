import pathlib

import pharmpy.config as config
from pharmpy.utils import normalize_user_given_path

from .model import Model, convert_model, detect_model

r"""
.. list-table:: Options for the nonmem plugin
   :widths: 25 25 50 150
   :header-rows: 1

   * - Option name
     - Default value
     - Type
     - Description
   * - ``parameter_names``
     - ``['basic']``
     - list
     - Naming scheme of NONMEM parameters. Possible settings are "abbr" ($ABBR), "comment", and
       "basic". The order denotes priority order
   * - ``default_nonmem_path``
     - Path()
     - pathlib.path
     - Full path to the default NONMEM installation director
   * - ``write_etas_in_abbr``
     - ``False``
     - bool
     - Whether to write etas as $ABBR records
"""


class NONMEMConfiguration(config.Configuration):
    module = 'pharmpy.plugins.nonmem'  # TODO: change default
    parameter_names = config.ConfigItem(
        ['basic'],
        'Naming scheme of NONMEM parameters. Possible settings are "abbr" ($ABBR), "comment", and '
        '"basic". The order denotes priority order',
        list,
    )
    default_nonmem_path = config.ConfigItem(
        pathlib.Path(''),
        'Full path to the default NONMEM installation directory',
        cls=normalize_user_given_path,
    )
    write_etas_in_abbr = config.ConfigItem(False, 'Whether to write etas as $ABBR records', bool)
    licfile = config.ConfigItem(
        None, 'Path to the NONMEM license file', cls=normalize_user_given_path
    )


conf = NONMEMConfiguration()


__all__ = ['detect_model', 'convert_model', 'Model']
