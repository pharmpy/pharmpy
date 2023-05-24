from .config import NONMEMConfiguration, conf
from .results import parse_modelfit_results

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
     - pathlib.Path
     - Full path to the default NONMEM installation director
   * - ``write_etas_in_abbr``
     - ``False``
     - bool
     - Whether to write etas as $ABBR records
"""

__all__ = (
    'parse_modelfit_results',
    'conf',
    'NONMEMConfiguration',
)
