r"""
.. list-table:: Options for the modelfit tool
   :widths: 25 25 50 150
   :header-rows: 1

   * - Option name
     - Default value
     - Type
     - Description
   * - ``default_tool``
     - 'nonmem'
     - str
     - Name of default estimation tool either 'nonmem' or 'nlmixr'
"""
import pharmpy.config as config

from .tool import create_fit_workflow, create_workflow


class ModelfitConfiguration(config.Configuration):
    module = 'pharmpy.tools.modelfit'  # TODO: change default
    default_tool = config.ConfigItem('nonmem', 'Name of default estimation tool', cls=str)


conf = ModelfitConfiguration()


__all__ = ('create_workflow', 'create_fit_workflow')
