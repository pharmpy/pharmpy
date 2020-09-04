import pharmpy.config as config

from .model import Model

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

"""


class NONMEMConfiguration(config.Configuration):
    module = 'pharmpy.plugins.nonmem'
    parameter_names = config.ConfigItem(
            'basic',
            'Naming scheme of NONMEM parameters. One of "basic" and "comment"')


conf = NONMEMConfiguration()


__all__ = ['Model']
