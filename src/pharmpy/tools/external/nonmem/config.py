from pathlib import Path

from pharmpy.config import ConfigItem, Configuration
from pharmpy.internals.fs.path import normalize_user_given_path


class NONMEMConfiguration(Configuration):
    module = 'pharmpy.plugins.nonmem'  # TODO: change default
    default_nonmem_path = ConfigItem(
        Path(''),
        'Full path to the default NONMEM installation directory',
        cls=normalize_user_given_path,
    )
    write_etas_in_abbr = ConfigItem(False, 'Whether to write etas as $ABBR records', bool)
    licfile = ConfigItem(None, 'Path to the NONMEM license file', cls=normalize_user_given_path)


conf = NONMEMConfiguration()
