import os
from unittest import mock

import pytest

import pharmpy.config as config
import pharmpy.tools.external.nonmem.config
from pharmpy.tools.external.nonmem.run import nmfe_path


@mock.patch.dict(os.environ, {"PATH": ""})
@pytest.mark.filterwarnings('ignore:User config file is disabled')
def test_nmfe_path():
    with config.ConfigurationContext(pharmpy.tools.external.nonmem.conf, default_nonmem_path='x'):
        with pytest.raises(FileNotFoundError, match='Cannot find nmfe script'):
            nmfe_path()

    with config.ConfigurationContext(pharmpy.tools.external.nonmem.conf, default_nonmem_path=''):
        with pytest.raises(FileNotFoundError, match='Cannot find pharmpy.conf'):
            nmfe_path()
