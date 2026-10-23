# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from unittest import mock

import pytest

import pharmpy.tools.external.nonmem.config
from pharmpy import config
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
