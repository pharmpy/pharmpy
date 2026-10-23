# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest

from pharmpy.model.external import utils


def test_detect_model():
    with pytest.raises(TypeError):
        utils.detect_model(None)


def test_load_plugins():
    plugins = utils._load_external_modules()
    names = [x.__name__ for x in plugins]
    assert set(names) == {
        'pharmpy.model.external.nonmem',
        'pharmpy.model.external.nlmixr',
        'pharmpy.model.external.fcon',
        'pharmpy.model.external.generic',
        'pharmpy.model.external.rxode',
    }
