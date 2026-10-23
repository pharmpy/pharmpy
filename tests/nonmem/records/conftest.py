# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest

import pharmpy.model.external.nonmem.nmtran_parser
import pharmpy.model.external.nonmem.records


@pytest.fixture
def records():
    return pharmpy.model.external.nonmem.records


@pytest.fixture
def parser():
    return pharmpy.model.external.nonmem.nmtran_parser.NMTranParser()
