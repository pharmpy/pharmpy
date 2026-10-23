# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from pathlib import Path


def test_title_limits(parser):
    rec = parser.parse("$ETAS FILE=run1.phi").records[0]
    assert rec.path == Path("run1.phi")
