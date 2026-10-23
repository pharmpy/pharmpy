# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from pathlib import Path

from pharmpy.internals.fs.tmp import TemporaryDirectory


def test_tempdir():
    with TemporaryDirectory() as td:
        path = Path(td)
        assert path.is_dir()
    assert not path.is_dir()
