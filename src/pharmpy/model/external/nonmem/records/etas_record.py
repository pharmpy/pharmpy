# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from pathlib import Path

from .option_record import OptionRecord


class EtasRecord(OptionRecord):
    @property
    def path(self):
        file_option = self.option_pairs['FILE']
        assert file_option is not None
        return Path(file_option)

    def set_path(self, value):
        newrec = self.set_option('FILE', str(value))
        return newrec
