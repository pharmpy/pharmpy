# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later


class RawRecord:
    """A record that just keeps contents unparsed.
    Used for unknown records and for anything coming before the first record
    """

    def __init__(self, content, name='', raw_name=''):
        self.name = name
        self.raw_name = raw_name
        self.content = content

    def __str__(self):
        return self.raw_name + self.content
