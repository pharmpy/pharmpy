# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import re


def sort_alphanum(strings):

    def keyfunc(s):
        tokens = [int(tok) if tok.isdigit() else tok for tok in re.findall(r"\d+|\D+", s)]
        return tokens

    return sorted(strings, key=keyfunc)
