# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass

from pharmpy.tools.common import ToolResults


@dataclass(frozen=True)
class PDSearchResults(ToolResults):
    pass


def calculate_results():
    res = PDSearchResults()
    return res
