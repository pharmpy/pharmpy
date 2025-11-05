from __future__ import annotations

from dataclasses import dataclass

from pharmpy.tools.common import ToolResults


@dataclass(frozen=True)
class PDSearchResults(ToolResults):
    pass


def calculate_results():
    res = PDSearchResults()
    return res
