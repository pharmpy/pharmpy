from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pharmpy.model import Model
from pharmpy.workflows import ModelfitResults, Results


@dataclass(frozen=True)
class PDSearchResults(Results):
    final_model: Optional[Model] = None
    final_model_results: Optional[ModelfitResults] = None


def calculate_results():
    res = PDSearchResults()
    return res
