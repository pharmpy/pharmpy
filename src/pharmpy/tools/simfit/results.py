from dataclasses import dataclass
from typing import Any, Optional

from pharmpy.model import Model, Results
from pharmpy.tools.external.nonmem.results import simfit_results


@dataclass(frozen=True)
class SimfitResults(Results):
    """Simfit results class"""

    modelfit_results: Optional[Any] = None


def calculate_results(modelfit_results):
    """Calculate simfit results"""
    return SimfitResults(modelfit_results=modelfit_results)


def psn_simfit_results(paths):
    modelfit_results = []
    for path in paths:
        model = Model.parse_model(path)
        modelfit_results.extend(simfit_results(model, path))
    res = calculate_results(modelfit_results)
    return res
