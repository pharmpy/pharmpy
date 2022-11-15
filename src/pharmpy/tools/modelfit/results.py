from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pharmpy.deps import pandas as pd
from pharmpy.model import Results


@dataclass(frozen=True)
class AggregatedModelfitResults(Results):
    ofv: Optional[float] = None
    parameter_estimates: Optional[Any] = None


def calculate_results(models):
    names = [model.name for model in models]

    ofvs = [model.modelfit_results.ofv for model in models]
    ofv = pd.Series(ofvs, index=names)
    ofv.name = 'OFV'

    params = pd.DataFrame(
        [model.modelfit_results.parameter_estimates for model in models], index=names
    )

    return AggregatedModelfitResults(ofv=ofv, parameter_estimates=params)
