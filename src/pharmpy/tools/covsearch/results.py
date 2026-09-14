from __future__ import annotations

from dataclasses import dataclass

from pharmpy.deps import pandas as pd
from pharmpy.tools.common import ToolResults


@dataclass(frozen=True)
class COVSearchResults(ToolResults):
    steps: pd.DataFrame | None = None
    ofv_summary: pd.DataFrame | None = None
    candidate_summary: pd.DataFrame | None = None
    linear_covariate_screening_summary: pd.DataFrame | None = None
