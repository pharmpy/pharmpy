from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pharmpy.deps import pandas as pd
from pharmpy.tools.common import ToolResults


@dataclass(frozen=True)
class COVSearchResults(ToolResults):
    steps: Optional[pd.DataFrame] = None
    ofv_summary: Optional[pd.DataFrame] = None
    candidate_summary: Optional[pd.DataFrame] = None
