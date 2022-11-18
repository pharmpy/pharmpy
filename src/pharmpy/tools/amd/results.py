from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pharmpy.deps import pandas as pd
from pharmpy.model import Results


@dataclass(frozen=True)
class AMDResults(Results):
    final_model: Optional[str] = None
    summary_tool: Optional[Any] = None
    summary_models: Optional[Any] = None
    summary_individuals_count: Optional[pd.DataFrame] = None
    summary_errors: Optional[pd.DataFrame] = None
