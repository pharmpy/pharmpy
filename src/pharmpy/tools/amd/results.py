from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pharmpy.deps import altair as alt
from pharmpy.deps import pandas as pd
from pharmpy.workflows import Results


@dataclass(frozen=True)
class AMDResults(Results):
    final_model: Optional[str] = None
    summary_tool: Optional[Any] = None
    summary_models: Optional[Any] = None
    summary_individuals_count: Optional[pd.DataFrame] = None
    summary_errors: Optional[pd.DataFrame] = None
    final_model_parameter_estimates: Optional[pd.DataFrame] = None
    final_model_dv_vs_ipred_plot: Optional[alt.Chart] = None
    final_model_dv_vs_pred_plot: Optional[alt.Chart] = None
    final_model_cwres_vs_idv_plot: Optional[alt.Chart] = None
    final_model_abs_cwres_vs_ipred_plot: Optional[alt.Chart] = None
    final_model_eta_distribution_plot: Optional[alt.Chart] = None
    final_model_eta_shrinkage: Optional[pd.Series] = None
    final_model_vpc_plot: Optional[alt.Chart] = None
