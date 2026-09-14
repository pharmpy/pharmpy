from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pharmpy.deps import altair as alt
from pharmpy.deps import pandas as pd
from pharmpy.workflows import ModelfitResults, Results


@dataclass(frozen=True)
class AMDResults(Results):
    final_model: str | None = None
    final_results: ModelfitResults | None = None
    summary_tool: Any | None = None
    summary_models: Any | None = None
    summary_errors: pd.DataFrame | None = None
    final_model_parameter_estimates: pd.DataFrame | None = None
    final_model_dv_vs_ipred_plot: alt.Chart | None = None
    final_model_dv_vs_pred_plot: alt.Chart | None = None
    final_model_cwres_vs_idv_plot: alt.Chart | None = None
    final_model_abs_cwres_vs_ipred_plot: alt.Chart | None = None
    final_model_eta_distribution_plot: alt.Chart | None = None
    final_model_eta_shrinkage: pd.Series | None = None
    final_model_vpc_plot: alt.Chart | None = None
