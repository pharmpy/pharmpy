from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from pharmpy.deps import pandas as pd
from pharmpy.model import Model, Results


@dataclass(frozen=True)
class CrossvalResults(Results):
    """Crossval results class"""

    runs: Optional[Any] = None
    prediction_ofv_sum: Optional[Any] = None


def calculate_results(estimation_models, prediction_models):
    """Calculate crossval results"""
    est_ofvs = [model.modelfit_results.ofv for model in estimation_models]
    pred_ofvs = [model.modelfit_results.ofv for model in prediction_models]
    runs = pd.DataFrame(
        {'estimation_ofv': est_ofvs, 'prediction_ofv': pred_ofvs},
        index=pd.RangeIndex(start=1, stop=len(est_ofvs) + 1),
    )

    return CrossvalResults(runs=runs, prediction_ofv_sum=sum(pred_ofvs))


def psn_crossval_results(path):
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    est_paths = [str(p) for p in (path / 'm1').glob('est_model*.mod')]
    est_paths.sort(key=natural_keys)
    pred_paths = [str(p) for p in (path / 'm1').glob('pred_model*.mod')]
    pred_paths.sort(key=natural_keys)
    ests = [Model.create_model(path) for path in est_paths]
    preds = [Model.create_model(path) for path in pred_paths]

    res = calculate_results(ests, preds)
    return res
