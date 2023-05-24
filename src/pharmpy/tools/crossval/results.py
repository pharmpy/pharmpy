from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from pharmpy.deps import pandas as pd
from pharmpy.model import Results
from pharmpy.tools import read_modelfit_results


@dataclass(frozen=True)
class CrossvalResults(Results):
    """Crossval results class"""

    runs: Optional[Any] = None
    prediction_ofv_sum: Optional[Any] = None


def calculate_results(estimation_results, prediction_results):
    """Calculate crossval results"""
    est_ofvs = [res.ofv for res in estimation_results]
    pred_ofvs = [res.ofv for res in prediction_results]
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
    ests = [read_modelfit_results(path) for path in est_paths]
    preds = [read_modelfit_results(path) for path in pred_paths]

    res = calculate_results(ests, preds)
    return res
