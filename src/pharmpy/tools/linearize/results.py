from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pharmpy.deps import pandas as pd
from pharmpy.model import Model, Results
from pharmpy.modeling import plot_iofv_vs_iofv
from pharmpy.tools import read_modelfit_results


@dataclass(frozen=True)
class LinearizeResults(Results):
    ofv: Optional[float] = None
    iofv: Optional[Any] = None
    iofv_plot: Optional[Any] = None


def calculate_results(base_model, base_model_results, linear_model, linear_model_results):
    iofv = pd.DataFrame(
        {
            'base': base_model_results.individual_ofv,
            'linear': linear_model_results.individual_ofv,
            'delta': linear_model_results.individual_ofv - base_model_results.individual_ofv,
        }
    )
    ofv = pd.DataFrame(
        {
            'ofv': [
                base_model_results.ofv,
                linear_model_results.ofv_iterations.iloc[0],
                linear_model_results.ofv,
            ]
        },
        index=['base', 'lin_evaluated', 'lin_estimated'],
    )
    iofv1 = base_model_results.individual_ofv
    iofv2 = linear_model_results.individual_ofv
    iofv_plot = plot_iofv_vs_iofv(iofv1, iofv2, base_model.name, linear_model.name)
    res = LinearizeResults(ofv=ofv, iofv=iofv, iofv_plot=iofv_plot)
    return res


def psn_linearize_results(path):
    """Create linearize results from a PsN linearize run

    :param path: Path to PsN linearize run directory
    :return: A :class:`QAResults` object
    """
    path = Path(path)

    base_model_path = path / 'scm_dir1' / 'derivatives.mod'
    base_model = Model.parse_model(base_model_path)
    base_model_results = read_modelfit_results(base_model_path)
    lin_path = list(path.glob('*_linbase.mod'))[0]
    lin_model = Model.parse_model(lin_path)
    lin_model_results = read_modelfit_results(lin_path)

    res = calculate_results(base_model, base_model_results, lin_model, lin_model_results)
    return res
