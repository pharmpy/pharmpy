from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model import Model
from pharmpy.modeling import plot_iofv_vs_iofv
from pharmpy.workflows import ModelfitResults, Results


@dataclass(frozen=True)
class LinearizeResults(Results):
    ofv: Optional[float] = None
    iofv: Optional[Any] = None
    iofv_plot: Optional[Any] = None
    final_model: Optional[Model] = None
    final_model_results: Optional[ModelfitResults] = None


def calculate_results(base_model, base_model_results, linear_model, linear_model_results):
    base_iofv = base_model_results.individual_ofv
    linear_iofv = linear_model_results.individual_ofv
    if base_iofv is not None and linear_iofv is not None:
        delta_iofv = linear_iofv - base_iofv
    else:
        delta_iofv = np.nan

    iofv = pd.DataFrame(
        {
            'base': base_iofv,
            'linear': linear_iofv,
            'delta': delta_iofv,
        }
    )
    ofv = pd.Series(
        [
            base_model_results.ofv,
            linear_model_results.ofv_iterations.iloc[0],
            linear_model_results.ofv,
        ],
        index=['base', 'lin_evaluated', 'lin_estimated'],
        name='OFV',
    )
    iofv1 = base_model_results.individual_ofv
    iofv2 = linear_model_results.individual_ofv
    iofv_plot = plot_iofv_vs_iofv(iofv1, iofv2, base_model.name, linear_model.name)
    res = LinearizeResults(
        ofv=ofv,
        iofv=iofv,
        iofv_plot=iofv_plot,
        final_model=linear_model,
        final_model_results=linear_model_results,
    )
    return res


def psn_linearize_results(path):
    """Create linearize results from a PsN linearize run

    :param path: Path to PsN linearize run directory
    :return: A :class:`QAResults` object
    """
    from pharmpy.tools import read_modelfit_results

    path = Path(path)

    base_model_path = path / 'scm_dir1' / 'derivatives.mod'
    base_model = Model.parse_model(base_model_path)
    base_model_results = read_modelfit_results(base_model_path)
    lin_path = list(path.glob('*_linbase.mod'))[0]
    lin_model = Model.parse_model(lin_path)
    lin_model_results = read_modelfit_results(lin_path)

    res = calculate_results(base_model, base_model_results, lin_model, lin_model_results)
    return res
