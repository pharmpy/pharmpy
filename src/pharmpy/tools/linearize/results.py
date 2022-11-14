from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pharmpy.deps import pandas as pd
from pharmpy.model import Model, Results
from pharmpy.modeling import plot_iofv_vs_iofv


@dataclass(frozen=True)
class LinearizeResults(Results):
    ofv: Optional[float] = None
    iofv: Optional[Any] = None
    iofv_plot: Optional[Any] = None


def calculate_results(base_model, linear_model):
    baseres = base_model.modelfit_results
    linearres = linear_model.modelfit_results
    iofv = pd.DataFrame(
        {
            'base': baseres.individual_ofv,
            'linear': linearres.individual_ofv,
            'delta': linearres.individual_ofv - baseres.individual_ofv,
        }
    )
    ofv = pd.DataFrame(
        {'ofv': [baseres.ofv, linearres.ofv_iterations.iloc[0], linearres.ofv]},
        index=['base', 'lin_evaluated', 'lin_estimated'],
    )
    iofv1 = base_model.modelfit_results.individual_ofv
    iofv2 = linear_model.modelfit_results.individual_ofv
    iofv_plot = plot_iofv_vs_iofv(iofv1, iofv2, base_model.name, linear_model.name)
    res = LinearizeResults(ofv=ofv, iofv=iofv, iofv_plot=iofv_plot)
    return res


def psn_linearize_results(path):
    """Create linearize results from a PsN linearize run

    :param path: Path to PsN linearize run directory
    :return: A :class:`QAResults` object
    """
    path = Path(path)

    base_model = Model.create_model(path / 'scm_dir1' / 'derivatives.mod')
    lin_path = list(path.glob('*_linbase.mod'))[0]
    lin_model = Model.create_model(lin_path)

    res = calculate_results(base_model, lin_model)
    return res
